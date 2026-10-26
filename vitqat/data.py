import hashlib
import json
import random
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .augment import AUGMENT_DEFAULTS, build_train_transform
from .constants import NUM_CLASSES


def stratified_split(labels, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    train, val = [], []
    for label in sorted(set(labels)):
        indices = [i for i, value in enumerate(labels) if value == label]
        if len(indices) < 2:
            raise ValueError("Each class needs at least two training images")
        indices = torch.tensor(indices)[torch.randperm(len(indices), generator=generator)].tolist()
        n = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
        val.extend(indices[:n])
        train.extend(indices[n:])
    return sorted(train), sorted(val)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_data(args):
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # train 증강은 augment.py가 조립한다(옵션을 안 주면 기존 transform과 동일).
    # 아래 evaluation은 val/test용이라 증강을 넣지 않는다.
    training = build_train_transform(args.image_size, args.crop_min_scale, args.random_erasing, norm)
    evaluation = transforms.Compose([
        transforms.Resize(round(args.image_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size), transforms.ToTensor(), norm,
    ])
    train_data = datasets.Food101(args.data_root, split="train", transform=training, download=args.download)
    val_data = datasets.Food101(args.data_root, split="train", transform=evaluation)
    test_data = datasets.Food101(args.data_root, split="test", transform=evaluation)
    if len(train_data.classes) != NUM_CLASSES or train_data.classes != test_data.classes:
        raise ValueError("Food-101 requires matching 101-class train/test mappings")
    train_indices, val_indices = stratified_split(train_data._labels, args.val_fraction, args.seed)
    manifest = dict(train_indices=train_indices, val_indices=val_indices,
                    classes=train_data.classes, seed=args.seed, val_fraction=args.val_fraction,
                    official_train_size=len(train_data), official_test_size=len(test_data))
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    train_split = Subset(train_data, train_indices)
    args.fixed_train_hash = None
    if getattr(args, "fixed_train_data", None):
        from .fixed_data import FixedAugmentedDataset
        for key, default in AUGMENT_DEFAULTS.items():
            if getattr(args, key) != default:
                raise ValueError(f"--fixed-train-data already contains augmentation; leave --{key.replace('_', '-')} at its default")
        fixed = FixedAugmentedDataset(args.fixed_train_data, image_size=args.image_size,
                                      split_hash=manifest["sha256"], num_classes=NUM_CLASSES)
        if fixed.classes != manifest["classes"]:
            raise ValueError("Fixed training data class mapping does not match Food-101")
        if fixed.manifest["config"]["label_smoothing"] != args.label_smoothing:
            raise ValueError("Fixed training data label-smoothing does not match --label-smoothing")
        args.fixed_train_hash = fixed.dataset_hash
        train_split = fixed
    return train_split, Subset(val_data, val_indices), test_data, manifest


class _FixedShardSampler(Sampler):
    """Keep shuffled reads local to a shard instead of reloading one per image."""
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        counts = [shard["count"] for shard in self.dataset.manifest["shards"]]
        offsets, total = [], 0
        for count in counts:
            offsets.append(total)
            total += count
        for shard in torch.randperm(len(counts), generator=generator).tolist():
            for index in torch.randperm(counts[shard], generator=generator).tolist():
                yield offsets[shard] + index


def loader(dataset, args, training=False, epoch=0):
    sampler = None
    if training and getattr(args, "fixed_train_data", None):
        from .fixed_data import FixedAugmentedDataset
        if isinstance(dataset, FixedAugmentedDataset):
            sampler = _FixedShardSampler(dataset, args.seed + epoch)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=training and sampler is None,
                      sampler=sampler,
                      num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
                      multiprocessing_context="spawn" if args.workers else None,
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(args.seed + epoch))


def make_ptq_data(args):
    """Reuse the training split with deterministic evaluation preprocessing."""
    raw_args = copy.copy(args)
    raw_args.fixed_train_data = None
    raw_args.fixed_train_hash = None
    _, val_data, test_data, manifest = make_data(raw_args)
    calibration_data = Subset(val_data.dataset, manifest["train_indices"])
    return calibration_data, val_data, test_data, manifest
