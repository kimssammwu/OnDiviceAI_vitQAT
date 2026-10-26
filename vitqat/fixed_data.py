"""Materialize training augmentation once and reuse the exact input tensors.

The cache keeps normalized floating point tensors and soft targets: writing JPEG
or PNG files would lose both Mixup labels and floating point augmentation values.
"""
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
import PIL
import timm
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

from .augment import augment_batch, build_mixup, build_train_transform, validate_augment_args
from .constants import NUM_CLASSES


CACHE_FORMAT = "vitqat_fixed_train_v1"


def _canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(seed, kind, view, index=0):
    digest = hashlib.sha256(f"{seed}:{kind}:{view}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


@contextmanager
def _seeded_rng(seed):
    """Seed augmentation on CPU without consuming the caller's random streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_config(args):
    validate_augment_args(args)
    for key in ("image_size", "views_per_image", "shard_size", "threads"):
        _positive_integer(getattr(args, key), key)
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    if not 0 < args.val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int):
        raise ValueError("seed must be an integer")
    if args.storage_dtype not in ("float32", "float16"):
        raise ValueError("storage_dtype must be float32 or float16")


def _validate_source_split(source, num_classes):
    if not isinstance(source, dict):
        raise ValueError("Fixed training data must include its source split")
    classes = source.get("classes")
    if (not isinstance(classes, list) or len(classes) != num_classes
            or any(not isinstance(item, str) or not item for item in classes)
            or len(set(classes)) != num_classes):
        raise ValueError("Fixed training data has an invalid class mapping")
    unhashed = {key: value for key, value in source.items() if key != "sha256"}
    expected = hashlib.sha256(json.dumps(unhashed, sort_keys=True).encode()).hexdigest()
    if source.get("sha256") != expected:
        raise ValueError("Fixed training data source split hash does not match its contents")
    train, val = source.get("train_indices"), source.get("val_indices")
    size = source.get("official_train_size")
    if (not isinstance(size, int) or isinstance(size, bool) or size < 1
            or not isinstance(train, list) or not train or not isinstance(val, list)
            or any(not isinstance(index, int) or isinstance(index, bool)
                   or not 0 <= index < size for index in train + val)
            or len(set(train)) != len(train) or len(set(val)) != len(val)
            or set(train) & set(val)):
        raise ValueError("Fixed training data has invalid or overlapping source split indices")
    return classes


def export_fixed_data(args):
    """Export the training partition only; return the completed manifest.

    Each view has a fixed pairing permutation. Both members receive independently
    seeded image transforms, then the existing batch augmentation handles the
    pair. With both alphas enabled, timm selects Mixup or CutMix for each pair.
    """
    _validate_config(args)

    output = Path(args.output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Export output must be an empty directory; use a new output path")
    output.mkdir(parents=True, exist_ok=True)

    # Import lazily: data.make_data also imports the cache loader for training.
    from .data import make_data
    source_args = copy.copy(args)
    source_args.fixed_train_data = None
    source_args.fixed_train_hash = None
    train, _, _, source_split = make_data(source_args)
    classes = _validate_source_split(source_split, NUM_CLASSES)
    train.dataset.transform = None
    indices = list(train.indices)
    if indices != source_split["train_indices"]:
        raise ValueError("Training dataset indices do not match the source split manifest")

    dtype = getattr(torch, args.storage_dtype)
    num_samples = len(indices) * args.views_per_image
    image_bytes = args.image_size * args.image_size * 3 * torch.empty((), dtype=dtype).element_size()
    expected_bytes = num_samples * (image_bytes + NUM_CLASSES * 4 + 3 * 8)
    print(f"Exporting {num_samples} fixed training inputs; tensor payload approximately "
          f"{expected_bytes / 1024**3:.2f} GiB ({args.storage_dtype}).", flush=True)
    if shutil.disk_usage(output).free < expected_bytes * 1.02 + 1024**2:
        raise ValueError("Not enough free disk space for the fixed training tensor payload")

    config = {key: getattr(args, key) for key in (
        "seed", "val_fraction", "image_size", "label_smoothing", "mixup", "cutmix",
        "random_erasing", "crop_min_scale", "views_per_image", "shard_size",
        "storage_dtype", "threads",
    )}
    manifest = dict(format=CACHE_FORMAT, config=config, source_split=source_split,
                    num_classes=len(classes), num_samples=num_samples, shards=[],
                    versions=dict(torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                                  timm=timm.__version__, numpy=np.__version__, pillow=PIL.__version__))
    transform = build_train_transform(args.image_size, args.crop_min_scale, args.random_erasing,
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    mixup_fn = build_mixup(args.mixup, args.cutmix, args.label_smoothing)
    pending = {key: [] for key in ("images", "targets", "source_indices", "partner_indices", "view_indices")}

    def flush():
        count = len(pending["images"])
        if not count:
            return
        payload = dict(images=torch.stack(pending["images"]), targets=torch.stack(pending["targets"]),
                       **{key: torch.tensor(pending[key], dtype=torch.int64)
                          for key in ("source_indices", "partner_indices", "view_indices")})
        filename = f"shard-{len(manifest['shards']):05d}.pt"
        path = output / filename
        temporary = output / (filename + ".part")
        torch.save(payload, temporary)
        temporary.replace(path)
        manifest["shards"].append(dict(file=filename, count=count, sha256=_file_hash(path)))
        for values in pending.values():
            values.clear()

    old_threads = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    try:
        for view in range(args.views_per_image):
            generator = torch.Generator().manual_seed(_seed(args.seed, "pair-order", view))
            order = torch.randperm(len(indices), generator=generator).tolist()
            for start in range(0, len(order), 2):
                pair = [indices[position] for position in order[start:start + 2]]
                images, labels = [], []
                for original_index in pair:
                    with _seeded_rng(_seed(args.seed, "image", view, original_index)):
                        image, label = train.dataset[original_index]
                        image = transform(image)
                    if not 0 <= int(label) < NUM_CLASSES:
                        raise ValueError("Source label is outside the Food-101 class mapping")
                    images.append(image)
                    labels.append(int(label))
                images, labels = torch.stack(images), torch.tensor(labels, dtype=torch.int64)
                with _seeded_rng(_seed(args.seed, "mix", view, start // 2)):
                    images, targets = augment_batch(mixup_fn, images, labels, args.label_smoothing)
                if targets.ndim == 1:
                    targets = torch.nn.functional.one_hot(targets, NUM_CLASSES).float()
                    targets = targets * (1 - args.label_smoothing) + args.label_smoothing / NUM_CLASSES
                for offset, original_index in enumerate(pair):
                    pending["images"].append(images[offset].to(dtype).clone())
                    pending["targets"].append(targets[offset].float().clone())
                    pending["source_indices"].append(original_index)
                    pending["partner_indices"].append(pair[-1 - offset])
                    pending["view_indices"].append(view)
                    if len(pending["images"]) == args.shard_size:
                        flush()
        flush()
    finally:
        torch.set_num_threads(old_threads)
    manifest["sha256"] = _canonical_hash(manifest)
    temporary_manifest = output / "manifest.json.part"
    temporary_manifest.write_text(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False,
                                             allow_nan=False) + "\n", encoding="utf-8")
    temporary_manifest.replace(output / "manifest.json")
    print(f"Fixed training data ready: {output}\nDataset SHA-256: {manifest['sha256']}", flush=True)
    return manifest


class FixedAugmentedDataset(Dataset):
    """Read a completed cache, checking shard integrity before safe tensor loading."""

    def __init__(self, root, *, image_size, split_hash, num_classes=NUM_CLASSES):
        self.root = Path(root).expanduser().resolve()
        path = self.root / "manifest.json"
        if not path.is_file():
            raise ValueError("Fixed training data is incomplete: manifest.json is missing")
        try:
            self.manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("Cannot read the fixed training data manifest") from error
        if not isinstance(self.manifest, dict) or self.manifest.get("format") != CACHE_FORMAT:
            raise ValueError("Unsupported fixed training data format")
        content = {key: value for key, value in self.manifest.items() if key != "sha256"}
        if self.manifest.get("sha256") != _canonical_hash(content):
            raise ValueError("Fixed training data manifest hash mismatch")
        self.dataset_hash = self.manifest["sha256"]
        self.classes = _validate_source_split(self.manifest.get("source_split"), num_classes)
        if self.manifest["source_split"]["sha256"] != split_hash:
            raise ValueError("Fixed training data source split does not match the current split")
        config = self.manifest.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("Invalid fixed training data configuration")
        try:
            _validate_config(SimpleNamespace(**config))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Invalid fixed training data configuration") from error
        if (config["seed"] != self.manifest["source_split"].get("seed")
                or config["val_fraction"] != self.manifest["source_split"].get("val_fraction")):
            raise ValueError("Fixed training data configuration does not match its source split")
        if config.get("image_size") != image_size:
            raise ValueError("Fixed training data image_size does not match the model input")
        if self.manifest.get("num_classes") != num_classes:
            raise ValueError("Fixed training data num_classes does not match the model")
        self.image_size, self.num_classes = image_size, num_classes
        self.dtype = getattr(torch, config["storage_dtype"])
        self.shards = self.manifest.get("shards")
        if not isinstance(self.shards, list) or not self.shards:
            raise ValueError("Fixed training data must contain at least one shard")
        self._ends, names, total = [], set(), 0
        for shard in self.shards:
            if not isinstance(shard, dict):
                raise ValueError("Invalid fixed training data shard record")
            name = shard.get("file")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Fixed training data shard paths must be unique relative paths")
            relative = Path(name)
            target = (self.root / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(self.root):
                raise ValueError("Fixed training data shard path escapes the data directory")
            if not target.is_file():
                raise ValueError(f"Fixed training data shard is missing: {name}")
            _positive_integer(shard.get("count"), "shard count")
            if shard["count"] > config["shard_size"]:
                raise ValueError("Fixed training data shard exceeds its configured size")
            digest = shard.get("sha256")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError("Invalid fixed training data shard hash")
            total += shard["count"]
            self._ends.append(total)
            names.add(name)
        expected_count = len(self.manifest["source_split"]["train_indices"]) * config["views_per_image"]
        if self.manifest.get("num_samples") != total or total != expected_count:
            raise ValueError("Fixed training data sample counts do not match the source split")
        self._length = total
        self._cached_index, self._cached_shard = None, None
        self._train_indices = set(self.manifest["source_split"]["train_indices"])

    def __len__(self):
        return self._length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cached_index"], state["_cached_shard"] = None, None
        return state

    def _read_shard(self, index):
        shard = self.shards[index]
        path = self.root / shard["file"]
        if _file_hash(path) != shard["sha256"]:
            raise ValueError(f"Fixed training data shard hash mismatch: {shard['file']}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError(f"Cannot safely load fixed training data shard: {shard['file']}") from error
        count = shard["count"]
        if not isinstance(payload, dict):
            raise ValueError("Fixed training data shard must contain a tensor dictionary")
        shapes = dict(images=(count, 3, self.image_size, self.image_size),
                      targets=(count, self.num_classes), source_indices=(count,),
                      partner_indices=(count,), view_indices=(count,))
        for key, shape in shapes.items():
            tensor = payload.get(key)
            dtype = self.dtype if key == "images" else torch.float32 if key == "targets" else torch.int64
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"Invalid fixed training data shard tensor: {key}")
        images, targets = payload["images"], payload["targets"]
        if not torch.isfinite(images).all() or not torch.isfinite(targets).all():
            raise ValueError("Fixed training data contains nonfinite tensors")
        if ((targets < 0).any() or (targets > 1).any()
                or not torch.allclose(targets.sum(dim=1), torch.ones(count), atol=1e-5, rtol=1e-5)):
            raise ValueError("Fixed training data soft targets must be probability distributions")
        for key in ("source_indices", "partner_indices"):
            if any(value not in self._train_indices for value in payload[key].tolist()):
                raise ValueError("Fixed training data includes indices outside its training split")
        views = payload["view_indices"]
        if (views < 0).any() or (views >= self.manifest["config"]["views_per_image"]).any():
            raise ValueError("Fixed training data contains an invalid view index")
        self._cached_index, self._cached_shard = index, payload

    def __getitem__(self, index):
        from bisect import bisect_right
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("Fixed training data index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Fixed training data index out of range")
        shard_index = bisect_right(self._ends, index)
        if self._cached_index != shard_index:
            self._read_shard(shard_index)
        offset = index - (self._ends[shard_index - 1] if shard_index else 0)
        # Copies protect the cache from in-place transforms or batch augmentation.
        return (self._cached_shard["images"][offset].float().clone(),
                self._cached_shard["targets"][offset].clone())
