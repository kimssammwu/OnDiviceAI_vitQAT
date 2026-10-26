"""Frozen augmentation stays reproducible and usable by FP32/QAT/PTQ experiments."""
import copy
import hashlib
import json
import math
from bisect import bisect_right
from collections import Counter
from itertools import accumulate, groupby
from pathlib import Path

import pytest
import torch
from timm.loss import SoftTargetCrossEntropy
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from vitqat.cli import parser as training_parser
from vitqat.data import loader, make_data, make_ptq_data
from vitqat.export_augmented import main as export_main, parser as export_parser
from vitqat.fixed_data import FixedAugmentedDataset
from vitqat.models import build_model
from vitqat.quantization import Quantizer
from vitqat.train import main as train_main
from vitqat.training import read_checkpoint, save_checkpoint, seed_all, train_epoch


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    torch.set_num_threads(2)


def export_args(data_root, output, **overrides):
    args = export_parser().parse_args([
        "--data-root", str(data_root), "--output", str(output),
        "--image-size", "32", "--seed", "7", "--views-per-image", "1",
        "--shard-size", "64", "--storage-dtype", "float16",
    ])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def export(data_root, output, **overrides):
    args = export_args(data_root, output, **overrides)
    export_main(args)
    return json.loads((Path(output) / "manifest.json").read_text())


def load_fixed(root, manifest, **overrides):
    kwargs = dict(image_size=manifest["config"]["image_size"],
                  split_hash=manifest["source_split"]["sha256"], num_classes=101)
    kwargs.update(overrides)
    return FixedAugmentedDataset(root, **kwargs)


def shard_payloads(root, manifest):
    return [torch.load(Path(root) / shard["file"], map_location="cpu", weights_only=True)
            for shard in manifest["shards"]]


def all_tensors(root, manifest):
    payloads = shard_payloads(root, manifest)
    return {key: torch.cat([payload[key] for payload in payloads]) for key in payloads[0]}


def training_args(data_root, output, cache=None, **overrides):
    args = training_parser().parse_args([
        "--mode", "fp32", "--data-root", str(data_root), "--output", str(output),
        "--image-size", "32", "--seed", "7", "--device", "cpu", "--workers", "0",
        "--threads", "2", "--epochs", "1", "--warmup-epochs", "0",
        "--batch-size", "64", "--drop-path", "0",
        "--latency-warmup", "0", "--latency-repeats", "1",
    ])
    if cache is not None:
        args.fixed_train_data = str(cache)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def rewrite_manifest(root, manifest):
    """Re-sign deliberate malformed fixtures so tensor validation is exercised too."""
    unsigned = {key: value for key, value in manifest.items() if key != "sha256"}
    manifest["sha256"] = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    (Path(root) / "manifest.json").write_text(json.dumps(manifest))


def test_export_defaults_enable_the_complete_existing_augmentation_recipe(tmp_path):
    args = export_parser().parse_args(["--output", str(tmp_path / "fixed")])
    assert args.mixup == 0.8 and args.cutmix == 1.0
    assert args.random_erasing == 0.25 and args.crop_min_scale == 0.08
    assert args.seed == 42 and args.views_per_image >= 1


def test_export_shards_contain_only_training_sources_and_valid_soft_targets(food_fixture):
    cache = food_fixture / "fixed"
    manifest = export(food_fixture, cache, views_per_image=2, shard_size=37)
    tensors = all_tensors(cache, manifest)
    split = manifest["source_split"]
    train_indices, val_indices = set(split["train_indices"]), set(split["val_indices"])
    expected_count = len(train_indices) * 2
    assert manifest["format"] == "vitqat_fixed_train_v1"
    assert manifest["num_samples"] == expected_count
    assert len(manifest["shards"]) == math.ceil(expected_count / 37)
    assert sum(shard["count"] for shard in manifest["shards"]) == expected_count
    assert all(0 < shard["count"] <= 37 for shard in manifest["shards"])
    for shard in manifest["shards"]:
        assert hashlib.sha256((cache / shard["file"]).read_bytes()).hexdigest() == shard["sha256"]
    assert set(tensors["source_indices"].tolist()) == train_indices
    assert set(tensors["partner_indices"].tolist()) <= train_indices
    assert not train_indices.intersection(val_indices)
    assert not set(tensors["partner_indices"].tolist()).intersection(val_indices)
    assert Counter(zip(tensors["view_indices"].tolist(), tensors["source_indices"].tolist())) == Counter(
        (view, index) for view in range(2) for index in train_indices)
    assert tensors["images"].shape == (expected_count, 3, 32, 32)
    assert tensors["images"].dtype == torch.float16
    targets = tensors["targets"]
    assert targets.shape == (expected_count, 101) and targets.dtype == torch.float32
    assert torch.isfinite(tensors["images"]).all() and torch.isfinite(targets).all()
    assert (targets >= 0).all() and (targets <= 1).all()
    assert torch.allclose(targets.sum(dim=1), torch.ones(expected_count), atol=1e-6)
    # At least some labels really mix two classes, beyond the smoothing floor.
    assert ((targets > 0.1 / 101 + 1e-5).sum(dim=1) > 1).any()


def test_reexport_is_identical_despite_unrelated_rng_use_and_output_path(food_fixture):
    first, second = food_fixture / "one", food_fixture / "two"
    a = export(food_fixture, first)
    seed_all(931)
    torch.randn(200)
    b = export(food_fixture, second)
    assert a == b
    assert a["sha256"] == b["sha256"]
    assert all(torch.equal(value, all_tensors(second, b)[key])
               for key, value in all_tensors(first, a).items())


def test_different_seed_changes_frozen_images_and_dataset_identity(food_fixture):
    first, second = food_fixture / "seed7", food_fixture / "seed8"
    a = export(food_fixture, first, seed=7)
    b = export(food_fixture, second, seed=8)
    assert a["sha256"] != b["sha256"]
    assert not torch.equal(all_tensors(first, a)["images"], all_tensors(second, b)["images"])


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_fixed_dataset_returns_the_same_values_on_every_read(food_fixture, dtype):
    cache = food_fixture / "fixed"
    manifest = export(food_fixture, cache, storage_dtype=dtype)
    dataset = load_fixed(cache, manifest)
    expected = all_tensors(cache, manifest)
    assert len(dataset) == manifest["num_samples"]
    assert dataset.dataset_hash == manifest["sha256"]
    assert dataset.manifest == manifest and dataset.classes == manifest["source_split"]["classes"]
    # Cross the shard boundary and perturb global RNGs between reads.
    for index in (0, 63, 64, len(dataset) - 1):
        image, target = dataset[index]
        seed_all(index + 100)
        other_image, other_target = dataset[index]
        assert image.dtype == target.dtype == torch.float32
        assert torch.equal(image, expected["images"][index].float())
        assert torch.equal(target, expected["targets"][index])
        assert torch.equal(image, other_image) and torch.equal(target, other_target)


def test_worker_count_does_not_change_fixed_inputs(food_fixture, multiprocessing_tensor_sharing):
    cache = food_fixture / "fixed"
    manifest = export(food_fixture, cache)
    sample = Subset(load_fixed(cache, manifest), [0, 1, 63, 64])
    expected = list(DataLoader(sample, batch_size=2, num_workers=0))
    try:
        actual = list(DataLoader(sample, batch_size=2, num_workers=2,
                                 multiprocessing_context="spawn", timeout=20))
    except PermissionError as exc:
        pytest.skip(f"Sandbox forbids multiprocessing synchronization: {exc}")
    assert len(actual) == len(expected)
    assert all(torch.equal(a, b) for left, right in zip(expected, actual)
               for a, b in zip(left, right))


def test_fixed_shard_order_is_complete_reproducible_and_local(food_fixture):
    cache = food_fixture / "fixed"
    manifest = export(food_fixture, cache, shard_size=37)
    dataset = load_fixed(cache, manifest)
    fp32 = training_args(food_fixture, food_fixture / "fp32", cache)
    qat = copy.copy(fp32)
    qat.mode, qat.workers = "qat", 2

    def order(args, epoch):
        batches = loader(dataset, args, training=True, epoch=epoch)
        assert len(batches.sampler) == len(dataset)
        return list(batches.sampler)

    first = order(fp32, 0)
    assert sorted(first) == list(range(len(dataset)))
    assert first == order(fp32, 0) == order(qat, 0)
    assert first != order(fp32, 1)
    assert order(fp32, 1) == order(qat, 1)
    counts = [shard["count"] for shard in manifest["shards"]]
    ends = list(accumulate(counts))
    groups = [(shard, len(list(items))) for shard, items in groupby(
        bisect_right(ends, index) for index in first)]
    # Each shard appears in one uninterrupted run, including the short final
    # shard. Global image shuffling would repeatedly evict the cached shard.
    assert len(groups) == len(counts)
    assert sorted(groups) == list(enumerate(counts))


@pytest.mark.parametrize("mixup,cutmix", [(0.8, 0.0), (0.0, 1.0)])
def test_each_mixing_method_is_available_with_random_erasing(food_fixture, mixup, cutmix):
    cache = food_fixture / "fixed"
    manifest = export(food_fixture, cache, mixup=mixup, cutmix=cutmix, random_erasing=1.0)
    targets = all_tensors(cache, manifest)["targets"]
    assert ((targets > 0.1 / 101 + 1e-5).sum(dim=1) > 1).any()
    assert manifest["config"]["mixup"] == mixup and manifest["config"]["cutmix"] == cutmix
    assert manifest["config"]["random_erasing"] == 1.0


def test_make_data_uses_frozen_training_inputs_and_original_eval_inputs(food_fixture):
    cache = food_fixture / "fixed"
    manifest = export(food_fixture, cache)
    args = training_args(food_fixture, food_fixture / "out", cache)
    train, val, test, split = make_data(args)
    assert isinstance(train, FixedAugmentedDataset)
    assert args.fixed_train_hash == manifest["sha256"]
    assert split == manifest["source_split"]
    assert torch.equal(train[0][0], train[0][0]) and train[0][1].shape == (101,)
    for dataset in (val, test):
        image, target = dataset[0]
        assert image.shape == (3, 32, 32) and isinstance(target, int)
        assert torch.equal(image, dataset[0][0])


@pytest.mark.parametrize("key,value", [("image_size", 64), ("label_smoothing", 0.2),
                                       ("seed", 8), ("val_fraction", 0.5)])
def test_training_rejects_a_cache_from_different_experiment_settings(food_fixture, key, value):
    cache = food_fixture / "fixed"
    export(food_fixture, cache)
    args = training_args(food_fixture, food_fixture / "out", cache, **{key: value})
    with pytest.raises(ValueError):
        make_data(args)


@pytest.mark.parametrize("key,value", [("mixup", 0.8), ("cutmix", 1.0),
                                       ("random_erasing", 0.25), ("crop_min_scale", 0.5)])
def test_training_rejects_online_augmentation_over_frozen_inputs(food_fixture, key, value):
    cache = food_fixture / "fixed"
    export(food_fixture, cache)
    args = training_args(food_fixture, food_fixture / "out", cache, **{key: value})
    with pytest.raises(ValueError):
        make_data(args)


def test_export_never_overwrites_existing_results(food_fixture):
    cache = food_fixture / "fixed"
    cache.mkdir()
    marker = cache / "previous-result.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="[Ee]xist|empty"):
        export_main(export_args(food_fixture, cache))
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("key,value", [("views_per_image", 0), ("shard_size", 0),
                                       ("random_erasing", 2.0), ("label_smoothing", 1.0)])
def test_invalid_export_settings_fail_before_creating_cache(food_fixture, key, value):
    cache = food_fixture / "fixed"
    with pytest.raises(ValueError):
        export_main(export_args(food_fixture, cache, **{key: value}))
    assert not (cache / "manifest.json").exists()


def test_cache_rejects_wrong_requested_split_or_image_size(food_fixture):
    cache = food_fixture / "fixed"
    manifest = export(food_fixture, cache)
    for overrides in ({"split_hash": "other-split"}, {"image_size": 64}, {"num_classes": 100}):
        with pytest.raises(ValueError):
            load_fixed(cache, manifest, **overrides)


@pytest.mark.parametrize("corruption", ["manifest", "shard_hash", "image_shape", "target_shape"])
def test_cache_integrity_failures_are_rejected(food_fixture, corruption):
    cache = food_fixture / "fixed"
    manifest = export(food_fixture, cache)
    if corruption == "manifest":
        manifest["num_samples"] += 1
        (cache / "manifest.json").write_text(json.dumps(manifest))
    else:
        shard = manifest["shards"][0]
        path = cache / shard["file"]
        if corruption == "shard_hash":
            path.write_bytes(path.read_bytes() + b"unexpected data")
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            key = "images" if corruption == "image_shape" else "targets"
            payload[key] = payload[key][..., :-1]
            torch.save(payload, path)
            shard["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            rewrite_manifest(cache, manifest)
    with pytest.raises(ValueError):
        dataset = load_fixed(cache, manifest)
        dataset[0]


def test_training_soft_targets_are_not_label_smoothed_twice(tmp_path):
    args = training_args(tmp_path, tmp_path / "out", batch_size=2)
    seed_all(45)
    images = torch.randn(2, 4)
    targets = torch.full((2, 101), 0.1 / 101)
    targets[0, 3] += 0.7 * 0.9
    targets[0, 7] += 0.3 * 0.9
    targets[1, 12] += 0.9
    model = nn.Linear(4, 101)
    expected = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    other_optimizer = torch.optim.SGD(expected.parameters(), lr=args.lr)
    loss = SoftTargetCrossEntropy()(expected(images), targets)
    loss.backward()
    nn.utils.clip_grad_norm_(expected.parameters(), args.clip_grad)
    other_optimizer.step()
    result = train_epoch(model, DataLoader(TensorDataset(images, targets), batch_size=2),
                         optimizer, args, 0, 1)
    assert result["train_loss"] == pytest.approx(loss.item())
    assert all(torch.equal(left, right) for left, right in zip(model.parameters(), expected.parameters()))


def test_checkpoint_identity_follows_cache_contents_instead_of_path(tmp_path):
    args = training_args(tmp_path, tmp_path / "out", tmp_path / "fixed")
    args.fixed_train_hash = "same-content"
    model = nn.Linear(4, 101)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 0, 1.0, args, "split")
    other = copy.copy(args)
    other.fixed_train_data = str(tmp_path / "copied-cache")
    read_checkpoint(path, model, other, "split")
    other.fixed_train_hash = "different-content"
    with pytest.raises(ValueError, match="fixed_train_hash"):
        read_checkpoint(path, model, other, "split")


def test_separate_exporter_provenance_is_compatible(food_fixture):
    cache = food_fixture / "server-cache"
    manifest = export(food_fixture, cache, storage_dtype="float32")
    manifest["upstream_source_sha256"] = {"augment.py": "a" * 64, "data.py": "b" * 64}
    rewrite_manifest(cache, manifest)
    # The existing server export also has progress/verification/preview files.
    (cache / "verification.json").write_text('{"status": "ready"}')
    args = training_args(food_fixture, food_fixture / "out", cache)
    train, _, _, _ = make_data(args)
    assert args.fixed_train_hash == manifest["sha256"]
    assert train[0][0].dtype == torch.float32
    assert train[0][1].shape == (101,)


def test_qat_accepts_the_same_frozen_soft_targets(food_fixture):
    cache = food_fixture / "fixed"
    manifest = export(food_fixture, cache)
    args = training_args(food_fixture, food_fixture / "qat", cache, mode="qat", batch_size=2)
    data = Subset(load_fixed(cache, manifest), [0, 1, 2, 3])
    model = build_model(args)
    assert any(isinstance(module, Quantizer) for module in model.modules())
    stats = train_epoch(model, DataLoader(data, batch_size=2),
                        torch.optim.AdamW(model.parameters(), lr=args.lr), args, 0, 2)
    assert math.isfinite(stats["train_loss"]) and stats["optimizer_steps"] == 2


def test_legacy_checkpoint_without_cache_identity_remains_usable(tmp_path):
    args = training_args(tmp_path, tmp_path / "out")
    model = nn.Linear(4, 101)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, model, optimizer, 0, 1.0, args, "split")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["config"].pop("fixed_train_hash", None)
    payload["config"].pop("fixed_train_data", None)
    torch.save(payload, path)
    read_checkpoint(path, model, args, "split")
    args.fixed_train_hash = "now-using-fixed-data"
    with pytest.raises(ValueError, match="fixed_train_hash"):
        read_checkpoint(path, model, args, "split")


def test_frozen_cropr_fp32_training_can_feed_ptq_without_the_cache(food_fixture):
    cache = food_fixture / "fixed"
    manifest = export(food_fixture, cache, image_size=64)
    args = training_args(food_fixture, food_fixture / "fp32", cache,
                         image_size=64, use_cropr=True, cropr_pruning_rate=1)
    trained = train_main(args)
    assert trained["test_samples"] == 101 and math.isfinite(trained["test_loss"])
    saved_config = json.loads((Path(args.output) / "config.json").read_text())
    assert saved_config["fixed_train_hash"] == manifest["sha256"]
    checkpoint = Path(args.output) / "best.pt"
    # The checkpoint records a cache that is no longer mounted on the PTQ machine.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["config"]["fixed_train_data"] = str(food_fixture / "unavailable-cache")
    torch.save(state, checkpoint)
    effective = copy.copy(args)
    effective.fixed_train_data = str(food_fixture / "unavailable-cache")
    calibration, val, test, split = make_ptq_data(effective)
    assert split == manifest["source_split"]
    assert calibration.indices == split["train_indices"]
    assert isinstance(calibration[0][1], int)
    assert torch.equal(calibration[0][0], calibration[0][0])
    assert not set(calibration.indices).intersection(val.indices)
    assert len(test) == 101
    ptq = training_args(food_fixture, food_fixture / "ptq", mode="ptq",
                        fp32_checkpoint=str(checkpoint), calibration_samples=2)
    metrics = train_main(ptq)
    assert metrics["mode"] == "ptq" and metrics["weight_bits"] == metrics["activation_bits"] == 2
    assert metrics["test_samples"] == 101 and metrics["split_hash"] == split["sha256"]
    assert math.isfinite(metrics["test_loss"])
