"""Experiment 2-a: calibrate a trained FP32 Cropr without updating its weights.

Uses uniform affine fake quantization, not native INT2/INT4 kernels or packed export.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import platform
from pathlib import Path

import timm
import torch
import torchvision
from torch.ao.quantization import FakeQuantize, MinMaxObserver, PerChannelMinMaxObserver
from torch.utils.data import Subset

from .data import loader, make_ptq_data
from .models import build_model
from .quantization import Quantizer, freeze_observers, prepare_qat
from .training import evaluate, seed_all


def prepare_ptq(model, weight_bits, activation_bits, scope="all"):
    """Reuse QAT targets/parameters, collecting cumulative ranges for static PTQ."""
    prepare_qat(model, weight_bits, activation_bits, scope)
    for module in model.modules():
        if isinstance(module, Quantizer):
            previous = module.fake_quant
            kwargs = dict(quant_min=0, quant_max=2**module.bits - 1,
                          dtype=torch.quint8, qscheme=previous.qscheme)
            if previous.qscheme == torch.per_channel_affine:
                kwargs.update(observer=PerChannelMinMaxObserver, ch_axis=0)
            else:
                kwargs.update(observer=MinMaxObserver)
            module.fake_quant = FakeQuantize(**kwargs).to(previous.scale.device)
    return model


@torch.no_grad()
def calibrate(model, batches, device, max_samples):
    """Observe FP32 inference activations; Cropr/dropout stay in evaluation mode."""
    if max_samples < 1:
        raise ValueError("calibration-samples must be positive")
    quantizers = [(name, m) for name, m in model.named_modules() if isinstance(m, Quantizer)]
    if not quantizers:
        raise ValueError("Prepare PTQ quantizers before calibration")
    model.eval()
    visited = set()
    hooks = [module.register_forward_hook(
        lambda module, inputs, output, name=name: visited.add(name))
        for name, module in quantizers]
    for _, module in quantizers:
        observer = module.fake_quant.activation_post_process
        observer.reset_min_max_vals()
        # PyTorch 2.3 per-channel reset creates empty CPU buffers even on CUDA.
        observer.to(module.fake_quant.scale.device)
        module.calibrating = True
        module.frozen = False
        module.fake_quant.disable_fake_quant()
    samples, count = 0, 0
    try:
        for images, _ in batches:
            images = images[:max_samples - samples]
            if not len(images):
                break
            model(images.to(device))
            samples += len(images)
            count += 1
            if samples == max_samples:
                break
        if not samples:
            raise ValueError("Calibration dataset is empty")
    finally:
        for hook in hooks:
            hook.remove()
        for _, module in quantizers:
            module.calibrating = False
            module.frozen = True
            module.fake_quant.disable_observer()
            module.fake_quant.enable_fake_quant()
    # Cropr auxiliary branches are training-only and deliberately never calibrated.
    observed = [name for name, m in quantizers
                if m.fake_quant.activation_post_process.min_val.numel()
                and torch.isfinite(m.fake_quant.activation_post_process.min_val).all()
                and torch.isfinite(m.fake_quant.activation_post_process.max_val).all()]
    if not visited or visited.difference(observed):
        raise ValueError("Calibration produced missing/non-finite observer ranges for inference")
    return dict(samples=samples, batches=count, observed_quantizers=observed,
                unobserved_quantizers=[name for name, _ in quantizers if name not in observed])


def load_ptq_checkpoint(path, device="cpu"):
    """Restore calibrated inference state without the original FP32 checkpoint/data."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "cropr_ptq_v1":
        raise ValueError("Expected a calibrated Cropr PTQ checkpoint (ptq.pt)")
    args = argparse.Namespace(**checkpoint["config"])
    model = build_model(args)
    prepare_ptq(model, args.weight_bits, args.activation_bits, args.ptq_scope)
    model.load_state_dict(checkpoint["model"], strict=True)
    freeze_observers(model, True)
    return model.to(device).eval()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _source_settings(args, checkpoint):
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("FP32 checkpoint must contain its training config")
    if config.get("mode") != "fp32" or not config.get("use_cropr"):
        raise ValueError("PTQ experiment 2-a requires an FP32 Cropr checkpoint")
    settings = argparse.Namespace(**{**vars(args), **config})
    for name in ("output", "data_root", "download", "device", "workers", "batch_size", "threads",
                 "latency_warmup", "latency_repeats", "fp32_checkpoint", "calibration_samples",
                 "ptq_scope", "weight_bits", "activation_bits"):
        setattr(settings, name, getattr(args, name))
    settings.mode = "ptq"
    settings.experiment = args.experiment or "2-a"
    settings.resume = settings.eval_checkpoint = None
    return settings


def run_ptq(args, *, data_factory=make_ptq_data, evaluate_fn=evaluate, measure_model=None):
    """Load an FP32 training result, calibrate on train only, and compare test metrics."""
    if not args.fp32_checkpoint:
        raise ValueError("--mode ptq requires --fp32-checkpoint")
    if args.resume or args.eval_checkpoint:
        raise ValueError("PTQ uses --fp32-checkpoint, not resume or eval-checkpoint")
    if args.calibration_samples < 1:
        raise ValueError("calibration-samples must be positive")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid batch size or workers")
    if args.latency_warmup < 0 or args.latency_repeats < 1:
        raise ValueError("Require latency_repeats >= 1 and latency_warmup >= 0")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory for PTQ")
    source = Path(args.fp32_checkpoint)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    settings = _source_settings(args, checkpoint)
    if settings.threads > 0:
        torch.set_num_threads(settings.threads)
    seed_all(settings.seed)
    calibration_data, _, test_data, manifest = data_factory(settings)
    if checkpoint["split_hash"] != manifest["sha256"]:
        raise ValueError("Checkpoint data split does not match")
    if not len(calibration_data):
        raise ValueError("Calibration dataset is empty")
    generator = torch.Generator().manual_seed(settings.seed)
    indices = torch.randperm(len(calibration_data), generator=generator)[:settings.calibration_samples].tolist()
    selected = Subset(calibration_data, indices)
    original_indices = ([calibration_data.indices[i] for i in indices]
                        if isinstance(calibration_data, Subset) else indices)
    model = build_model(settings).to(settings.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if measure_model is None:
        from .train import measure_model
    baseline_args = copy.copy(settings)
    baseline_args.mode = "fp32"
    baseline = measure_model(model, baseline_args)
    baseline_test = evaluate_fn(model, loader(test_data, settings), settings.device)
    baseline.update(top1_acc=baseline_test["top1_acc"], test_loss=baseline_test["loss"],
                    test_samples=baseline_test["samples"])
    with io.BytesIO() as state:
        torch.save(model.state_dict(), state)
        baseline["storage_serialized_state_bytes"] = state.tell()
    prepare_ptq(model, settings.weight_bits, settings.activation_bits, settings.ptq_scope)
    # Quantizers are newly attached on CPU even if the trained model is on CUDA.
    model.to(settings.device)
    calibration = calibrate(model, loader(selected, settings), settings.device, settings.calibration_samples)
    calibration.update(split="train", preprocessing="evaluation", seed=settings.seed,
                       requested_samples=settings.calibration_samples, indices=original_indices,
                       observer="cumulative_minmax", fake_quant_during_calibration=False,
                       weight_updates=0, scope=settings.ptq_scope)
    test = evaluate_fn(model, loader(test_data, settings), settings.device)
    metrics = measure_model(model, settings)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    source_hash = digest.hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model_state.pt")
    torch.save(dict(format="cropr_ptq_v1", model=model.state_dict(), config=vars(settings),
                    calibration=calibration, split_hash=manifest["sha256"],
                    source_checkpoint=str(source.resolve()), source_sha256=source_hash,
                    selected_epoch=checkpoint["epoch"]), output / "ptq.pt")
    metrics.update(experiment=settings.experiment, top1_acc=test["top1_acc"],
                   test_loss=test["loss"], test_samples=test["samples"],
                   fp32_top1_acc=baseline["top1_acc"],
                   top1_drop_pp=baseline["top1_acc"] - test["top1_acc"],
                   selected_epoch=checkpoint["epoch"], best_val_top1_acc=checkpoint["best_acc"],
                   checkpoint=str((output / "ptq.pt").resolve()),
                   source_checkpoint=str(source.resolve()), source_sha256=source_hash,
                   split_hash=manifest["sha256"], calibration_samples=calibration["samples"],
                   calibration_batches=calibration["batches"], weight_updates=0,
                   quantization_method="static_uniform_affine_fake_quant",
                   exported_packed_integer_model=False,
                   storage_model_state_bytes=(output / "model_state.pt").stat().st_size,
                   seed=settings.seed, device=settings.device,
                   device_name=torch.cuda.get_device_name(settings.device)
                   if settings.device.startswith("cuda") else platform.processor(),
                   torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                   timm_version=timm.__version__, python_version=platform.python_version(),
                   cpu_threads=torch.get_num_threads(), image_size=settings.image_size,
                   pretrained=False, amp=False, tf32=False,
                   fixed_train_hash=getattr(settings, "fixed_train_hash", None))
    _write_json(output / "config.json", vars(settings))
    _write_json(output / "split.json", manifest)
    _write_json(output / "calibration.json", calibration)
    _write_json(output / "fp32_metrics.json", baseline)
    _write_json(output / "metrics.json", metrics)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics
