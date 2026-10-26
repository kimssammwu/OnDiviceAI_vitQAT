"""Experiment 2-a: calibrate a Cropr FP32 checkpoint without training it."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vitqat.cli import parser
from vitqat.data import make_data, make_ptq_data
from vitqat.models import build_model
from vitqat.ptq import calibrate, load_ptq_checkpoint, prepare_ptq, run_ptq
from vitqat.quantization import Quantizer
from vitqat.train import main
from vitqat.training import evaluate, save_checkpoint, seed_all


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    torch.set_num_threads(2)


def source_args(tmp_path, *, llf=False):
    args = parser().parse_args([
        "--mode", "fp32", "--use-cropr", "--cropr-pruning-rate", "1",
        "--image-size", "64", "--device", "cpu", "--epochs", "2",
        "--warmup-epochs", "0", "--batch-size", "2", "--workers", "0",
        "--seed", "7", "--drop-path", "0", "--output", str(tmp_path / "fp32"),
    ])
    args.cropr_llf = llf
    return args


def save_source(tmp_path, args=None, split_hash="ptq-fixture"):
    args = args or source_args(tmp_path)
    seed_all(args.seed)
    model = build_model(args)
    path = tmp_path / "fp32.pt"
    save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), 1, 12.0, args, split_hash)
    return path, model, args


def ptq_args(tmp_path, checkpoint, *options):
    return parser().parse_args([
        "--mode", "ptq", "--fp32-checkpoint", str(checkpoint),
        "--device", "cpu", "--workers", "0", "--batch-size", "3",
        "--threads", "2", "--latency-warmup", "0", "--latency-repeats", "1",
        "--calibration-samples", "3", "--output", str(tmp_path / "ptq"), *options,
    ])


def fixture_datasets():
    generator = torch.Generator().manual_seed(81)
    return tuple(TensorDataset(torch.randn(count, 3, 64, 64, generator=generator),
                               torch.arange(count) % 101) for count in (5, 2, 2))


def observer_buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()
            if name.endswith(("scale", "zero_point", "min_val", "max_val"))}


class CalibrationProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.head = nn.Linear(2, 2, bias=False)
        self.head.weight.data.copy_(torch.eye(2))
        self.calls = []

    def forward(self, images):
        quantizers = [module for module in self.modules() if isinstance(module, Quantizer)]
        self.calls.append((self.training, torch.is_grad_enabled(), images.detach().clone(),
                           [module.fake_quant.fake_quant_enabled.item() for module in quantizers]))
        return self.head(images)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("bits", [2, 4])
def test_cuda_ptq_calibration_preserves_observer_device(bits):
    model = prepare_ptq(CalibrationProbe(), bits, bits).cuda()
    images = torch.tensor([[-2.0, 1.0], [1.0, 3.0]])
    result = calibrate(model, [(images, torch.zeros(2, dtype=torch.long))], "cuda", 2)
    assert result["samples"] == 2
    assert all(buffer.device.type == "cuda" for buffer in model.buffers())
    with torch.inference_mode():
        assert torch.isfinite(model(images.cuda())).all()


def test_ptq_cli_defaults_and_four_bit_selection(tmp_path):
    args = ptq_args(tmp_path, tmp_path / "fp32.pt")
    defaults = parser().parse_args([
        "--mode", "ptq", "--fp32-checkpoint", "fp32.pt", "--output", "ptq",
    ])
    assert defaults.weight_bits == defaults.activation_bits == 2
    assert defaults.calibration_samples == 1024 and defaults.ptq_scope == "all"
    assert args.mode == "ptq"
    args = ptq_args(tmp_path, tmp_path / "fp32.pt", "--weight-bits", "4", "--activation-bits", "4")
    assert args.weight_bits == args.activation_bits == 4


def test_calibration_accumulates_extrema_caps_samples_and_never_trains():
    model = prepare_ptq(CalibrationProbe(), 2, 2)
    parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    images = torch.tensor([[-10.0, 2.0], [1.0, 3.0], [-2.0, 10.0], [1000.0, 1000.0]])
    batches = DataLoader(TensorDataset(images, torch.zeros(4, dtype=torch.long)), batch_size=2)
    result = calibrate(model, batches, "cpu", max_samples=3)
    assert result["samples"] == 3 and result["batches"] == 2
    assert result["observed_quantizers"]
    assert all(not training and not grad for training, grad, _, _ in model.calls)
    assert all(enabled == 0 for _, _, _, flags in model.calls for enabled in flags)
    assert torch.equal(torch.cat([call[2] for call in model.calls]), images[:3])
    observer = model.head.input_quant.fake_quant.activation_post_process
    assert observer.min_val.item() == -10 and observer.max_val.item() == 10
    assert not model.training
    assert all(torch.equal(parameters[name], parameter) and parameter.grad is None
               for name, parameter in model.named_parameters())
    quantizers = [module for module in model.modules() if isinstance(module, Quantizer)]
    assert all(module.fake_quant.fake_quant_enabled.item() == 1 for module in quantizers)
    assert all(module.fake_quant.observer_enabled.item() == 0 for module in quantizers)
    frozen = observer_buffers(model)
    model(torch.tensor([[-10000.0, 10000.0]]))
    buffers = dict(model.named_buffers())
    assert all(torch.equal(value, buffers[name]) for name, value in frozen.items())


@pytest.mark.parametrize("max_samples", [0, -1])
def test_calibration_requires_positive_sample_limit(max_samples):
    model = prepare_ptq(CalibrationProbe(), 2, 2)
    with pytest.raises(ValueError, match="positive|samples"):
        calibrate(model, [], "cpu", max_samples=max_samples)


def test_calibration_rejects_empty_data():
    model = prepare_ptq(CalibrationProbe(), 2, 2)
    with pytest.raises(ValueError, match="[Ee]mpty|samples"):
        calibrate(model, [], "cpu", max_samples=1)


@pytest.mark.parametrize("field,value,match", [
    ("fp32_checkpoint", None, "fp32-checkpoint"),
    ("resume", "last.pt", "resume"),
    ("eval_checkpoint", "best.pt", "eval-checkpoint"),
    ("calibration_samples", 0, "positive"),
])
def test_invalid_ptq_request_fails_before_loading_data(tmp_path, field, value, match):
    args = ptq_args(tmp_path, tmp_path / "missing.pt")
    setattr(args, field, value)
    with pytest.raises(ValueError, match=match):
        main(args)


@pytest.mark.parametrize("bits,scope,llf", [(2, "all", False), (4, "all", True),
                                          (2, "backbone", True), (4, "backbone", False)])
def test_cropr_ptq_pipeline_preserves_weights_split_and_reload(tmp_path, bits, scope, llf):
    checkpoint, original, source = save_source(tmp_path, source_args(tmp_path, llf=llf))
    args = ptq_args(tmp_path, checkpoint, "--weight-bits", str(bits), "--activation-bits", str(bits),
                    "--ptq-scope", scope, "--data-root", str(tmp_path / "runtime-data"))
    datasets = fixture_datasets()
    effective = []
    evaluations = []

    def data_factory(config):
        effective.append(vars(config).copy())
        return *datasets, {"sha256": "ptq-fixture"}

    def evaluate_fn(model, batches, device):
        assert not model.training
        assert batches.dataset is datasets[2]
        before = observer_buffers(model)
        result = evaluate(model, batches, device)
        buffers = dict(model.named_buffers())
        assert all(torch.equal(value, buffers[name]) for name, value in before.items())
        evaluations.append(result)
        return result

    metrics = run_ptq(args, data_factory=data_factory, evaluate_fn=evaluate_fn)
    out = Path(args.output)
    expected_files = {"config.json", "split.json", "calibration.json", "ptq.pt", "model_state.pt",
                      "metrics.json", "metrics.csv", "fp32_metrics.json"}
    assert expected_files <= {path.name for path in out.iterdir()}
    assert len(evaluations) == 2
    assert metrics["mode"] == "ptq" and metrics["use_cropr"] is True
    assert metrics["weight_bits"] == metrics["activation_bits"] == bits
    assert metrics["ptq_scope"] == scope and metrics["test_samples"] == 2
    assert metrics["top1_drop_pp"] == pytest.approx(evaluations[0]["top1_acc"] - evaluations[1]["top1_acc"])
    assert metrics["latency_backend"] == "fake_quant_fp32" and metrics["integer_latency_ms"] is None
    assert metrics["split_hash"] == "ptq-fixture"
    assert metrics["storage_model_state_bytes"] == (out / "model_state.pt").stat().st_size
    config = json.loads((out / "config.json").read_text())
    assert config["image_size"] == 64 and config["seed"] == source.seed
    assert config["cropr_pruning_rate"] == 1 and config["cropr_llf"] == llf
    assert effective[0]["batch_size"] == 3 and effective[0]["device"] == "cpu"
    assert effective[0]["data_root"] == str(tmp_path / "runtime-data")
    calibration = json.loads((out / "calibration.json").read_text())
    assert calibration["samples"] == 3 and calibration["observed_quantizers"]
    assert json.loads((out / "split.json").read_text())["sha256"] == "ptq-fixture"
    restored = load_ptq_checkpoint(out / "ptq.pt", device="cpu")
    assert not restored.training
    restored_parameters = dict(restored.named_parameters())
    assert set(dict(original.named_parameters())) == set(restored_parameters)
    assert all(torch.equal(parameter, restored_parameters[name])
               for name, parameter in original.named_parameters())
    pruner_quantizers = [module for module in restored.cropr.modules() if isinstance(module, Quantizer)]
    assert bool(pruner_quantizers) == (scope == "all")
    before = observer_buffers(restored)
    restored_result = evaluate(restored, DataLoader(datasets[2], batch_size=1), "cpu")
    assert restored_result["top1_acc"] == evaluations[-1]["top1_acc"]
    assert restored_result["loss"] == pytest.approx(evaluations[-1]["loss"], abs=1e-5)
    buffers = dict(restored.named_buffers())
    assert all(torch.equal(value, buffers[name]) for name, value in before.items())
    state = torch.load(out / "model_state.pt", map_location="cpu", weights_only=True)
    restored_state = restored.state_dict()
    assert all(torch.equal(value, restored_state[name]) for name, value in state.items())


@pytest.mark.parametrize("field,value", [("mode", "qat"), ("use_cropr", False)])
def test_ptq_rejects_incompatible_source(tmp_path, field, value):
    checkpoint, _, _ = save_source(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["config"][field] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="FP32|fp32|Cropr|cropr"):
        run_ptq(ptq_args(tmp_path, checkpoint))


def test_ptq_rejects_changed_data_split(tmp_path):
    checkpoint, _, _ = save_source(tmp_path)
    with pytest.raises(ValueError, match="split"):
        run_ptq(ptq_args(tmp_path, checkpoint),
                data_factory=lambda args: (*fixture_datasets(), {"sha256": "changed-split"}))


def test_ptq_requires_strict_source_state(tmp_path):
    checkpoint, _, _ = save_source(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model"].pop("head.weight")
    torch.save(payload, checkpoint)
    with pytest.raises((RuntimeError, ValueError), match="head.weight|[Mm]issing"):
        run_ptq(ptq_args(tmp_path, checkpoint),
                data_factory=lambda args: (*fixture_datasets(), {"sha256": "ptq-fixture"}))


def test_ptq_refuses_to_overwrite_existing_output(tmp_path):
    checkpoint, _, _ = save_source(tmp_path)
    args = ptq_args(tmp_path, checkpoint)
    output = Path(args.output)
    output.mkdir()
    marker = output / "previous-result.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="empty|[Ee]xist"):
        run_ptq(args)
    assert marker.read_text() == "keep"


def test_ptq_real_food101_loader_uses_source_split_and_deterministic_calibration(food_fixture):
    source = source_args(food_fixture)
    source.data_root = str(food_fixture)
    _, _, _, original_manifest = make_data(source)
    checkpoint, _, _ = save_source(food_fixture, source, original_manifest["sha256"])
    args = ptq_args(food_fixture, checkpoint, "--data-root", str(food_fixture),
                    "--batch-size", "32", "--calibration-samples", "2")
    metrics = main(args)
    assert metrics["test_samples"] == 101
    assert metrics["split_hash"] == original_manifest["sha256"]
    effective = parser().parse_args([
        "--mode", "fp32", "--output", "unused", "--data-root", str(food_fixture),
        "--image-size", "64", "--seed", "7",
    ])
    calibration, validation, _, manifest = make_ptq_data(effective)
    assert manifest["sha256"] == original_manifest["sha256"]
    assert not set(calibration.indices).intersection(validation.indices)
    assert calibration.indices == original_manifest["train_indices"]
    assert torch.equal(calibration[0][0], calibration[0][0])


def test_calibration_requires_prepared_quantizers():
    with pytest.raises(ValueError, match="Prepare PTQ"):
        calibrate(CalibrationProbe(), [], "cpu", max_samples=1)


def test_recalibration_starts_with_fresh_cumulative_ranges():
    model = prepare_ptq(CalibrationProbe(), 4, 4)
    for value in (100.0, 1.0):
        data = TensorDataset(torch.tensor([[-value, value]]), torch.zeros(1, dtype=torch.long))
        calibrate(model, DataLoader(data), "cpu", max_samples=1)
    observer = model.head.input_quant.fake_quant.activation_post_process
    assert observer.min_val.item() == -1.0 and observer.max_val.item() == 1.0


def test_calibration_rejects_nonfinite_ranges_and_always_closes_observers():
    model = prepare_ptq(CalibrationProbe(), 2, 2)
    data = TensorDataset(torch.full((1, 2), float("inf")), torch.zeros(1, dtype=torch.long))
    with pytest.raises((ValueError, AssertionError), match="finite|inf|minimum|maximum|min.*max"):
        calibrate(model, DataLoader(data), "cpu", max_samples=1)
    for module in model.modules():
        if isinstance(module, Quantizer):
            assert not module.calibrating and module.frozen
            assert module.fake_quant.observer_enabled.item() == 0


def test_restored_ptq_checkpoint_requires_strict_calibrated_state(tmp_path):
    checkpoint, _, _ = save_source(tmp_path)
    args = ptq_args(tmp_path, checkpoint)
    run_ptq(args, data_factory=lambda config: (*fixture_datasets(), {"sha256": "ptq-fixture"}))
    path = Path(args.output) / "ptq.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["model"].pop("head.weight")
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="head.weight|[Mm]issing"):
        load_ptq_checkpoint(path)
