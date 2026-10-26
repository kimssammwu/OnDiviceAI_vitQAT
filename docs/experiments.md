# Food-101 ViT 실험 1-1 / 1-2

- 1-1: `vit_tiny_patch16_224`, 101 classes, CLS pooling, FP32 scratch 학습.
- 1-2: 동일 모델과 초기값으로 시작하는 QAT-only scratch 학습. Pruning 없음.
- 같은 실행 코드에서 Cropr on/off와 QAT on/off를 독립적으로 조합할 수 있다.
- `pretrained=False`이며 pretrained checkpoint 다운로드/로딩 경로가 없다.

저장소 루트에서 Conda 환경을 생성하고 활성화한다.

```powershell
conda env create -f environment.yml
conda activate vitqat
python -m pytest tests -q
```

`environment.yml`과 `requirements.txt`는 Python 3.10, PyTorch 2.3.0+cu121,
torchvision 0.18.0+cu121, timm 1.0.3, NumPy 1.26.4를 지정한다.
CUDA 12.1 runtime은 PyTorch wheel에 포함되며 NVIDIA GPU driver가 필요하다.
[공식 PyTorch 버전별 설치 안내](https://pytorch.org/get-started/previous-versions/)의 조합을 사용한다.
이미 생성한 환경에서는 `conda activate vitqat`만 실행한다.
활성화 없이 실행하려면 `conda run --no-capture-output -n vitqat python -m vitqat.train ...`을 사용한다.

```powershell
python -m vitqat.train --mode fp32 --data-root data --download --output runs/1-1_fp32 --device cuda
python -m vitqat.train --mode qat --weight-bits 2 --activation-bits 2 --data-root data --output runs/1-2_w2a2 --device cuda
python -m vitqat.train --mode qat --weight-bits 4 --activation-bits 8 --data-root data --output runs/1-2_w4a8 --device cuda
python -m vitqat.train --mode fp32 --use-cropr --cropr-pruning-rate 8 --data-root data --output runs/cropr_fp32 --device cuda
python -m vitqat.train --mode qat --use-cropr --weight-bits 2 --activation-bits 2 --data-root data --output runs/cropr_w2a2 --device cuda
```

`--weight-bits`, `--activation-bits`는 독립적으로 2/3/4/8을 지원한다. FP32 mode는 항상 W32A32이다.
Food-101 파일은 `data/food-101/{images,meta}`에서 읽는다. `--download`는 명시했을 때만 다운로드한다.

`--model vit_tiny_patch16_224`가 기본 모델이며, `--model vit_small_patch16_224`처럼
표준 timm ViT 아키텍처를 지정할 수 있다. `--experiment`는 결과에 저장할 자유로운 실험명이다.
생략하면 Cropr off에서는 FP32=`1-1`, QAT=`1-2`, Cropr on에서는 `cropr_fp32`/`cropr_qat`로 기록한다.
실험명은 모델 동작을 바꾸지 않는다. Cropr FP32 학습 뒤 PTQ는 아래 `--mode ptq`로 별도 실행한다.
기존 `python -m experiments.train` 명령은 `python -m vitqat.train`으로 변경되었다.

## 실험 2-a: Cropr FP32 → PTQ

회의록의 이민우 담당 실험이다. **scratch FP32 Cropr 학습 → best checkpoint 선택 →
학습 데이터 일부로 PTQ 보정 → FP32/PTQ test 비교** 순서로 실행한다.
기존 FP32 학습 및 체크포인트 형식을 재사용하며, PTQ 단계에서 optimizer나 backward를 실행하지 않는다.

FP32 Cropr 학습 결과가 없다면 기존 학습 코드를 그대로 실행한다.

```powershell
python -m vitqat.train --mode fp32 --use-cropr --cropr-pruning-rate 8 --data-root data --download --output runs/2-a_cropr_fp32 --device cuda
```

학습이 끝나면 **2bit를 먼저** 평가한다. `--weight-bits`와 `--activation-bits`의 기본값은 모두 2이다.
PTQ는 체크포인트에서 Cropr 사용 여부와 구조를 가져오므로 `--use-cropr`를 다시 지정할 필요가 없다.

```powershell
python -m vitqat.train --mode ptq --fp32-checkpoint runs/2-a_cropr_fp32/best.pt --data-root data --output runs/2-a_ptq_w2a2 --device cuda
python -m vitqat.train --mode ptq --fp32-checkpoint runs/2-a_cropr_fp32/best.pt --weight-bits 4 --activation-bits 4 --data-root data --output runs/2-a_ptq_w4a4 --device cuda
```

가중치와 activation bit는 독립적이다. 팀이 W4A8을 비교 조건으로 정하면 두 번째 명령의
`--activation-bits 4`를 `--activation-bits 8`로 변경한다. FP32 모델을 다시 학습할 필요는 없다.
지원 bit는 기존 QAT와 동일한 2/3/4/8이며, 2-a의 우선 실험은 W2A2와 W4A4이다.

| PTQ 옵션 | 기본값 / 의미 |
|---|---|
| `--fp32-checkpoint` | 필수. 기존 FP32 Cropr 학습의 `best.pt` 또는 `last.pt`. QAT checkpoint와 weight-only `model_state.pt`는 입력으로 쓰지 않는다. |
| `--calibration-samples` | 1024. 학습 split에서 seed로 무작위 추출하는 최대 이미지 수. 남은 이미지가 더 적으면 전부 사용한다. |
| `--ptq-scope all` | 기본값. 기존 QAT all과 동일한 양자화 대상(ViT 및 Cropr scorer). |
| `--ptq-scope backbone` | ViT만 양자화하며 Cropr 모듈은 FP32로 유지. 비교하는 QAT scope도 동일하게 맞춘다. |
| `--batch-size`, `--workers`, `--threads` | 보정 및 평가 실행 설정. 학습 때와 달라도 된다. |

모델명, 입력 크기, Cropr pruning rate/LLF, seed, validation 비율 및 학습 설정은 **원본 checkpoint 설정을 우선**한다.
CLI에서 이 값을 바꿔도 PTQ 모델 구조/분할을 변경하지 않는다. 따라서 학습 때의 epochs나 pruning 옵션을
다시 입력하지 않는다. 데이터 경로, download, 장치, batch/workers/threads, latency 측정 설정,
실험명, output, PTQ 비트와 보정 옵션은 현재 CLI 값을 사용한다.
학습 때와 같은 Food-101 데이터 및 분할 hash가 필요하며, 불일치하면 실행을 거부한다.
`--resume`/`--eval-checkpoint`는 PTQ에서 사용하지 않는다. 각 실행은 비어 있는 새 output을 지정한다.

보정에는 **학습 split의 이미지만** 사용한다. 전처리는 평가와 같은 Resize/CenterCrop/Normalize이며
validation/test를 관측기 통계 수집에 사용하지 않는다. Cropr와 전체 모델을 `eval()`로 유지하여
실제 추론 scorer/pruning/LLF 경로를 보정하며 보조 학습 head는 실행하지 않는다.
보정 동안 fake quantization을 끄고 누적 MinMax로 FP32 범위를 수집한 뒤,
관측기를 고정하고 fake quantization을 켠다. QAT의 EMA observer와는 다른 정적 PTQ 보정이다.
양자화 함수는 동일한 uniform affine 방식이다. 가중치는 output channel별, activation은 tensor별로 보정한다.
보정이 끝나면 test 입력으로 scale/zero point/min/max를 갱신하지 않는다.

각 PTQ output에는 다음을 저장한다.

- `config.json`, `split.json`: 실제 적용한 설정과 원본 학습 분할.
- `calibration.json`: 선택한 학습 이미지 인덱스, seed, 실제 보정 이미지/배치 수 및 관측한 quantizer 목록.
- `fp32_metrics.json`: 같은 원본 checkpoint를 이 실행 환경에서 평가한 FP32 기준 결과.
- `metrics.json`, `metrics.csv`: PTQ 결과. `fp32_top1_acc`, `top1_drop_pp`(FP32 − PTQ, %p),
  `weight_updates=0`, 실제 비트, scope, 원본 checkpoint SHA-256 등을 포함한다.
- `model_state.pt`: FP32 master weight와 고정된 PTQ quantizer 상태.
- `ptq.pt`: 모델 구조 설정과 보정 정보를 포함하는 재로딩용 checkpoint. optimizer 상태는 포함하지 않는다.

`fp32_metrics.json`의 `storage_serialized_state_bytes`는 FP32 state dict를 메모리에서 직렬화한 바이트 수이고,
PTQ의 `storage_model_state_bytes`는 실제 저장 파일 크기이다. 두 값 모두 저비트 packed 모델 크기가 아니다.
packed 크기의 이론 추정은 각각 `storage_estimated_packed_bytes`로 비교한다.

보정한 모델은 원래 데이터나 FP32 checkpoint 없이 다음처럼 다시 로드한다.

```python
import torch
from vitqat.ptq import load_ptq_checkpoint

model = load_ptq_checkpoint("runs/2-a_ptq_w2a2/ptq.pt", device="cuda")
with torch.inference_mode():
    logits = model(images)  # 학습/평가와 동일하게 전처리한 입력
```

이 경로는 **PTQ 정확도 실험을 위한 부동소수점 fake quantization**이다. 실제 INT2/INT4 kernel이나
packed export를 제공하지 않는다. latency는 `fake_quant_fp32`, `integer_latency_ms`는 `null`이며,
fake quantization latency를 실제 저비트 추론 가속으로 해석하지 않는다.
실제 Food-101 정확도와 수렴은 전체 학습 및 PTQ 평가를 완료한 뒤 보고한다.

검증은 `python -m pytest tests/test_ptq.py -q`로 실행한다.
보정/평가 중 가중치 보존, 추론 모드 유지, 관측기 고정, 2/4bit와 scope/LLF,
원본 설정 승계, 분할 불일치 거부, 저장/재로딩 및 FP32/PTQ 비교 경로를 확인한다.
참고: [PyTorch 2.3 정적 PTQ와 보정](https://docs.pytorch.org/docs/2.3/quantization.html#post-training-static-quantization).

## 코드 확장 방법

실험별로 학습 코드를 복사하지 않고, 단일 실행 진입점에 옵션을 전달한다.
실험 결과는 각각 다른 `--output runs/...` 경로에 저장한다.

```text
vitqat/
  train.py          # 실행 옵션에 따라 구성하고 runner 호출
  cli.py            # 사용자 옵션
  models.py         # 모델 선택 및 QAT 적용
  cropr_adapter.py  # 원본 파일 수정 없이 기존 Cropr 모델 생성/연결
  quantization.py   # quantizer와 QAT 연산
  ptq.py            # FP32 checkpoint 로드, PTQ 보정/비교 평가/저장/재로딩
  data.py           # Food-101 split, 전처리, loader
  training.py       # 학습/평가 step, scheduler, checkpoint
  runner.py         # epoch 반복, best 모델 선택, 결과 저장
  metrics.py        # 실제 block별 token 수에 따른 ViT/Cropr 지표
cropr/cls/          # 기존 ViT forward, Cropr scorer/pruning/LLF 구현 재사용
tests/              # 회귀 및 옵션/확장 검증
docs/experiments.md # 실행 방법
```

모델 크기·학습 조건·정밀도 비교는 옵션만 바꾼다. 새 알고리즘은 해당 기능 모듈에 구현하고
`train.py`에서 선택하도록 연결한다. `runner.run()`은 `build_model`, `measure_model`을 받고,
필요하면 `train_epoch_fn`, `evaluate_fn`, `data_factory`도 교체할 수 있어 공통 epoch/checkpoint
코드를 수정하지 않고 새 모델, 보조 loss, 평가 방식을 연결할 수 있다.

`cropr/`의 원본 파일은 수정하지 않는다. `cropr_adapter.py`에서 기존
`cropr.cls.vision_transformer.VisionTransformer`를 생성할 때 원본의 `use_cropr` 설정을 전달한다.
timm에서 만든 scratch backbone의 초기 파라미터를 가져와 Cropr on/off의 backbone 초기값을 맞춘다.
Cropr on/off 모두 원본의 forward/pruning/LLF 경로를 사용하고, 선택된 모델에 QAT를 적용한다.
원본의 스크립트용 import는 adapter에서 런타임 package alias로 연결한다.
QAT 전용 CrossAttention subclass는 `quantization.py`에 두며 원본 scorer를 수정하지 않는다.
원본 transformer block과 호환되는 표준 ViT 아키텍처(Tiny/Small 등의 MLP ratio 4, qkv bias 사용)를 지원한다.
`measure_vit()`는 실제 block 입력과 pruner 출력을 관측하여 토큰 수와 연산량을 계산한다.
변경한 기능의 테스트는 `tests/test_<기능>.py`로 추가한다.
동일 파일을 수정하면 충돌은 여전히 가능하지만, 학습 실행 조건을 바꾸기 위해 코드를 수정할 필요는 없다.

## Cropr 옵션

| 옵션 | 기본값 / 의미 |
|---|---|
| `--use-cropr` / `--no-use-cropr` | 기본 off. `--mode fp32` 또는 `--mode qat`와 독립 |
| `--cropr-pruning-rate` | 8. 각 Cropr가 제거하는 token 수 |
| `--cropr-llf` / `--no-cropr-llf` | 기본 off. LLF on이면 마지막 block 앞에서 제거한 token 복원 |
| `--qat-scope all` | 기본값. ViT 및 Cropr의 Linear, learnable queries, Q/K/V/probability 양자화 |
| `--qat-scope backbone` | ViT만 QAT 적용, Cropr 모듈은 FP32 |
| `--aux-loss-weight` | 기본 1.0. 각 Cropr 보조 head loss의 계수 |

기존 Cropr의 schedule을 유지하여 첫 모듈은 `pruning_rate + 1`개를 제거한다.
따라서 rate=0도 첫 모듈에서 한 token을 제거한다. Pruning 없는 기준선은 Cropr를 끈다.
224 입력, 12 blocks, rate=8이면 LLF off에서 11개 모듈이 89개 patch token을 제거한다.
LLF on에서는 10개 모듈이 81개를 제거한 후 마지막 block에 전체 token을 복원한다.
입력을 줄일 때는 rate도 낮춰야 하며, 모든 token이 제거되는 설정은 오류로 처리한다.

학습 loss는 `main CE + aux_loss_weight * sum(aux CE)`로 원본의 보조 head 학습을 유지한다.
Top-k 선택 자체는 미분되지 않으므로 learnable queries의 학습에는 보조 loss가 필요하다.
평가에서는 원본 `forward_scorer` 경로를 사용하여 보조 head를 실행하지 않는다.
QAT scope=all에서는 학습/평가 scorer의 query/key quantizer를 공유한다.
Cropr의 query 수와 head 수는 각 1개이며, q/k/v projection은 원본 기본값인 Identity를 사용한다.

## 공통 학습 조건

기본값은 입력 224, 300 epochs, warmup 20 epochs, batch 128, accumulation 1, AdamW,
LR 5e-4, cosine decay, min LR 1e-6, weight decay 0.05, label smoothing 0.1,
drop path 0.1, gradient clip norm 1.0이다. LR은 실제 지정값이며 batch size에 자동 비례하지 않는다.
초기 scratch 비교를 위한 설정이며 수렴이 확인된 최적 recipe라는 의미는 아니다.
AMP와 TF32를 사용하지 않는다. Layer-wise LR decay도 사용하지 않는다.

학습 전처리는 RandomResizedCrop, horizontal flip, RandAugment, ImageNet normalization이다.
평가는 Resize(256/224 배율), CenterCrop, 같은 normalization을 사용한다.

### 선택형 데이터 증강

`--mixup`, `--cutmix`, `--random-erasing`, `--crop-min-scale` 옵션으로 train 증강을 추가할 수 있다.
기본값은 모두 꺼짐이라 옵션을 주지 않으면 위의 기본 전처리와 같다.
**비교 실험(FP32/QAT/Cropr)은 모두 같은 증강 설정을 써야 하고**, 증강을 켜면 `train_loss`를 끈 실험과 비교할 수 없다.
옵션 설명, 값 가이드, 확인 방법은 [augmentation.md](augmentation.md)를 참고한다.

동일한 seed에서 FP32/QAT 모델의 모든 학습 파라미터 초기값이 일치한다.

Food-101 공식 train 75,750장을 `--seed 42 --val-fraction 0.1` 기준 클래스별로
train 68,175 / validation 7,575장으로 분리한다. 공식 test 25,250장은 epoch별 모델 선택에
사용하지 않고 best validation 모델의 최종 평가에만 사용한다. `split.json`에 정확한 인덱스와 hash를 저장한다.
서로 다른 정밀도 비교 시 seed와 학습 조건을 유지한다. 필요하면 여러 seed를 별도 output에서 반복한다.

## QAT 정의

PyTorch FakeQuantize의 uniform affine quantization을 사용한다.
`q = clamp(round(x/scale) + zero_point, 0, 2**bits-1)`, `x_hat = (q-zero_point)*scale`.
학습 가능한 FP32 master weight에 fake quantization을 적용하고 STE로 gradient를 전달한다.
이는 실제 INT2/INT4 kernel이나 모델 배포 형식의 구현은 아니다.

| 대상 | 적용 방식 |
|---|---|
| 모든 Linear: QKV, attention projection, MLP, classifier | weight는 output channel별, 입력/출력 activation은 tensor별 quantization |
| Patch embedding Conv2d | 같은 weight/activation quantization |
| Attention Q/K/V 및 Softmax 이후 probability | tensor별 activation quantization |
| Cropr (`--qat-scope all`) | learnable queries weight quantization, Q/K/V/probability 및 Linear quantization. Query tensor `[1, M, D]`의 axis 0을 기준으로 quantize |
| LayerNorm, Softmax, GELU, residual add, bias, CLS/position embedding | FP32 |

Weight observer는 매 학습 forward에서 min/max를 갱신하고, activation observer는 EMA(0.01)를 사용한다.
처음부터 QAT를 켜며 별도 FP32 pretraining은 하지 않는다. 평가에서는 모든 observer를 정지한다.
`--freeze-observers-epoch N`을 지정하면 0부터 센 N번째 epoch부터 weight/activation observer를 모두 고정한다.
기본값 -1은 학습 내내 갱신한다. 정밀도와 observer 상태는 checkpoint에 저장된다.
FP32와 QAT 모두 명시적 attention matmul을 사용한다.

## 결과와 재개

각 output 폴더에 다음을 저장한다.

- `config.json`, `split.json`: 설정과 데이터 분할.
- `epochs.jsonl`: total/main/aux train loss, LR, optimizer step 수, validation loss/Top-1.
- `last.pt`, `best.pt`: 모델, quantizer, optimizer, epoch, best accuracy, RNG, 설정, split hash.
- `model_state.pt`: 선택된 모델의 state dict. QAT는 FP32 master weights와 quantizer 상태를 포함한다.
- `metrics.json`, `metrics.csv`: 최종 test와 모델 크기/속도 지표, 실행 환경.

중단된 실행은 **원래와 같은 설정**에 `--resume runs/1-2_w2a2/last.pt`를 추가한다.
동일한 output 폴더와 그 폴더의 best.pt가 필요하다. 정밀도, seed, 학습 길이 등 불일치는 오류로 처리한다.
일반 실행에서 비어 있지 않은 output을 덮어쓰지 않는다.
다시 평가하려면 원래 학습 옵션에 `--eval-checkpoint runs/1-2_w2a2/best.pt`와 새로운 `--output`을 지정한다.
`--epochs`, `--warmup-epochs` 등 checkpoint 설정도 원래 값과 맞춘다.

## 지표의 의미

| PDF 지표 | 결과 필드와 기준 |
|---|---|
| Top-1 accuracy | `top1_acc`: best validation checkpoint의 공식 test 정확도(%) |
| GFLOPs | `gflops`: batch 1, MAC=2 FLOPs, 추론에 실행된 Conv/Linear, ViT QK/AV 및 Cropr QK. bias/norm/activation/sorting/quantization 비용 제외 |
| Params | `params`: 보조 head를 포함한 전체 학습 파라미터. `inference_params`: 추론에서 사용하지 않는 보조 분기 제외. observer buffer 제외 |
| Storage cost | `storage_model_state_bytes`: 실제 state 파일 크기. `storage_estimated_packed_bytes`: packed weight와 FP32 파라미터, scale/zero point의 이론 payload 크기 |
| Token pruning ratio | `token_pruning_ratio`: LLF 복원 전 최대 patch token 제거 비율(CLS 제외). `final_token_pruning_ratio`: 마지막 block 입력 기준. Cropr off면 둘 다 0 |
| Latency | `latency_ms_mean/median/p95`: batch 1, 장치에 올린 입력, forward만, warmup 10회 후 100회 측정 |

GFLOPs와 Params는 양자화 bit 수만 줄여도 감소하지 않는다. Storage 추정에는 파일 형식/정렬/런타임 overhead가
포함되지 않으며 실제 저장 크기로 보고하지 않는다. QAT latency의 `latency_backend=fake_quant_fp32`는
fake quantization 비용까지 포함한다. 실제 저비트 backend가 없으므로 `integer_latency_ms=null`이다.
실제 저비트 추론 속도/저장 비용 연구에는 별도 packed export와 지원 kernel이 필요하다.
CUDA 측정은 전후 synchronize를 수행한다. 장치명, 라이브러리 버전, CPU thread 수도 결과에 저장한다.
`tokens_per_block`, `tokens_after_pruning`에 실제 관측한 token 수를 저장한다.
현재 저장 파일과 packed storage 추정에는 학습용 보조 head/aggregation 분기도 포함된다.

## 검증

```powershell
python -m pytest tests -q
```

양자화 level/gradient, observer 고정, 초기값 및 fake quant off 출력 일치,
W2A2/W2A8/W4A4/W4A8 backward, split disjointness, 마지막 accumulation 처리,
FP32/QAT checkpoint 재개 재현, GFLOPs 공식, 실제 Food101 loader를 통과하는 생성 이미지 fixture의
학습→검증→저장→최종 평가→재개→재평가 경로를 검증한다.
fixture의 결과는 Food-101 실제 정확도나 수렴 검증 결과가 아니다.

2026-10-14 초기 구현 검증 결과: CPU 환경(Python 3.11, PyTorch 2.14.1+cpu,
torchvision 0.16.2+cu121)에서 timm 0.9.16 및 별도 임시 설치한 timm 1.0.3 각각
**23개 테스트가 통과**했다. CLI 도움말, Python 컴파일 검사도 통과했다.
당시 시스템 Python에서는 torchvision native image extension 로딩 경고가 있었고,
PIL 이미지 경로로 CPU 테스트를 수행했다.

옵션 기반 구조로 정리한 뒤 timm 0.9.16 환경에서 **30개 테스트가 통과**했다.
기존 23개 회귀 검증에 더해 옵션으로 ViT-Small 선택, 정밀도 옵션,
잘못된 observer 설정 거부, 공통 runner에 다른 모델/학습/평가/지표 함수를 연결하는 경로를 검증했다.

Conda `vitqat` 환경(Python 3.10.22, PyTorch 2.3.0+cu121, torchvision 0.18.0+cu121,
timm 1.0.3)을 새로 생성한 뒤 **34개 테스트가 모두 통과**했다.
`pip check`는 의존성 충돌이 없었고, torchvision native 연산 및 CUDA 12.1 사용을 확인했다.
RTX 3060 Laptop GPU에서 224×224 입력의 FP32/W2A2/W4A4/W4A8 forward,
backward, optimizer step, latency 측정을 검증했다. GPU가 없는 환경에서는 이 4개 테스트를 건너뛴다.
실제 Food-101 전체 학습, RTX 4090 latency, 저비트 native kernel 성능은 아직 검증하지 않았다.

2026-10-15 Cropr 원본 보존 adapter 적용 후 Conda `vitqat` 환경에서 전체 **48개 테스트가 통과**했다
(`219.01s`, 경고 1개). Cropr on/off, LLF, QAT 적용 범위(all/backbone), 원본 생성자와의 동등성,
auxiliary loss의 query gradient, checkpoint 재개, 공통 학습 파이프라인 및 CUDA 실행을 포함한다.
`git status --short -- cropr`와 `git diff -- cropr`에서 원본 소스 변경이 없음을 확인했다.

구현 참고: [PyTorch FakeQuantize](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html),
[torchvision Food101](https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/food101.html).
