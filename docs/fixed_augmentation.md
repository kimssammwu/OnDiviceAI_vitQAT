# 동일한 증강 데이터를 모든 실험에서 재사용하기

`python -m vitqat.export_augmented`는 학습용 증강을 **한 번 생성해 파일로 저장**한다.
FP32/QAT/Cropr 실험은 같은 폴더를 읽는다. 학습 중 증강을 다시 추첨하지 않으므로
모델 구조, 모델의 난수 사용량, 데이터 로딩 worker 수가 증강 이미지 값을 바꾸지 않는다.

## 1. 원본 준비 및 고정 데이터 생성

저장소 폴더에서 기존 `vitqat` 환경을 활성화한 뒤 실행한다.
Food-101을 이미 받았다면 `--download`를 생략한다.

```bash
python -m vitqat.export_augmented \
  --data-root data --download \
  --output data/food101_fixed_s42 \
  --seed 42 --image-size 224 --val-fraction 0.1 \
  --mixup 0.8 --cutmix 1.0 --random-erasing 0.25 \
  --crop-min-scale 0.08 --views-per-image 1
```

위 증강 옵션은 이 생성 명령의 기본값이므로, `--output`과 데이터 경로/다운로드 여부만
지정해도 같은 증강 설정을 쓴다. 출력 폴더는 비어 있어야 한다.

| 기법 | 생성 과정 |
|---|---|
| RandomResizedCrop | 최소 면적 0.08의 무작위 crop. `--crop-min-scale`로 조절한다. |
| HorizontalFlip | 기존 무작위 좌우 반전을 사용한다. |
| RandAugment | 기존 torchvision RandAugment를 사용한다. |
| RandomErasing | 기본 확률 0.25. `--random-erasing 0`으로 끌 수 있다. |
| Mixup | 기본 alpha 0.8. `--mixup 0`으로 끌 수 있다. |
| CutMix | 기본 alpha 1.0. `--cutmix 0`으로 끌 수 있다. |

**모든 기법을 사용할 수 있지만 모든 이미지에 모든 변환을 강제로 적용하는 의미는 아니다.**
Flip/Erasing 등의 적용 여부와 변형량도 시드에 따라 한 번 결정되어 저장된다.
Mixup/CutMix를 모두 켜면 기존 구현처럼 각 이미지 쌍에 둘 중 하나를 선택한다.
홀수로 남은 마지막 한 장은 섞지 않고 label smoothing만 적용한다.

원본 train 75,750장 중 seed와 validation 비율로 분리한 **학습 68,175장만** 증강한다.
validation 7,575장과 공식 test 25,250장은 증강 데이터에 포함하지 않는다.
기본 `--views-per-image 1`은 학습 이미지마다 결과 한 장을 저장한다.
`--views-per-image 3`이면 서로 다른 고정 변형을 세 장씩 저장하며 저장 용량과 epoch당 학습량도 세 배다.

## 2. seed가 고정하는 것

`42`는 특별한 값이 아니라 재현할 난수 결과를 고르는 번호다. 모든 실험에서 같은 값을 사용한다.

- 원본 이미지의 변형은 `seed + 변형 번호 + 원본 이미지 ID`로 결정한다.
- 이미지를 섞을 상대와 Mixup/CutMix의 선택/비율도 별도 시드 규칙으로 결정한다.
- 모델 학습에 쓰는 난수와 생성기의 난수를 분리하고, 생성 전후의 난수 상태를 복원한다.
- 같은 원본, 설정, 라이브러리 버전과 seed로 다시 생성하면 같은 텐서와 데이터 SHA-256이 나온다.
- 한 번 생성한 폴더를 공유하면, 재생성할 필요 없이 동일한 데이터 바이트를 그대로 사용한다.

시드만 맞추는 것보다 **생성한 폴더와 그 안의 `manifest.json`을 함께 공유**하는 것이 확실하다.
`manifest.json`은 증강 설정, 원본 분할, 라이브러리 버전, 파일별 해시 및 전체 데이터 해시를 기록한다.

## 3. FP32 / QAT 학습에서 같은 폴더 사용

이미 내보낸 `vitqat_fixed_train_v1` 폴더는 다시 생성하지 않고 연결한다.
별도 내보내기 도구가 추가한 출처 메타데이터나 preview/verification 파일도 호환된다.
`aix408-11`에 준비된 데이터는 `/home/user/minwoo/data/food101_fixed_aug_s42_fp32_v1`이다.
이 서버에서 원본 `--data-root`는 `/home/user/minwoo/data`로 지정한다.

```bash
python -m vitqat.train --mode fp32 --use-cropr \
  --data-root data --fixed-train-data data/food101_fixed_s42 \
  --seed 42 --image-size 224 --val-fraction 0.1 \
  --output runs/fixed_cropr_fp32 --device cuda

python -m vitqat.train --mode qat --use-cropr --weight-bits 2 --activation-bits 2 \
  --data-root data --fixed-train-data data/food101_fixed_s42 \
  --seed 42 --image-size 224 --val-fraction 0.1 \
  --output runs/fixed_cropr_qat_w2a2 --device cuda
```

**학습 명령에는 `--mixup`, `--cutmix`, `--random-erasing`을 다시 추가하지 않는다.**
이미 저장된 이미지와 soft target을 사용하므로, 추가 온라인 증강 옵션을 켜면 오류로 거부한다.
기본 Crop/Flip/RandAugment도 고정 데이터를 읽을 때는 다시 적용하지 않는다.
label smoothing은 저장된 정답에 포함돼 있으며 학습 손실에서 이중 적용하지 않는다.

입력 크기, seed/validation 비율에 따른 데이터 분할, 클래스 매핑 및 label smoothing은
생성 설정과 맞아야 한다. FP32/QAT 간에는 batch size, epoch 수, optimizer 등 나머지 비교 조건도 통일한다.
학습 배치 순서는 `seed + epoch`로 정해져 epoch마다 바뀌지만 같은 실험 설정에서 재현된다.
큰 shard 파일을 이미지마다 다시 읽지 않도록 shard 순서와 내부 순서를 섞는 방식을 쓴다.
여러 worker는 spawn으로 실행하여 CPU/CUDA 초기화 이후 fork 시 발생할 수 있는 교착을 피한다.

`config.json`과 `metrics.json`의 `fixed_train_hash`가 같으면 같은 고정 데이터 manifest를 사용한 것이다.
학습 재개 시에는 이 해시를 비교하므로 데이터 폴더의 내용을 바꿔서 이어 학습할 수 없다.
폴더 위치만 옮기고 내용은 같다면 해시가 유지되어 재개할 수 있다.
시드 고정은 데이터 입력을 재현한다. 다른 GPU/라이브러리 버전에서 학습 결과까지 비트 단위로
같아지는 것을 보장하지는 않는다.

## 4. 2-a PTQ

FP32 학습 후 같은 checkpoint로 2bit와 4bit를 평가한다.
PTQ 보정은 고정 증강 파일이 아닌 **원본 학습 split에 평가 전처리**를 적용한 이미지를 사용한다.
PTQ 실행 시 고정 데이터 폴더는 없어도 되지만 원본 Food-101은 필요하다.

```bash
python -m vitqat.train --mode ptq --fp32-checkpoint runs/fixed_cropr_fp32/best.pt \
  --data-root data --output runs/fixed_ptq_w2a2 --device cuda
python -m vitqat.train --mode ptq --fp32-checkpoint runs/fixed_cropr_fp32/best.pt \
  --weight-bits 4 --activation-bits 4 \
  --data-root data --output runs/fixed_ptq_w4a4 --device cuda
```

## 용량과 저장 형식

- `manifest.json`: 설정, 분할, 개수 및 해시.
- `shard-00000.pt` 등: 이미 정규화된 이미지 텐서, soft target, 원본/상대 이미지 ID 및 변형 번호.
- 학습은 필요한 shard만 읽는다. 파일 해시, 입력 형태, 정답 확률 및 학습 split 소속을 검사한다.

기본 224×224, 학습 68,175장, 변형 한 장일 때 float32 이미지 payload는 **약 41GB(38.2GiB)**다.
실제 필요 공간에는 정답과 메타데이터도 포함된다. 생성 전에 예상 용량과 남은 공간을 검사한다.
`--storage-dtype float16`이면 대략 절반이지만 값에 float16 반올림이 들어간다.
모든 비교 실험은 같은 저장 dtype의 같은 폴더를 사용한다. 로딩 시에는 float32로 변환한다.
JPEG/PNG는 Mixup/CutMix의 soft target이나 정규화 후 값을 온전히 담지 못하므로 학습용 저장에 쓰지 않는다.

이 방식은 각 epoch에서 같은 고정 변형을 재사용한다. 매 epoch 새 증강을 만드는 기존 온라인 방식과
학습 조건이 다르므로 결과를 구분해 기록한다.
