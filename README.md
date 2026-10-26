# OnDiviceAI_vitQAT
온디바이스 AI - vit , QAT , ptq

```powershell
conda env create -f environment.yml
conda activate vitqat
```

Food-101 scratch 학습 실험 **1-1 FP32 ViT / 1-2 QAT-only ViT**의 실행 방법과 지표 정의는
[docs/experiments.md](docs/experiments.md)를 참고하세요.

실험 **2-a Cropr FP32 학습 후 PTQ**도 같은 실행 진입점을 사용합니다.
학습된 FP32 Cropr `best.pt`를 `--mode ptq --fp32-checkpoint`로 지정하면,
기본 W2A2 보정과 FP32/PTQ 비교 평가를 수행합니다. 4bit 실행 방법과 저장 결과는
[PTQ 실험 안내](docs/experiments.md#실험-2-a-cropr-fp32--ptq)를 참고하세요.

모든 실험에 **완전히 같은 증강 이미지와 정답**을 쓰려면
[고정 증강 데이터 생성 및 학습 안내](docs/fixed_augmentation.md)를 참고하세요.
`python -m vitqat.export_augmented --output data/food101_fixed_s42 --seed 42`로 한 번 저장한 뒤,
학습에서 `--fixed-train-data data/food101_fixed_s42`로 재사용합니다.
