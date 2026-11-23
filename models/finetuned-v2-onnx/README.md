# 캠퍼스 재질 파인튜닝 2차 — 현장 시험용

Material 2차 학습의 최종 5에폭 모델과 기존 Area 7에폭 모델입니다. CPU ONNX Runtime으로 실행합니다. NPU 변환 모델은 아닙니다.

## Pi 실행

저장소 루트에서 실행하세요. 기존 수집 프로그램은 먼저 종료하세요.

```bash
git fetch origin
git switch feat/finetuned-v2
git pull --ff-only
git lfs pull
python -m pip install onnxruntime
INFERENCE_BACKEND=onnxruntime MODEL_DIR="$PWD/models/finetuned-v2-onnx" python main.py
```

현재 장치의 가상환경에서 실행하세요. 기존 GPS·IMU·프레임 저장 환경변수는 그대로 유지합니다. 수집 결과의 model_version은 `campus-material-ft-v2-20261123-e5`여야 합니다.

기존 모델로 돌아가려면 MODEL_DIR 환경변수를 지정하지 않고 기존 실행 명령을 사용하세요. 기본 모델과 기본 backend는 변경하지 않았습니다.

## 근거와 한계

- 학습 run: material-20261123-230114, 최종 epoch 5. Area는 재학습하지 않았습니다.
- 원래 검증셋 mIoU: 기존 68.22%, 1차 70.38%, 2차 69.69%.
- 2차 흙·돌 IoU 52.3%, 우레탄 60.2%. 전체 평균 상승이 모든 재질의 개선을 의미하지 않습니다.
- 기존 사진 비교에는 같은 주행의 인접 학습 사진이 포함될 가능성이 있어 독립 평가로 볼 수 없습니다. 새 주행에서 블록 외에 흙·돌, 판석, 실내 오검출을 함께 확인하세요.
- thresholds: Area sidewalk >= 0.8 AND Material >= 0.8. 계단·파손 출력은 없습니다.
- Pi의 새 모델 실행 및 처리 시간은 아직 검증 전입니다. 모델 추론 성공과 버전 기록을 확인한 뒤 수집하세요.

ONNX 파일은 Git LFS로 관리합니다. checksums.json은 SHA256, training-metrics.jsonl은 에폭별 원래 검증셋 결과입니다.
