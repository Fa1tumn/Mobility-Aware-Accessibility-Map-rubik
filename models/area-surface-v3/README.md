# Area + Surface 모델 업데이트

Area best epoch 7 / Material best epoch 10, LR-ASPP MobileNetV3 Large.

- ONNX 2개: 추론용 FP32 가중치. export JSON의 SHA256과 대조 완료.
- checkpoints/: PyTorch best 체크포인트. 신뢰할 수 있는 본 프로젝트 학습 산출물이며 재학습/재변환용.
- model-spec.json: 클래스 순서·입력·전처리·결합 규칙. 기존 프로그램이 자동으로 읽는 설정은 아니며 연결용 명세입니다.
- checksums.json: 모든 가중치 SHA256.

이 폴더를 대상 저장소의 models/area-surface-v3로 복사할 수 있습니다. 현재 iCloud 대상 폴더는 macOS 접근 거부로 읽거나 수정하지 못했습니다. 따라서 실 프로젝트 업데이트·연결 완료 상태가 아닙니다.

기존 YOLO 7종 QNN decoder에 이 가중치를 그대로 넣으면 안 됩니다. 출력은 dense logits이며 Area 7채널/Material 6채널입니다. RGB 640×360, ImageNet 정규화, 각 head별 softmax가 필요합니다. Area의 sidewalk confidence ≥ 0.8 AND Material confidence ≥ 0.8인 픽셀만 재질 관측으로 채택합니다. 계단·파손 모델은 이번 패키지에 포함하지 않았습니다.

QCS6490 NPU용 DLC/context binary로 변환된 파일이 아닙니다. NPU 변환·실기기 추론 검증은 별도 단계입니다.
