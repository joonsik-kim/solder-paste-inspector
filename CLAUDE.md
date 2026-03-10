# Solder Paste Inspector — Project Context

## Overview
VT-S730 AOI 장비의 3D 높이맵 이미지에서 솔더 필렛을 U-Net으로 세그멘테이션하고,
형상 분석(높이/경사/CE비율)을 수행하는 검사 시스템.

## CRITICAL — 이 이미지는 일반 사진이 아님
VT-S730이 찍는 이미지는 **3D 높이 데이터를 RGB로 인코딩**한 것:
- **Blue 높음** → 표면이 높음 (솔더 중심부)
- **Green 높음** → 중간 높이 (경사면)
- **Red 높음** → 표면이 낮음 (PCB/가장자리)

일반 카메라 사진처럼 처리하면 안 됨. 색상 = 높이 정보.

## 핵심 기술 결정사항

### 전처리
- **CLAHE 필수**: LAB 색공간 L채널, clip_limit=3.0, grid_size=(8,8)
- 학습과 추론 모두 동일하게 적용해야 함 (안 하면 성능 저하)

### 모델
- **U-Net + ResNet18 encoder** (segmentation_models_pytorch)
- 입력: 128x128, BCE+Dice loss
- 현재 최고 성능: **Val IoU 0.9245** (run_006, epoch 121)

### 형상 분석 지표
- **Height Proxy**: B/(R+G+B) → 0~1 상대 높이. JET 컬러맵에서 빨강=높음, 파랑=낮음
- **Slope**: Height map에 Sobel 필터 적용. 값 높으면 급경사
- **CE Ratio** (Center-Edge): Distance transform 기반. >1이면 볼록, <1이면 오목
- **면적**: pixel 수 × pixel_size² (VT-S730 기본 15µm/px)

## 학습 히스토리 & 시행착오

### 데이터셋
| 소스 | 라벨 수 | 비고 |
|------|---------|------|
| images_main | 181 | 주 데이터셋 |
| images_gull | 52 | 걸윙 타입 |
| images_TNMX | 360 | TNMX 보드, 가장 많음 |
| **합계** | **593** | |

### 학습 실행 기록
- **run_001~005**: 초기 실험들 (images_main + gull 기반)
- **run_006**: TNMX 360장 추가 → **최종 모델** (epoch 121, Val IoU 0.9245)
- 기존 OpenCV 방식 최고 IoU: 0.8165 → U-Net으로 **+13%p 개선**

### 알려진 이슈
- Val 데이터에 노이즈 있음 (라벨 품질 불균일) → 향후 정리 필요
- Training loss는 계속 감소하지만 val loss는 epoch 80 이후 정체 → 과적합 징후
- Val IoU가 0.93~0.94 사이에서 진동 → val 데이터 정리하면 개선 가능성 있음

## 프로젝트 구조 (3-tier)

```
solder_paste_inspector/
├── [연구/프로토타입] 루트 레벨
│   ├── main.py            ← 레거시 OpenCV 검출기 (U-Net 아님)
│   ├── config.py           ← 검출 파라미터 설정
│   ├── detector.py         ← 다중 알고리즘 검출 엔진
│   ├── measurement.py      ← 면적 계산, 통계, CSV/JSON
│   └── ...
│
├── dl/                     [학습 파이프라인]
│   ├── train.py            ← U-Net 학습 스크립트
│   ├── predict.py          ← 개발용 추론 (CLAHE/분석 없음)
│   ├── models/             ← 학습된 모델들
│   └── results/run_001~006 ← 학습 결과 히스토리
│
├── deploy/                 [배포용 — 자체 완결]
│   ├── spi.py              ← 통합 CLI (predict/export/update-model/info)
│   ├── models/             ← best_model.pth + spi_model.onnx
│   ├── model_registry.json ← 모델 버전 관리
│   ├── requirements.txt
│   └── README.md           ← 사용법
│
├── scripts/                [유틸리티]
│   ├── convert_labelme_to_masks.py  ← LabelMe → 바이너리 마스크
│   ├── generate_prelabels.py        ← 사전 라벨링 자동화
│   ├── analyze_fillet.py            ← 필렛 형상 분석 (연구용)
│   └── ...
│
├── annotations/            [GT 데이터]
│   ├── gt_masks/           ← images_main용 마스크
│   ├── gt_masks_gull/      ← images_gull용 마스크
│   └── gt_masks_tnmx/      ← TNMX용 마스크
│
└── images_main/, images_gull/, images_TNMX/  [원본 이미지]
```

## 배포 도구 (deploy/spi.py) 현황
- ✅ 단일 이미지 + 배치 추론
- ✅ CLAHE 전처리 포함 (학습과 동일)
- ✅ 4-panel 시각화 (오버레이/Height/Slope/Profile)
- ✅ JSON/CSV 결과 출력 (영어)
- ✅ ONNX 내보내기 + ONNX Runtime 추론
- ✅ 모델 버전 관리 (update-model)
- ✅ 상대 경로 — 폴더 통째로 이동 가능

## 미완료 / 향후 작업
1. **Val 데이터 정리**: 노이즈 라벨 검수 후 재학습 → IoU 추가 개선 기대
2. **운영자용 UX**: analyze_fillet.py에 컬러 퍼센트(R/G/B %) + Excel 내보내기
3. **추가 학습 데이터**: 불량 샘플 확보 → Pass/Fail 판정 모델로 발전

## 기술 스택
- Python 3.10+, PyTorch, segmentation_models_pytorch, albumentations
- OpenCV, NumPy, ONNX Runtime (엣지 배포)
- VT-S730 AOI: 4MP 카메라, FOV 30×30.72mm, ~15µm/pixel
