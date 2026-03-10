"""
솔더 필렛 형상 분석
==================
U-Net 세그멘테이션 마스크 내에서 RGB 기반 3D 프로파일 분석.

VT-S730 AOI 3D 높이맵:
  Blue 우세  → 높은 표면 (솔더 중심부)
  Green 우세 → 중간 높이 (경사면)
  Red 우세   → 낮은 표면 (가장자리/PCB)

측정 항목:
  - 면적 (mm²)
  - 높이 프록시: B/(R+G+B) → 0~1 스케일
  - 경사도: Sobel gradient magnitude
  - 중심-가장자리 비율
  - RGB 영역 비율 (%)
  - 프로파일 단면 (가로/세로)

사용법:
    python scripts/analyze_fillet.py                         # 3개 폴더 전부
    python scripts/analyze_fillet.py images_gull             # 특정 폴더만
    python scripts/analyze_fillet.py 20                      # 폴더당 20장
    python scripts/analyze_fillet.py --use-gt                # GT 마스크 사용
    python scripts/analyze_fillet.py --no-charts             # 차트 생략
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import cv2 as cv
from pathlib import Path

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spi_core import (
    apply_clahe, compute_height_proxy, compute_slope,
    analyze_region, create_visualization,
    generate_scatter_charts, generate_histograms,
    generate_rgb_distribution, generate_thumbnail_grid,
)

MODEL_PATH = PROJECT_ROOT / "dl" / "models" / "best_model.pth"
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VT-S730 AOI 해상도
PIXEL_SIZE_UM = 15.0

# 전체 이미지 소스 폴더 (3개 데이터셋)
ALL_SOURCE_DIRS = ["images_main", "images_gull", "images_TNMX"]

# GT 마스크 폴더 매핑 (--use-gt 용)
GT_MASK_MAP = {
    "images_main": "annotations/gt_masks",
    "images_gull": "annotations/gt_masks_gull",
    "images_TNMX": "annotations/gt_masks_tnmx",
}


# ============================================================
# U-Net 추론
# ============================================================
def load_model():
    """U-Net 모델 로드."""
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"모델 로드: epoch {checkpoint['epoch']}, Val IoU={checkpoint['val_iou']:.4f}")
    return model


def get_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


@torch.no_grad()
def predict_mask(model, img_bgr, transform):
    """이미지에서 바이너리 마스크 예측 (CLAHE 적용)."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_clahe = apply_clahe(img_rgb)

    augmented = transform(image=img_clahe, mask=np.zeros((h, w), dtype=np.float32))
    img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    return cv.resize(pred_mask, (w, h), interpolation=cv.INTER_NEAREST)


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="솔더 필렛 형상 분석")
    parser.add_argument("source_dir", nargs="?", default=None,
                        help="소스 이미지 디렉토리 (미지정시 3개 전부)")
    parser.add_argument("limit", nargs="?", type=int, default=0,
                        help="폴더당 분석할 최대 이미지 수 (0=전체)")
    parser.add_argument("--use-gt", action="store_true",
                        help="U-Net 대신 GT 마스크 사용")
    parser.add_argument("--output", default=None,
                        help="출력 기본 디렉토리 (기본: dl/analysis/)")
    parser.add_argument("--no-charts", action="store_true",
                        help="차트 생성 생략 (matplotlib 필요)")
    args = parser.parse_args()

    # 처리할 폴더 목록
    if args.source_dir:
        dir_names = [args.source_dir]
    else:
        dir_names = ALL_SOURCE_DIRS

    # 타임스탬프 run 폴더
    output_base = Path(args.output) if args.output else PROJECT_ROOT / "dl" / "analysis"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / run_stamp
    analysis_dir = run_dir / "analysis"
    os.makedirs(analysis_dir, exist_ok=True)

    print("=" * 60)
    print("솔더 필렛 형상 분석")
    print("=" * 60)
    print(f"대상: {', '.join(dir_names)}")
    print(f"출력: {run_dir}/")
    print(f"마스크: {'GT' if args.use_gt else 'U-Net 예측'}")

    # U-Net 모델 로드 (GT 모드가 아닌 경우)
    model = None
    transform = None
    if not args.use_gt:
        model = load_model()
        transform = get_transform()

    # 전체 이미지 수집 (3개 소스 합산)
    img_entries = []  # (img_path, gt_mask_dir)
    for dir_name in dir_names:
        source_dir = PROJECT_ROOT / dir_name
        if not source_dir.exists():
            print(f"\n[{dir_name}] 폴더 없음 → 건너뜀")
            continue
        gt_mask_dir = PROJECT_ROOT / GT_MASK_MAP.get(dir_name, "annotations/gt_masks")
        paths = sorted(source_dir.glob("*.png"))
        if args.limit > 0:
            paths = paths[:args.limit]
        for p in paths:
            img_entries.append((p, gt_mask_dir))
        print(f"  [{dir_name}] {len(paths)}장")

    print(f"  합계: {len(img_entries)}장")

    # 분석 실행
    results = []
    overlays = []  # 썸네일 격자용

    for i, (img_path, gt_mask_dir) in enumerate(img_entries):
        img_bgr = cv.imread(str(img_path))
        if img_bgr is None:
            continue

        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        # 마스크 획득
        if args.use_gt:
            mask_path = gt_mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                continue
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
        else:
            mask = predict_mask(model, img_bgr, transform)

        # 형상 분석
        analysis = analyze_region(img_rgb, mask, pixel_size_um=PIXEL_SIZE_UM)
        analysis["stem"] = img_path.stem

        if not analysis["valid"]:
            continue

        results.append(analysis)

        # 시각화 저장 (analysis/ 서브폴더)
        vis = create_visualization(img_bgr, mask, analysis, img_path.stem)
        cv.imwrite(str(analysis_dir / f"{img_path.stem}_analysis.png"), vis)

        # 오버레이 썸네일용
        mask_bool = mask > 127
        ov = img_bgr.copy()
        ov_color = np.zeros_like(img_bgr)
        ov_color[mask_bool] = [0, 255, 0]
        ov = cv.addWeighted(ov, 0.7, ov_color, 0.3, 0)
        overlays.append((img_path.stem, ov))

        if (i + 1) % 100 == 0 or i < 3:
            hp = analysis["height_proxy"]
            print(f"    [{i+1}/{len(img_entries)}] {img_path.stem} → "
                  f"{analysis['area_mm2']:.4f}mm2, "
                  f"H={hp['mean']:.3f}, CE={hp['center_edge_ratio']:.2f}")

    # 결과 저장
    if results:
        summary = [{k: v for k, v in r.items() if k != "profile"} for r in results]
        with open(run_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(run_dir / "analysis_full.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 통계 출력
    if results:
        areas = [r["area_mm2"] for r in results]
        heights = [r["height_proxy"]["mean"] for r in results]
        ce_ratios = [r["height_proxy"]["center_edge_ratio"] for r in results]
        slopes = [r["slope"]["mean"] for r in results]

        print(f"\n{'=' * 60}")
        print(f"분석 완료: {len(results)}장")
        print(f"{'=' * 60}")
        print(f"  면적:   평균 {np.mean(areas):.4f} mm2 (std {np.std(areas):.4f})")
        print(f"  높이:   평균 {np.mean(heights):.4f} (std {np.std(heights):.4f})")
        print(f"  CE비율: 평균 {np.mean(ce_ratios):.4f} (std {np.std(ce_ratios):.4f})")
        print(f"  경사도: 평균 {np.mean(slopes):.4f} (std {np.std(slopes):.4f})")

        # 차트 생성
        if not args.no_charts:
            print("\n차트 생성 중...")
            generate_scatter_charts(results, run_dir)
            generate_histograms(results, run_dir)
            generate_rgb_distribution(results, run_dir)
            generate_thumbnail_grid(overlays, run_dir)
            print("  차트 저장 완료")

        print(f"\n출력: {run_dir}/")
    else:
        print("\n분석 가능한 이미지가 없습니다.")


if __name__ == "__main__":
    main()
