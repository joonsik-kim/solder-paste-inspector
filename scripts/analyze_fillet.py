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
  - 프로파일 단면 (가로/세로)

사용법:
    python scripts/analyze_fillet.py                         # images_main 전체
    python scripts/analyze_fillet.py images_gull 20           # gull 20장
    python scripts/analyze_fillet.py images_main --use-gt     # GT 마스크 사용
"""

import os
import sys
import json
import argparse
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
MODEL_PATH = PROJECT_ROOT / "dl" / "models" / "best_model.pth"
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VT-S730 AOI 해상도
PIXEL_SIZE_UM = 15.0
PIXEL_AREA_MM2 = (PIXEL_SIZE_UM / 1000.0) ** 2


# ============================================================
# 전처리 (학습과 동일)
# ============================================================
def apply_clahe(img_rgb, clip_limit=3.0, grid_size=(8, 8)):
    """LAB 색공간 L채널에 CLAHE 적용."""
    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv.cvtColor(lab, cv.COLOR_LAB2RGB)


# ============================================================
# 높이/경사 분석
# ============================================================
def compute_height_proxy(img_rgb):
    """RGB에서 높이 프록시 계산. B/(R+G+B) → 0~1."""
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    b = img_rgb[:, :, 2].astype(np.float64)
    total = r + g + b + 1e-6
    return b / total


def compute_slope(height_map):
    """높이맵에서 경사도와 방향 계산."""
    h32 = height_map.astype(np.float32)
    gx = cv.Sobel(h32, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(h32, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)
    return magnitude, direction


def analyze_region(img_rgb, mask):
    """세그멘테이션된 영역 내에서 형상 분석.

    Args:
        img_rgb: 원본 RGB 이미지 (CLAHE 미적용)
        mask: 바이너리 마스크 (0/255)

    Returns:
        분석 결과 dict
    """
    h, w = mask.shape[:2]
    mask_bool = mask > 127

    solder_px = int(np.count_nonzero(mask_bool))
    if solder_px < 5:
        return {"area_px": 0, "area_mm2": 0.0, "valid": False}

    area_mm2 = solder_px * PIXEL_AREA_MM2

    # 높이 프록시
    height = compute_height_proxy(img_rgb)
    height_in_mask = height[mask_bool]

    # 경사도
    slope_mag, slope_dir = compute_slope(height)
    slope_in_mask = slope_mag[mask_bool]

    # 중심-가장자리 분석
    # 마스크의 distance transform으로 중심/가장자리 구분
    dist = cv.distanceTransform(mask, cv.DIST_L2, 3)
    if dist.max() > 0:
        dist_norm = dist / dist.max()
    else:
        dist_norm = dist

    center_mask = (dist_norm > 0.5) & mask_bool
    edge_mask = (dist_norm <= 0.5) & (dist_norm > 0) & mask_bool

    center_h = float(np.mean(height[center_mask])) if np.any(center_mask) else 0.0
    edge_h = float(np.mean(height[edge_mask])) if np.any(edge_mask) else 0.0
    ce_ratio = center_h / (edge_h + 1e-6)

    # 프로파일 단면 (무게중심 기준)
    ys, xs = np.where(mask_bool)
    cy, cx = int(np.mean(ys)), int(np.mean(xs))

    # 가로 단면
    h_profile = []
    if 0 <= cy < h:
        row = mask_bool[cy, :]
        cols = np.where(row)[0]
        if len(cols) > 0:
            h_profile = height[cy, cols[0]:cols[-1] + 1].tolist()

    # 세로 단면
    v_profile = []
    if 0 <= cx < w:
        col = mask_bool[:, cx]
        rows = np.where(col)[0]
        if len(rows) > 0:
            v_profile = height[rows[0]:rows[-1] + 1, cx].tolist()

    # 경사 균일도 (낮을수록 균일)
    slope_uniformity = float(1.0 - np.std(slope_in_mask) / (np.mean(slope_in_mask) + 1e-6))
    slope_uniformity = max(0.0, min(1.0, slope_uniformity))

    return {
        "valid": True,
        "area_px": solder_px,
        "area_mm2": round(area_mm2, 6),
        "height_proxy": {
            "mean": round(float(np.mean(height_in_mask)), 4),
            "std": round(float(np.std(height_in_mask)), 4),
            "min": round(float(np.min(height_in_mask)), 4),
            "max": round(float(np.max(height_in_mask)), 4),
            "center": round(center_h, 4),
            "edge": round(edge_h, 4),
            "center_edge_ratio": round(ce_ratio, 4),
        },
        "slope": {
            "mean": round(float(np.mean(slope_in_mask)), 4),
            "max": round(float(np.max(slope_in_mask)), 4),
            "uniformity": round(slope_uniformity, 4),
        },
        "profile": {
            "horizontal": [round(v, 4) for v in h_profile],
            "vertical": [round(v, 4) for v in v_profile],
        },
        "centroid": {"x": cx, "y": cy},
    }


# ============================================================
# 시각화
# ============================================================
def create_visualization(img_orig_bgr, mask, analysis, stem):
    """분석 결과 종합 시각화 이미지 생성.

    Returns:
        비교 이미지 (BGR)
    """
    h, w = img_orig_bgr.shape[:2]
    img_rgb = cv.cvtColor(img_orig_bgr, cv.COLOR_BGR2RGB)
    mask_bool = mask > 127

    # 1. 원본 + 오버레이
    overlay = img_orig_bgr.copy()
    overlay_color = np.zeros_like(img_orig_bgr)
    overlay_color[mask_bool] = [0, 255, 0]
    panel_overlay = cv.addWeighted(overlay, 0.7, overlay_color, 0.3, 0)

    # 2. 높이맵 히트맵
    height = compute_height_proxy(img_rgb)
    height_vis = np.zeros((h, w), dtype=np.uint8)
    if mask_bool.any():
        h_min = height[mask_bool].min()
        h_max = height[mask_bool].max()
        if h_max > h_min:
            height_norm = ((height - h_min) / (h_max - h_min) * 255).clip(0, 255).astype(np.uint8)
        else:
            height_norm = np.full((h, w), 128, dtype=np.uint8)
        height_vis[mask_bool] = height_norm[mask_bool]

    height_color = cv.applyColorMap(height_vis, cv.COLORMAP_JET)
    height_color[~mask_bool] = [0, 0, 0]

    # 3. 경사도 맵
    slope_mag, _ = compute_slope(height)
    slope_vis = np.zeros((h, w), dtype=np.uint8)
    if mask_bool.any():
        s_max = slope_mag[mask_bool].max()
        if s_max > 0:
            slope_norm = (slope_mag / s_max * 255).clip(0, 255).astype(np.uint8)
        else:
            slope_norm = np.zeros((h, w), dtype=np.uint8)
        slope_vis[mask_bool] = slope_norm[mask_bool]

    slope_color = cv.applyColorMap(slope_vis, cv.COLORMAP_HOT)
    slope_color[~mask_bool] = [0, 0, 0]

    # 4. 프로파일 시각화 (간단한 바 형태)
    profile_img = np.zeros_like(img_orig_bgr)
    hp = analysis.get("profile", {}).get("horizontal", [])
    if hp and len(hp) > 1:
        bar_w = max(1, w // len(hp))
        for i, val in enumerate(hp):
            bar_h = int(val * h * 0.9)
            x1 = i * bar_w
            x2 = min(x1 + bar_w - 1, w - 1)
            y1 = h - bar_h
            color_val = int(val * 255)
            cv.rectangle(profile_img, (x1, y1), (x2, h - 1),
                         (color_val, 255 - color_val, 0), -1)

    # 스케일업
    scale = max(1, 200 // max(h, w))
    panels = []
    for p in [panel_overlay, height_color, slope_color, profile_img]:
        panels.append(cv.resize(p, (w * scale, h * scale), interpolation=cv.INTER_NEAREST))

    comparison = np.hstack(panels)

    # 라벨 추가
    labels = ["Original+Mask", "Height Proxy", "Slope", "Profile"]
    for i, label in enumerate(labels):
        x = i * w * scale + 3
        cv.putText(comparison, label, (x, 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # 면적/높이 정보
    info = analysis.get("height_proxy", {})
    text = f"{analysis['area_mm2']:.4f}mm2 | H:{info.get('mean', 0):.3f} CE:{info.get('center_edge_ratio', 0):.2f}"
    cv.putText(comparison, text, (3, comparison.shape[0] - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return comparison


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
    parser.add_argument("source_dir", nargs="?", default="images_main",
                        help="소스 이미지 디렉토리 (기본: images_main)")
    parser.add_argument("limit", nargs="?", type=int, default=0,
                        help="분석할 최대 이미지 수 (0=전체)")
    parser.add_argument("--use-gt", action="store_true",
                        help="U-Net 대신 GT 마스크 사용")
    parser.add_argument("--output", default=None,
                        help="출력 디렉토리 (기본: dl/analysis/)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir

    dir_name = source_dir.name  # images_main or images_gull

    # 출력 디렉토리
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "dl" / "analysis" / dir_name
    os.makedirs(output_dir, exist_ok=True)

    # GT 마스크 디렉토리
    if dir_name == "images_gull":
        gt_mask_dir = PROJECT_ROOT / "annotations" / "gt_masks_gull"
    else:
        gt_mask_dir = PROJECT_ROOT / "annotations" / "gt_masks"

    print("=" * 60)
    print("솔더 필렛 형상 분석")
    print("=" * 60)
    print(f"소스: {source_dir}")
    print(f"출력: {output_dir}")
    print(f"마스크: {'GT' if args.use_gt else 'U-Net 예측'}")

    # 이미지 수집
    img_paths = sorted(source_dir.glob("*.png"))
    if args.limit > 0:
        img_paths = img_paths[:args.limit]
    print(f"분석 대상: {len(img_paths)}장")

    # U-Net 모델 로드 (GT 모드가 아닌 경우)
    model = None
    transform = None
    if not args.use_gt:
        model = load_model()
        transform = get_transform()

    # 분석 실행
    results = []
    for i, img_path in enumerate(img_paths):
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
        analysis = analyze_region(img_rgb, mask)
        analysis["stem"] = img_path.stem

        if not analysis["valid"]:
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(img_paths)}] {img_path.stem} → 솔더 미검출")
            continue

        results.append(analysis)

        # 시각화 저장
        vis = create_visualization(img_bgr, mask, analysis, img_path.stem)
        cv.imwrite(str(output_dir / f"{img_path.stem}_analysis.png"), vis)

        if (i + 1) % 500 == 0 or i < 3:
            hp = analysis["height_proxy"]
            print(f"  [{i+1}/{len(img_paths)}] {img_path.stem} → "
                  f"{analysis['area_mm2']:.4f}mm², "
                  f"H={hp['mean']:.3f}, CE={hp['center_edge_ratio']:.2f}")

    # 결과 저장
    if results:
        # 프로파일 데이터가 너무 크므로 요약본과 상세본 분리
        summary = []
        for r in results:
            s = {k: v for k, v in r.items() if k != "profile"}
            summary.append(s)

        with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(output_dir / "analysis_full.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 통계
        areas = [r["area_mm2"] for r in results]
        heights = [r["height_proxy"]["mean"] for r in results]
        ce_ratios = [r["height_proxy"]["center_edge_ratio"] for r in results]
        slopes = [r["slope"]["mean"] for r in results]

        print(f"\n{'=' * 60}")
        print(f"분석 완료: {len(results)}장")
        print(f"{'=' * 60}")
        print(f"  면적:     평균 {np.mean(areas):.4f} mm² (std {np.std(areas):.4f})")
        print(f"  높이:     평균 {np.mean(heights):.4f} (std {np.std(heights):.4f})")
        print(f"  CE비율:   평균 {np.mean(ce_ratios):.4f} (std {np.std(ce_ratios):.4f})")
        print(f"  경사도:   평균 {np.mean(slopes):.4f} (std {np.std(slopes):.4f})")
        print(f"\n출력: {output_dir}")
    else:
        print("\n분석 가능한 이미지가 없습니다.")


if __name__ == "__main__":
    main()
