"""
SPI Core — 솔더 필렛 분석 공유 모듈
====================================
전처리 및 형상 분석 함수 (numpy/cv2만 사용, torch 의존성 없음).

사용처:
  - scripts/analyze_fillet.py
  - dl/predict.py
  - dl/train.py

Note: deploy/spi.py는 자체 완결 배포 도구이므로 이 모듈을 import하지 않고
      자체 복사본을 유지합니다 (deploy/ 폴더만 복사하면 바로 실행 가능).
"""

from pathlib import Path

import numpy as np
import cv2 as cv


# ============================================================
# 전처리 (학습/추론 공통)
# ============================================================
def apply_clahe(img_rgb, clip_limit=3.0, grid_size=(8, 8)):
    """LAB 색공간 L채널에 CLAHE 적용. 학습·추론 동일하게 사용.

    Args:
        img_rgb: RGB 이미지 (numpy array)
        clip_limit: CLAHE clip limit (default: 3.0)
        grid_size: CLAHE tile grid size (default: (8, 8))

    Returns:
        CLAHE 적용된 RGB 이미지
    """
    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv.cvtColor(lab, cv.COLOR_LAB2RGB)


# ============================================================
# 높이/경사 분석
# ============================================================
def compute_height_proxy(img_rgb):
    """RGB에서 높이 프록시 계산. B/(R+G+B) → 0~1.

    VT-S730 3D 높이맵 인코딩:
      Blue 우세  → 높은 표면 (솔더 중심부)
      Green 우세 → 중간 높이 (경사면)
      Red 우세   → 낮은 표면 (가장자리/PCB)
    """
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    b = img_rgb[:, :, 2].astype(np.float64)
    total = r + g + b + 1e-6
    return b / total


def compute_slope(height_map):
    """높이맵에서 경사도와 방향 계산 (Sobel).

    Returns:
        (magnitude, direction) 튜플
    """
    h32 = height_map.astype(np.float32)
    gx = cv.Sobel(h32, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(h32, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    return magnitude, direction


def analyze_region(img_rgb, mask, pixel_size_um=15.0):
    """세그멘테이션된 영역 내에서 형상 분석.

    Args:
        img_rgb: 원본 RGB 이미지 (CLAHE 미적용)
        mask: 바이너리 마스크 (0/255)
        pixel_size_um: 픽셀 크기 (µm). VT-S730 기본 15.0

    Returns:
        분석 결과 dict
    """
    h, w = mask.shape[:2]
    mask_bool = mask > 127
    pixel_area_mm2 = (pixel_size_um / 1000.0) ** 2

    solder_px = int(np.count_nonzero(mask_bool))
    if solder_px < 5:
        return {"area_px": 0, "area_mm2": 0.0, "valid": False}

    area_mm2 = solder_px * pixel_area_mm2

    # 높이 프록시
    height = compute_height_proxy(img_rgb)
    height_in_mask = height[mask_bool]

    # 경사도
    slope_mag, slope_dir = compute_slope(height)
    slope_in_mask = slope_mag[mask_bool]

    # 중심-가장자리 분석 (distance transform)
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

    h_profile = []
    if 0 <= cy < h:
        row = mask_bool[cy, :]
        cols = np.where(row)[0]
        if len(cols) > 0:
            h_profile = height[cy, cols[0] : cols[-1] + 1].tolist()

    v_profile = []
    if 0 <= cx < w:
        col = mask_bool[:, cx]
        rows = np.where(col)[0]
        if len(rows) > 0:
            v_profile = height[rows[0] : rows[-1] + 1, cx].tolist()

    slope_uniformity = float(
        1.0 - np.std(slope_in_mask) / (np.mean(slope_in_mask) + 1e-6)
    )
    slope_uniformity = max(0.0, min(1.0, slope_uniformity))

    # RGB 영역 비율 (컬러 존)
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    b = img_rgb[:, :, 2].astype(np.float64)
    r_mask = r[mask_bool]
    g_mask = g[mask_bool]
    b_mask = b[mask_bool]
    red_dom = np.sum((r_mask > g_mask) & (r_mask > b_mask))
    green_dom = np.sum((g_mask > r_mask) & (g_mask > b_mask))
    blue_dom = np.sum((b_mask > r_mask) & (b_mask > g_mask))
    total_dom = red_dom + green_dom + blue_dom + 1e-6
    red_pct = round(float(red_dom / total_dom * 100), 1)
    green_pct = round(float(green_dom / total_dom * 100), 1)
    blue_pct = round(float(blue_dom / total_dom * 100), 1)

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
        "color_zone": {
            "red_pct": red_pct,
            "green_pct": green_pct,
            "blue_pct": blue_pct,
        },
    }


# ============================================================
# 시각화
# ============================================================
def create_visualization(img_orig_bgr, mask, analysis, stem):
    """분석 결과 4패널 시각화 이미지 생성.

    패널: Original+Overlay | Height Heatmap | Slope Map | Profile Bars
    """
    h, w = img_orig_bgr.shape[:2]
    img_rgb = cv.cvtColor(img_orig_bgr, cv.COLOR_BGR2RGB)
    mask_bool = mask > 127

    # Panel 1: Original + mask overlay
    overlay = img_orig_bgr.copy()
    overlay_color = np.zeros_like(img_orig_bgr)
    overlay_color[mask_bool] = [0, 255, 0]
    panel_overlay = cv.addWeighted(overlay, 0.7, overlay_color, 0.3, 0)

    # Panel 2: Height heatmap (JET: red=high, blue=low)
    height = compute_height_proxy(img_rgb)
    height_vis = np.zeros((h, w), dtype=np.uint8)
    if mask_bool.any():
        h_min = height[mask_bool].min()
        h_max = height[mask_bool].max()
        if h_max > h_min:
            height_norm = (
                ((height - h_min) / (h_max - h_min) * 255).clip(0, 255).astype(np.uint8)
            )
        else:
            height_norm = np.full((h, w), 128, dtype=np.uint8)
        height_vis[mask_bool] = height_norm[mask_bool]

    height_color = cv.applyColorMap(height_vis, cv.COLORMAP_JET)
    height_color[~mask_bool] = [0, 0, 0]

    # Panel 3: Slope map (HOT colormap)
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

    # Panel 4: Profile bars
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
            cv.rectangle(
                profile_img, (x1, y1), (x2, h - 1), (color_val, 255 - color_val, 0), -1
            )

    # Scale up for visibility
    scale = max(1, 200 // max(h, w))
    panels = []
    for p in [panel_overlay, height_color, slope_color, profile_img]:
        panels.append(
            cv.resize(p, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)
        )

    comparison = np.hstack(panels)

    # Panel labels
    labels = ["Original+Mask", "Height Proxy", "Slope", "Profile"]
    for i, label in enumerate(labels):
        x = i * w * scale + 3
        cv.putText(
            comparison, label, (x, 12), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1
        )

    # Bottom info text
    info = analysis.get("height_proxy", {})
    text = (
        f"{analysis['area_mm2']:.4f}mm2 | "
        f"H:{info.get('mean', 0):.3f} "
        f"CE:{info.get('center_edge_ratio', 0):.2f}"
    )
    cv.putText(
        comparison,
        text,
        (3, comparison.shape[0] - 5),
        cv.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
    )

    return comparison


# ============================================================
# 배치 통계 차트 (matplotlib, lazy import)
# ============================================================
def _get_plt():
    """matplotlib.pyplot lazy import. None 반환 시 미설치."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  [WARN] matplotlib 미설치, 차트 생략")
        return None


def generate_scatter_charts(results, output_dir):
    """산포도 2개: 면적vs높이, CE비율vs균일도."""
    plt = _get_plt()
    if plt is None or len(results) < 2:
        return

    areas = [r["area_mm2"] for r in results]
    heights = [r["height_proxy"]["mean"] for r in results]
    ce_ratios = [r["height_proxy"]["center_edge_ratio"] for r in results]
    uniformities = [r["slope"]["uniformity"] for r in results]

    # 면적 vs 높이
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(areas, heights, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax.set_xlabel("Area (mm2)")
    ax.set_ylabel("Height Proxy Mean")
    ax.set_title("Area vs Height Proxy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "scatter_area_vs_height.png"), dpi=150)
    plt.close(fig)

    # CE비율 vs 균일도
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ce_ratios, uniformities, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax.set_xlabel("Center-Edge Ratio")
    ax.set_ylabel("Slope Uniformity")
    ax.set_title("CE Ratio vs Slope Uniformity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "scatter_ce_vs_uniformity.png"), dpi=150)
    plt.close(fig)


def generate_histograms(results, output_dir):
    """2x2 히스토그램: 면적, 높이, CE비율, 경사도."""
    plt = _get_plt()
    if plt is None or len(results) < 2:
        return

    metrics = [
        ([r["area_mm2"] for r in results], "Area (mm2)", "steelblue"),
        ([r["height_proxy"]["mean"] for r in results], "Height Proxy Mean", "darkorange"),
        ([r["height_proxy"]["center_edge_ratio"] for r in results], "CE Ratio", "seagreen"),
        ([r["slope"]["mean"] for r in results], "Slope Mean", "indianred"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (data, label, color) in zip(axes.flat, metrics):
        bins = min(30, max(5, len(data) // 3))
        ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}")
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "histograms.png"), dpi=150)
    plt.close(fig)


def generate_rgb_distribution(results, output_dir):
    """RGB 영역 비율 스택바 차트."""
    plt = _get_plt()
    if plt is None or len(results) < 1:
        return

    # color_zone이 없는 결과 필터링
    valid = [r for r in results if "color_zone" in r]
    if not valid:
        return

    stems = [r.get("stem", str(i)) for i, r in enumerate(valid)]
    reds = [r["color_zone"]["red_pct"] for r in valid]
    greens = [r["color_zone"]["green_pct"] for r in valid]
    blues = [r["color_zone"]["blue_pct"] for r in valid]

    n = len(valid)
    x = np.arange(n)

    fig_w = max(8, n * 0.3)
    fig, ax = plt.subplots(figsize=(min(fig_w, 40), 6))

    ax.bar(x, blues, label="Blue (High)", color="royalblue", alpha=0.85)
    ax.bar(x, greens, bottom=blues, label="Green (Mid)", color="limegreen", alpha=0.85)
    bottoms = [b + g for b, g in zip(blues, greens)]
    ax.bar(x, reds, bottom=bottoms, label="Red (Low)", color="tomato", alpha=0.85)

    if n <= 30:
        ax.set_xticks(x)
        ax.set_xticklabels(stems, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"Images ({n} total)")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("RGB Color Zone Distribution per Fillet")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "rgb_distribution.png"), dpi=150)
    plt.close(fig)


def generate_thumbnail_grid(overlay_images, output_dir, max_count=100):
    """오버레이 이미지들을 격자로 한 장에 모아서 저장.

    Args:
        overlay_images: list of (stem, bgr_image) 튜플
        output_dir: 출력 디렉토리
        max_count: 최대 이미지 수 (기본 100)
    """
    if not overlay_images:
        return

    items = overlay_images[:max_count]
    n = len(items)
    cols = min(10, n)
    rows = (n + cols - 1) // cols

    thumb_size = 128
    grid = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)

    for idx, (stem, img) in enumerate(items):
        r, c = divmod(idx, cols)
        thumb = cv.resize(img, (thumb_size, thumb_size))
        y1 = r * thumb_size
        x1 = c * thumb_size
        grid[y1 : y1 + thumb_size, x1 : x1 + thumb_size] = thumb

        # 파일명 표시
        cv.putText(
            grid, stem[:15], (x1 + 2, y1 + thumb_size - 5),
            cv.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1,
        )

    cv.imwrite(str(Path(output_dir) / "thumbnail_grid.png"), grid)
