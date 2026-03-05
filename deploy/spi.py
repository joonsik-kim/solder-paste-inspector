"""
SPI v1.0 - Solder Paste Inspector (Edge Deployment Tool)
=========================================================
Self-contained CLI for solder fillet segmentation, shape analysis,
and model management on edge devices.

This folder is fully portable: copy deploy/ to any machine and run.

Usage:
    python spi.py predict image.png
    python spi.py predict images_folder/ --output results/
    python spi.py export --format onnx
    python spi.py update-model path/to/new_model.pth
    python spi.py info

Deployment:
    1. Copy this entire deploy/ folder to edge device
    2. Install packages: pip install --no-index --find-links=./packages -r requirements.txt
    3. Run: python spi.py predict <image>
"""

import os
import sys
import json
import csv
import shutil
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2 as cv

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# Constants - All paths relative to this script's directory
# ============================================================
VERSION = "1.0.0"
DEPLOY_DIR = Path(__file__).resolve().parent
MODEL_DIR = DEPLOY_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.pth"
REGISTRY_PATH = DEPLOY_DIR / "model_registry.json"
IMG_SIZE = 128

# VT-S730 AOI default resolution
DEFAULT_PIXEL_SIZE_UM = 15.0

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ============================================================
# Preprocessing (identical to training pipeline)
# ============================================================
def apply_clahe(img_rgb, clip_limit=3.0, grid_size=(8, 8)):
    """Apply CLAHE on LAB L-channel. Must match training preprocessing."""
    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv.cvtColor(lab, cv.COLOR_LAB2RGB)


# ============================================================
# Height / Slope Analysis
# ============================================================
def compute_height_proxy(img_rgb):
    """Compute height proxy from RGB. B/(R+G+B) -> 0~1 scale.

    VT-S730 3D height map encoding:
      Blue-dominant  -> high surface (solder center)
      Green-dominant -> mid height (slope)
      Red-dominant   -> low surface (edge/PCB)
    """
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    b = img_rgb[:, :, 2].astype(np.float64)
    total = r + g + b + 1e-6
    return b / total


def compute_slope(height_map):
    """Compute slope magnitude and direction from height map using Sobel."""
    h32 = height_map.astype(np.float32)
    gx = cv.Sobel(h32, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(h32, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    return magnitude, direction


def analyze_region(img_rgb, mask, pixel_size_um=DEFAULT_PIXEL_SIZE_UM):
    """Comprehensive shape analysis within segmented region.

    Args:
        img_rgb: Original RGB image (no CLAHE applied)
        mask: Binary mask (0/255)
        pixel_size_um: Pixel size in micrometers

    Returns:
        Analysis result dict
    """
    h, w = mask.shape[:2]
    mask_bool = mask > 127
    pixel_area_mm2 = (pixel_size_um / 1000.0) ** 2

    solder_px = int(np.count_nonzero(mask_bool))
    if solder_px < 5:
        return {"area_px": 0, "area_mm2": 0.0, "valid": False}

    area_mm2 = solder_px * pixel_area_mm2

    # Height proxy
    height = compute_height_proxy(img_rgb)
    height_in_mask = height[mask_bool]

    # Slope
    slope_mag, slope_dir = compute_slope(height)
    slope_in_mask = slope_mag[mask_bool]

    # Center-edge analysis via distance transform
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

    # Profile cross-sections (centroid-based)
    ys, xs = np.where(mask_bool)
    cy, cx = int(np.mean(ys)), int(np.mean(xs))

    # Horizontal profile
    h_profile = []
    if 0 <= cy < h:
        row = mask_bool[cy, :]
        cols = np.where(row)[0]
        if len(cols) > 0:
            h_profile = height[cy, cols[0] : cols[-1] + 1].tolist()

    # Vertical profile
    v_profile = []
    if 0 <= cx < w:
        col = mask_bool[:, cx]
        rows = np.where(col)[0]
        if len(rows) > 0:
            v_profile = height[rows[0] : rows[-1] + 1, cx].tolist()

    # Slope uniformity (lower = more uniform)
    slope_uniformity = float(
        1.0 - np.std(slope_in_mask) / (np.mean(slope_in_mask) + 1e-6)
    )
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
# Visualization
# ============================================================
def create_visualization(img_orig_bgr, mask, analysis, stem):
    """Create 4-panel analysis visualization image.

    Panels: Original+Overlay | Height Heatmap | Slope Map | Profile Bars
    """
    h, w = img_orig_bgr.shape[:2]
    img_rgb = cv.cvtColor(img_orig_bgr, cv.COLOR_BGR2RGB)
    mask_bool = mask > 127

    # Panel 1: Original + mask overlay
    overlay = img_orig_bgr.copy()
    overlay_color = np.zeros_like(img_orig_bgr)
    overlay_color[mask_bool] = [0, 255, 0]
    panel_overlay = cv.addWeighted(overlay, 0.7, overlay_color, 0.3, 0)

    # Panel 2: Height heatmap
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

    # Panel 3: Slope map
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
# Model Loading
# ============================================================
def get_device(device_str=None):
    """Resolve device string to torch.device."""
    if device_str and device_str != "auto":
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path=None, device=None):
    """Load trained U-Net model from checkpoint."""
    if model_path is None:
        model_path = MODEL_PATH
    model_path = Path(model_path)

    if device is None:
        device = get_device()

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_iou = checkpoint.get("val_iou", 0.0)
    print(f"[MODEL] Loaded: {model_path.name} (epoch {epoch}, Val IoU={val_iou:.4f})")

    return model, checkpoint


def get_transform():
    """Get inference transform (must match training)."""
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def predict_mask(model, img_bgr, transform, device, threshold=0.5):
    """Predict binary mask from image with CLAHE preprocessing.

    Pipeline: BGR -> RGB -> CLAHE -> Normalize -> U-Net -> Sigmoid -> Threshold
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_clahe = apply_clahe(img_rgb)

    augmented = transform(image=img_clahe, mask=np.zeros((h, w), dtype=np.float32))
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
    pred_mask = (pred > threshold).astype(np.uint8) * 255
    return cv.resize(pred_mask, (w, h), interpolation=cv.INTER_NEAREST)


def predict_mask_onnx(session, img_bgr, threshold=0.5):
    """Predict binary mask using ONNX Runtime."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_clahe = apply_clahe(img_rgb)

    # Manual preprocessing (same as albumentations)
    img_resized = cv.resize(img_clahe, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_resized - mean) / std
    img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]  # NCHW

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_chw})[0]

    # Sigmoid
    pred = 1.0 / (1.0 + np.exp(-output.squeeze()))
    pred_mask = (pred > threshold).astype(np.uint8) * 255
    return cv.resize(pred_mask, (w, h), interpolation=cv.INTER_NEAREST)


# ============================================================
# Registry
# ============================================================
def load_registry():
    """Load model registry from JSON."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"current_version": 0, "models": []}


def save_registry(registry):
    """Save model registry to JSON."""
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# ============================================================
# Subcommand: predict
# ============================================================
def cmd_predict(args):
    """Run inference + shape analysis on single image or directory."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    device = get_device(args.device)
    threshold = args.threshold
    pixel_size = args.pixel_size
    no_viz = args.no_viz

    # Collect image paths
    if input_path.is_file():
        img_paths = [input_path]
    else:
        img_paths = sorted(
            p
            for p in input_path.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
            and not p.name.startswith(".")
        )
        if args.limit and args.limit > 0:
            img_paths = img_paths[: args.limit]

    if not img_paths:
        print("[ERROR] No images found.")
        sys.exit(1)

    print(f"[SPI] Solder Paste Inspector v{VERSION}")
    print(f"  Input:      {input_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Images:     {len(img_paths)}")
    print(f"  Device:     {device}")
    print(f"  Threshold:  {threshold}")
    print(f"  Pixel size: {pixel_size} um/px")
    print()

    # Load model (PyTorch or ONNX)
    use_onnx = args.runtime == "onnx"
    onnx_session = None

    if use_onnx:
        onnx_path = MODEL_DIR / "spi_model.onnx"
        if not onnx_path.exists():
            print(f"[ERROR] ONNX model not found: {onnx_path}")
            print("  Run 'python spi.py export --format onnx' first.")
            sys.exit(1)
        try:
            import onnxruntime as ort

            onnx_session = ort.InferenceSession(str(onnx_path))
            print(f"[MODEL] ONNX Runtime loaded: {onnx_path.name}")
        except ImportError:
            print("[ERROR] onnxruntime not installed. Install with: pip install onnxruntime")
            sys.exit(1)
        model = None
        transform = None
    else:
        model_path = Path(args.model) if args.model else None
        model, _ = load_model(model_path=model_path, device=device)
        transform = get_transform()

    # Process images
    results = []
    total_time = 0.0

    for i, img_path in enumerate(img_paths):
        t0 = time.time()
        img_bgr = cv.imread(str(img_path))
        if img_bgr is None:
            print(f"  [{i+1}/{len(img_paths)}] {img_path.name} -> FAILED (cannot load)")
            continue

        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        # Inference
        if use_onnx:
            mask = predict_mask_onnx(onnx_session, img_bgr, threshold)
        else:
            mask = predict_mask(model, img_bgr, transform, device, threshold)

        # Shape analysis
        analysis = analyze_region(img_rgb, mask, pixel_size)
        analysis["filename"] = img_path.name
        analysis["image_size"] = {"width": img_bgr.shape[1], "height": img_bgr.shape[0]}

        elapsed = time.time() - t0
        total_time += elapsed
        analysis["inference_time_ms"] = round(elapsed * 1000, 1)

        # Save mask
        stem = img_path.stem
        cv.imwrite(str(output_dir / f"{stem}_mask.png"), mask)

        # Save overlay
        overlay = img_bgr.copy()
        overlay_color = np.zeros_like(img_bgr)
        mask_bool = mask > 127
        overlay_color[mask_bool] = [0, 0, 255]  # Red in BGR
        overlay = cv.addWeighted(overlay, 0.7, overlay_color, 0.3, 0)
        cv.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay)

        # Save analysis visualization
        if not no_viz and analysis.get("valid", False):
            vis = create_visualization(img_bgr, mask, analysis, stem)
            cv.imwrite(str(output_dir / f"{stem}_analysis.png"), vis)

        # Save per-image JSON result
        result_data = {k: v for k, v in analysis.items() if k != "profile"}
        result_data["pixel_size_um"] = pixel_size
        with open(output_dir / f"{stem}_result.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        results.append(analysis)

        # Progress log
        if analysis.get("valid", False):
            hp = analysis["height_proxy"]
            print(
                f"  [{i+1}/{len(img_paths)}] {img_path.name} -> "
                f"{analysis['area_mm2']:.4f} mm2, "
                f"H={hp['mean']:.3f}, CE={hp['center_edge_ratio']:.2f}, "
                f"{elapsed*1000:.0f}ms"
            )
        else:
            area_pct = np.count_nonzero(mask > 127) / mask.size * 100
            print(
                f"  [{i+1}/{len(img_paths)}] {img_path.name} -> "
                f"no solder detected ({area_pct:.1f}% mask), {elapsed*1000:.0f}ms"
            )

    # Batch summary
    valid_results = [r for r in results if r.get("valid", False)]
    print(f"\n{'='*60}")
    print(f"[DONE] {len(results)} images processed in {total_time:.1f}s")
    print(f"  Valid:   {len(valid_results)}/{len(results)}")

    if valid_results:
        areas = [r["area_mm2"] for r in valid_results]
        heights = [r["height_proxy"]["mean"] for r in valid_results]
        ce_ratios = [r["height_proxy"]["center_edge_ratio"] for r in valid_results]

        print(f"  Area:    mean={np.mean(areas):.4f} mm2 (std={np.std(areas):.4f})")
        print(f"  Height:  mean={np.mean(heights):.4f} (std={np.std(heights):.4f})")
        print(f"  CE Ratio: mean={np.mean(ce_ratios):.2f} (std={np.std(ce_ratios):.2f})")

        # Save batch summary JSON
        summary = {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "input": str(input_path),
            "total_images": len(results),
            "valid_images": len(valid_results),
            "pixel_size_um": pixel_size,
            "threshold": threshold,
            "total_time_s": round(total_time, 2),
            "statistics": {
                "area_mm2": {
                    "mean": round(float(np.mean(areas)), 6),
                    "std": round(float(np.std(areas)), 6),
                    "min": round(float(np.min(areas)), 6),
                    "max": round(float(np.max(areas)), 6),
                },
                "height_proxy": {
                    "mean": round(float(np.mean(heights)), 4),
                    "std": round(float(np.std(heights)), 4),
                },
                "center_edge_ratio": {
                    "mean": round(float(np.mean(ce_ratios)), 4),
                    "std": round(float(np.std(ce_ratios)), 4),
                },
            },
            "per_image": [
                {
                    "filename": r["filename"],
                    "area_mm2": r["area_mm2"],
                    "height_mean": r["height_proxy"]["mean"],
                    "ce_ratio": r["height_proxy"]["center_edge_ratio"],
                    "slope_mean": r["slope"]["mean"],
                    "inference_time_ms": r.get("inference_time_ms", 0),
                }
                for r in valid_results
            ],
        }

        with open(output_dir / "batch_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save batch summary CSV
        with open(output_dir / "batch_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "filename",
                    "area_mm2",
                    "area_px",
                    "height_mean",
                    "height_std",
                    "height_center",
                    "height_edge",
                    "ce_ratio",
                    "slope_mean",
                    "slope_max",
                    "slope_uniformity",
                    "centroid_x",
                    "centroid_y",
                    "inference_ms",
                ]
            )
            for r in valid_results:
                writer.writerow(
                    [
                        r["filename"],
                        r["area_mm2"],
                        r["area_px"],
                        r["height_proxy"]["mean"],
                        r["height_proxy"]["std"],
                        r["height_proxy"]["center"],
                        r["height_proxy"]["edge"],
                        r["height_proxy"]["center_edge_ratio"],
                        r["slope"]["mean"],
                        r["slope"]["max"],
                        r["slope"]["uniformity"],
                        r["centroid"]["x"],
                        r["centroid"]["y"],
                        r.get("inference_time_ms", 0),
                    ]
                )

        print(f"\n  Summary: {output_dir / 'batch_summary.json'}")
        print(f"  CSV:     {output_dir / 'batch_summary.csv'}")

    print(f"  Output:  {output_dir}")


# ============================================================
# Subcommand: export
# ============================================================
def cmd_export(args):
    """Export model to ONNX format."""
    if args.format != "onnx":
        print(f"[ERROR] Unsupported format: {args.format}. Only 'onnx' is supported.")
        sys.exit(1)

    device = get_device(args.device)
    model_path = Path(args.model) if args.model else None
    model, checkpoint = load_model(model_path=model_path, device=device)

    # Output path
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)

    output_path = output_dir / "spi_model.onnx"

    # Export
    print(f"[EXPORT] Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["mask"],
        dynamic_axes={"image": {0: "batch"}, "mask": {0: "batch"}},
        opset_version=11,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[DONE] ONNX model saved: {output_path} ({file_size_mb:.1f} MB)")
    print(f"  Input:  [batch, 3, {IMG_SIZE}, {IMG_SIZE}] (float32)")
    print(f"  Output: [batch, 1, {IMG_SIZE}, {IMG_SIZE}] (float32, pre-sigmoid)")
    print(f"\n  Use with: python deploy/spi.py predict image.png --runtime onnx")


# ============================================================
# Subcommand: update-model
# ============================================================
def cmd_update_model(args):
    """Update model with a newly trained checkpoint."""
    new_model_path = Path(args.model_path)
    if not new_model_path.exists():
        print(f"[ERROR] Model file not found: {new_model_path}")
        sys.exit(1)

    # Validate new model
    print(f"[UPDATE] Validating new model: {new_model_path.name}")
    device = get_device()

    try:
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        ).to(device)

        checkpoint = torch.load(new_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Dummy inference test
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        with torch.no_grad():
            output = model(dummy)
        assert output.shape == (1, 1, IMG_SIZE, IMG_SIZE), f"Unexpected output shape: {output.shape}"

        epoch = checkpoint.get("epoch", "?")
        val_iou = checkpoint.get("val_iou", 0.0)
        print(f"  Validation passed: epoch {epoch}, Val IoU={val_iou:.4f}")

    except Exception as e:
        print(f"[ERROR] Model validation failed: {e}")
        sys.exit(1)

    # Load registry
    registry = load_registry()
    current_ver = registry["current_version"]
    new_ver = current_ver + 1

    # Backup current model
    if MODEL_PATH.exists():
        backup_name = f"best_model_v{current_ver}.pth"
        backup_path = MODEL_PATH.parent / backup_name
        shutil.copy2(str(MODEL_PATH), str(backup_path))
        print(f"  Backup: {MODEL_PATH.name} -> {backup_name}")

    # Copy new model
    shutil.copy2(str(new_model_path), str(MODEL_PATH))
    print(f"  Updated: {new_model_path.name} -> {MODEL_PATH.name}")

    # Update registry
    new_entry = {
        "version": new_ver,
        "filename": "best_model.pth",
        "epoch": checkpoint.get("epoch", 0),
        "val_iou": round(checkpoint.get("val_iou", 0.0), 4),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "note": args.note or "",
    }
    registry["current_version"] = new_ver
    registry["models"].append(new_entry)
    save_registry(registry)

    print(f"\n[DONE] Model updated to v{new_ver}")
    print(f"  Epoch:   {new_entry['epoch']}")
    print(f"  Val IoU: {new_entry['val_iou']}")
    if args.note:
        print(f"  Note:    {args.note}")


# ============================================================
# Subcommand: info
# ============================================================
def cmd_info(args):
    """Display current model and system information."""
    print(f"SPI v{VERSION} - Solder Paste Inspector")
    print(f"{'='*50}")

    # Model info
    if MODEL_PATH.exists():
        device = get_device()
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        epoch = checkpoint.get("epoch", "?")
        val_iou = checkpoint.get("val_iou", 0.0)
        file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)

        print(f"  Model:      {MODEL_PATH.name} ({file_size_mb:.1f} MB)")
        print(f"  Epoch:      {epoch}")
        print(f"  Val IoU:    {val_iou:.4f}")
    else:
        print(f"  Model:      NOT FOUND ({MODEL_PATH})")

    # Registry info
    registry = load_registry()
    print(f"  Version:    v{registry['current_version']}")
    print(f"  History:    {len(registry['models'])} version(s)")

    # System info
    print(f"\n  Device:     {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  OpenCV:     {cv.__version__}")
    print(f"  Input size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Pixel size: {DEFAULT_PIXEL_SIZE_UM} um/px (default)")

    # ONNX status
    onnx_path = DEPLOY_DIR / "models" / "spi_model.onnx"
    if onnx_path.exists():
        onnx_size = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX:       {onnx_path.name} ({onnx_size:.1f} MB)")
    else:
        print(f"  ONNX:       not exported")

    # Version history
    if registry["models"]:
        print(f"\n{'='*50}")
        print(f"  Version History:")
        for m in registry["models"]:
            print(
                f"    v{m['version']}: epoch {m.get('epoch','?')}, "
                f"IoU={m.get('val_iou', 0):.4f}, "
                f"{m.get('date', '?')}"
                f"{' - ' + m['note'] if m.get('note') else ''}"
            )


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        prog="spi",
        description=f"SPI v{VERSION} - Solder Paste Inspector (Edge Deployment Tool)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single image:   python spi.py predict image.png
  Batch:          python spi.py predict images_folder/ --output results/
  Custom model:   python spi.py predict image.png --model /path/to/model.pth
  ONNX export:    python spi.py export --format onnx
  ONNX inference: python spi.py predict image.png --runtime onnx
  Update model:   python spi.py update-model new_model.pth --note "retrained"
  Model info:     python spi.py info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # predict
    p_predict = subparsers.add_parser("predict", help="Run inference + shape analysis")
    p_predict.add_argument("input", help="Input image file or directory")
    p_predict.add_argument("--output", default="spi_output", help="Output directory (default: spi_output/)")
    p_predict.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold (default: 0.5)")
    p_predict.add_argument("--pixel-size", type=float, default=DEFAULT_PIXEL_SIZE_UM, help="Pixel size in um (default: 15.0)")
    p_predict.add_argument("--no-viz", action="store_true", help="Skip analysis visualization")
    p_predict.add_argument("--device", default="auto", help="Device: cuda/cpu/auto (default: auto)")
    p_predict.add_argument("--runtime", choices=["pytorch", "onnx"], default="pytorch", help="Inference runtime (default: pytorch)")
    p_predict.add_argument("--model", default=None, help="Custom model path (default: models/best_model.pth)")
    p_predict.add_argument("--limit", type=int, default=0, help="Max images to process (0=all)")

    # export
    p_export = subparsers.add_parser("export", help="Export model to ONNX")
    p_export.add_argument("--format", default="onnx", help="Export format (default: onnx)")
    p_export.add_argument("--output", default=None, help="Output directory")
    p_export.add_argument("--device", default="auto", help="Device for export")
    p_export.add_argument("--model", default=None, help="Custom model path")

    # update-model
    p_update = subparsers.add_parser("update-model", help="Update model with new checkpoint")
    p_update.add_argument("model_path", help="Path to new model .pth file")
    p_update.add_argument("--note", default="", help="Description of this model version")

    # info
    subparsers.add_parser("info", help="Show model and system information")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "update-model":
        cmd_update_model(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
