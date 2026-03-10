"""
솔더 페이스트 U-Net 추론
========================
학습된 모델로 새 이미지에 대한 세그멘테이션 수행.

사용법:
    python dl/predict.py                          # images_main/ 전체 추론
    python dl/predict.py --input test/image.png   # 단일 이미지
    python dl/predict.py --input some_dir/        # 디렉토리 전체
"""

import os
import sys
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
sys.path.insert(0, str(PROJECT_ROOT))
from spi_core import apply_clahe

MODEL_PATH = PROJECT_ROOT / "dl" / "models" / "best_model.pth"
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """학습된 U-Net 모델 로드."""
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"모델 로드: {MODEL_PATH.name} (epoch {checkpoint['epoch']}, IoU={checkpoint['val_iou']:.4f})")
    return model


def get_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


@torch.no_grad()
def predict_single(model, img_path, transform, output_dir=None):
    """단일 이미지 추론."""
    img = cv.imread(str(img_path))
    if img is None:
        print(f"  [!] 이미지 로드 실패: {img_path}")
        return None

    h, w = img.shape[:2]
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_clahe = apply_clahe(img_rgb)

    augmented = transform(image=img_clahe, mask=np.zeros((h, w), dtype=np.float32))
    img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    # 원본 크기로 복원
    pred_resized = cv.resize(pred_mask, (w, h), interpolation=cv.INTER_NEAREST)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(img_path).stem

        # 마스크 저장
        cv.imwrite(str(Path(output_dir) / f"{stem}_mask.png"), pred_resized)

        # 오버레이 저장
        overlay = img.copy()
        overlay_color = np.zeros_like(img)
        overlay_color[pred_resized > 127] = [0, 0, 255]  # 빨간색
        overlay = cv.addWeighted(overlay, 0.7, overlay_color, 0.3, 0)
        cv.imwrite(str(Path(output_dir) / f"{stem}_overlay.png"), overlay)

    area_pct = np.count_nonzero(pred_resized) / pred_resized.size * 100
    return pred_resized, area_pct


def main():
    parser = argparse.ArgumentParser(description="솔더 페이스트 U-Net 추론")
    parser.add_argument("--input", type=str, default=str(PROJECT_ROOT / "images_main"),
                        help="입력 이미지 또는 디렉토리")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "dl" / "predictions"),
                        help="결과 저장 디렉토리")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="이진화 임계값 (기본: 0.5)")
    args = parser.parse_args()

    model = load_model()
    transform = get_transform()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        img_paths = [input_path]
    else:
        img_paths = sorted(input_path.glob("*.png"))
        # JSON 제외, 이미지만
        img_paths = [p for p in img_paths if p.suffix.lower() in ('.png', '.jpg', '.bmp')]

    print(f"\n추론 대상: {len(img_paths)}개 이미지")
    print(f"결과 저장: {output_dir}")

    for i, img_path in enumerate(img_paths):
        result = predict_single(model, img_path, transform, output_dir)
        if result:
            _, area_pct = result
            if (i + 1) % 100 == 0 or i < 5:
                print(f"  [{i+1}/{len(img_paths)}] {img_path.name} → 솔더 영역: {area_pct:.1f}%")

    print(f"\n추론 완료: {len(img_paths)}개 이미지 → {output_dir}")


if __name__ == "__main__":
    main()
