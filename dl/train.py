"""
솔더 페이스트 세그멘테이션 U-Net 학습
=====================================
- 모델: U-Net (ResNet18 encoder, ImageNet pretrained)
- 입력: 128x128 RGB (CLAHE 전처리 적용)
- 출력: 1채널 바이너리 마스크
- 손실: BCE + Dice Loss
- 증강: Flip, Rotate90, BrightnessContrast, GaussNoise
- 전처리: CLAHE (LAB L채널) - 학습/추론 동일 적용
"""

import os
import sys
import json
import random
import numpy as np
import cv2 as cv
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

# UTF-8 출력
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 설정
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT_ROOT / "images_main"
GULL_DIR = PROJECT_ROOT / "images_gull"
TNMX_DIR = PROJECT_ROOT / "images_TNMX"
MASK_DIR = PROJECT_ROOT / "annotations" / "gt_masks"
TNMX_MASK_DIR = PROJECT_ROOT / "annotations" / "gt_masks_tnmx"
OUTPUT_DIR = PROJECT_ROOT / "dl" / "models"
RESULT_BASE = PROJECT_ROOT / "dl" / "results"

IMG_SIZE = 128
BATCH_SIZE = 8
NUM_EPOCHS = 200
LR = 1e-3
VAL_RATIO = 0.2
SEED = 42

# VT-S730 AOI 해상도: ~15 µm/pixel (4MP, FOV 30mm)
PIXEL_SIZE_UM = 15.0
PIXEL_AREA_MM2 = (PIXEL_SIZE_UM / 1000.0) ** 2  # 0.000225 mm²

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CLAHE 전처리 (학습/추론 공통)
# ============================================================
def apply_clahe(img_rgb, clip_limit=3.0, grid_size=(8, 8)):
    """LAB 색공간 L채널에 CLAHE 적용. 학습·추론 동일하게 사용."""
    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv.cvtColor(lab, cv.COLOR_LAB2RGB)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 데이터셋
# ============================================================
class SolderPasteDataset(Dataset):
    """솔더 페이스트 세그멘테이션 데이터셋."""

    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 이미지 로드 (BGR → RGB)
        img = cv.imread(str(self.img_paths[idx]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # CLAHE 전처리 (고정 적용)
        img = apply_clahe(img)

        # 마스크 로드 (그레이스케일, 0/255 → 0/1)
        mask = cv.imread(str(self.mask_paths[idx]), cv.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
            mask = cv.resize(mask, (IMG_SIZE, IMG_SIZE))
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()

        # 마스크 차원 추가 (H, W) → (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return img, mask


def get_transforms(is_train=True):
    """학습/검증 증강 파이프라인."""
    if is_train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            # 기하학적 변환
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # 색상/밝기 변환 (조명 변화 대응)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # 노이즈 (센서 노이즈 대응)
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            # 정규화
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


# ============================================================
# 손실 함수
# ============================================================
class DiceBCELoss(nn.Module):
    """BCE + Dice 복합 손실."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice

        return bce_loss + dice_loss


# ============================================================
# 메트릭
# ============================================================
def compute_iou(pred, target, threshold=0.5):
    """IoU (Intersection over Union) 계산."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def compute_dice(pred, target, threshold=0.5):
    """Dice coefficient 계산."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    total = pred_bin.sum() + target.sum()
    if total == 0:
        return 1.0
    return (2.0 * intersection / total).item()


# ============================================================
# 데이터 준비
# ============================================================
def get_label_status(json_path):
    """JSON 파일의 라벨 상태 반환: manual, reviewed, auto, empty."""
    if not json_path.exists():
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    shapes = data.get('shapes', [])
    if not shapes:
        return 'empty'
    descs = [(s.get('description', '') or '') for s in shapes]
    if any('auto-generated' in d for d in descs):
        return 'auto'
    elif any('reviewed' in d for d in descs):
        return 'reviewed'
    else:
        return 'manual'


def find_modified_auto_labels(img_dir, buffer_sec=60.0):
    """타임스탬프로 수정된 auto-generated 라벨 감지.

    방법 1: .prelabel_meta.json에 저장된 배치 완료 시각 사용
    방법 2: (폴백) 타임스탬프 갭 분석으로 배치 클러스터 감지
    """
    auto_files = []
    for j in sorted(img_dir.glob("*.json")):
        if get_label_status(j) == 'auto':
            auto_files.append((j, os.path.getmtime(j)))

    if len(auto_files) < 10:
        return set()

    # 방법 1: 메타데이터 파일에서 배치 완료 시각 읽기
    meta_path = img_dir / ".prelabel_meta.json"
    gen_end = None
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        gen_end = meta.get("batch_end_time")

    # CLAHE 전처리가 적용된 경우, _orig/ 생성 시각을 기준으로 사용
    # (CLAHE 스크립트가 JSON mtime을 변경하므로 prelabel 시각 대신 사용)
    orig_dir = img_dir / "_orig"
    if orig_dir.exists() and gen_end is not None:
        clahe_time = orig_dir.stat().st_mtime
        if clahe_time > gen_end:
            gen_end = clahe_time

    # 방법 2: 갭 분석 폴백 (메타데이터 없을 때)
    if gen_end is None:
        times_sorted = sorted(t for _, t in auto_files)
        # 연속된 타임스탬프 간 갭 계산
        gaps = []
        for i in range(1, len(times_sorted)):
            gaps.append((times_sorted[i] - times_sorted[i - 1], i))

        # 가장 큰 갭이 60초 이상이면 배치/수정 경계로 판단
        if gaps:
            max_gap, gap_idx = max(gaps, key=lambda x: x[0])
            if max_gap > buffer_sec:
                # 갭 이전의 마지막 타임스탬프 = 배치 완료 시각
                gen_end = times_sorted[gap_idx - 1]
                from datetime import datetime
                dt = datetime.fromtimestamp(gen_end)
                print(f"  갭 분석 기준 시각: {dt.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"(갭: {max_gap:.0f}초)")

    if gen_end is None:
        return set()

    # 배치 완료 시각 이후에 수정된 파일 = 사용자가 수정한 파일
    modified = set()
    for j, t in auto_files:
        if t > gen_end + buffer_sec:
            modified.add(j.stem)

    return modified


def collect_labeled_data(img_dir, mask_dir):
    """특정 디렉토리에서 검증된 라벨(manual + reviewed + 수정된 auto) 데이터 수집."""
    img_paths = []
    mask_paths = []
    auto_paths = []

    # 타임스탬프로 수정된 auto 감지
    modified_stems = find_modified_auto_labels(img_dir)
    if modified_stems:
        print(f"  타임스탬프 감지 - 수정된 auto: {len(modified_stems)}개")

    for json_path in sorted(img_dir.glob("*.json")):
        img_path = img_dir / (json_path.stem + ".png")
        mask_path = mask_dir / (json_path.stem + ".png")
        if not img_path.exists() or not mask_path.exists():
            continue

        status = get_label_status(json_path)
        if status in ('manual', 'reviewed'):
            img_paths.append(img_path)
            mask_paths.append(mask_path)
        elif status == 'auto' and json_path.stem in modified_stems:
            # 타임스탬프가 다른 auto = 사용자가 수정함 → 학습에 포함
            img_paths.append(img_path)
            mask_paths.append(mask_path)
        elif status == 'auto':
            auto_paths.append(img_path)

    return img_paths, mask_paths, auto_paths


def collect_labeled_data_recursive(img_base_dir, mask_base_dir):
    """TNMX처럼 하위폴더 구조의 데이터를 재귀적으로 수집.

    config가 있으면 include 폴더만, 없으면 모든 하위폴더를 처리.
    """
    config_path = img_base_dir / ".tnmx_folder_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        folders = [img_base_dir / e["folder"] for e in config["folders"]
                   if e["status"] == "include"]
    else:
        folders = sorted([d for d in img_base_dir.iterdir() if d.is_dir()])

    all_imgs, all_masks, all_auto = [], [], []
    for folder in folders:
        sub_mask_dir = mask_base_dir / folder.name
        if not sub_mask_dir.exists():
            continue
        imgs, masks, auto = collect_labeled_data(folder, sub_mask_dir)
        all_imgs.extend(imgs)
        all_masks.extend(masks)
        all_auto.extend(auto)

    return all_imgs, all_masks, all_auto


TEST_SET_FILE = PROJECT_ROOT / "dl" / "test_set.json"


def load_or_create_test_set(main_auto, gull_auto, labeled_stems,
                            tnmx_auto=None):
    """고정 테스트셋 로드. 없으면 생성. 라벨링된 이미지는 자동 교체."""
    test_data = None
    if TEST_SET_FILE.exists():
        with open(TEST_SET_FILE, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

    def resolve_list(saved_stems, all_auto_paths, count=50):
        """저장된 테스트셋에서 라벨링된 이미지를 교체."""
        auto_stem_set = {p.stem for p in all_auto_paths}
        auto_by_stem = {p.stem: p for p in all_auto_paths}

        result = []
        replaced = 0
        if saved_stems:
            for stem in saved_stems:
                if stem not in labeled_stems and stem in auto_stem_set:
                    result.append(auto_by_stem[stem])
                else:
                    replaced += 1
        # 부족분 채우기 (교체 또는 초기 생성)
        need = count - len(result)
        if need > 0:
            used = {p.stem for p in result}
            candidates = [p for p in all_auto_paths
                          if p.stem not in used and p.stem not in labeled_stems]
            random.shuffle(candidates)
            result.extend(candidates[:need])
            replaced += min(need, len(candidates))

        if replaced > 0 and saved_stems:
            print(f"    교체된 테스트 이미지: {replaced}개 (라벨링됨 → 대체)")
        return result

    saved_main = test_data.get("main", []) if test_data else []
    saved_gull = test_data.get("gull", []) if test_data else []
    saved_tnmx = test_data.get("tnmx", []) if test_data else []

    test_main = resolve_list(saved_main, main_auto, 100)
    test_gull = resolve_list(saved_gull, gull_auto, 100)
    test_tnmx = resolve_list(saved_tnmx, tnmx_auto or [], 100)

    # 저장
    new_data = {
        "main": [p.stem for p in test_main],
        "gull": [p.stem for p in test_gull],
        "tnmx": [p.stem for p in test_tnmx],
    }
    os.makedirs(TEST_SET_FILE.parent, exist_ok=True)
    with open(TEST_SET_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)

    if not test_data:
        print(f"  고정 테스트셋 생성: {TEST_SET_FILE}")
    else:
        print(f"  고정 테스트셋 로드: {TEST_SET_FILE}")

    return test_main, test_gull, test_tnmx


def prepare_data():
    """images_main + images_gull + images_TNMX 에서 검증된 라벨로 train/val 분할."""
    # 각 디렉토리별 마스크 경로
    main_mask_dir = PROJECT_ROOT / "annotations" / "gt_masks"
    gull_mask_dir = PROJECT_ROOT / "annotations" / "gt_masks_gull"

    main_imgs, main_masks, main_auto = collect_labeled_data(IMG_DIR, main_mask_dir)
    gull_imgs, gull_masks, gull_auto = collect_labeled_data(GULL_DIR, gull_mask_dir)

    # TNMX 재귀 수집 (폴더 존재 시)
    tnmx_imgs, tnmx_masks, tnmx_auto = [], [], []
    if TNMX_DIR.exists() and TNMX_MASK_DIR.exists():
        tnmx_imgs, tnmx_masks, tnmx_auto = collect_labeled_data_recursive(
            TNMX_DIR, TNMX_MASK_DIR)

    img_paths = main_imgs + gull_imgs + tnmx_imgs
    mask_paths = main_masks + gull_masks + tnmx_masks

    print(f"학습 가능 데이터: {len(img_paths)}개 "
          f"(main:{len(main_imgs)} + gull:{len(gull_imgs)} + tnmx:{len(tnmx_imgs)})")

    # 라벨링된 이미지 stem 집합 (테스트셋에서 제외용)
    labeled_stems = {p.stem for p in img_paths}

    # 고정 테스트셋
    test_main, test_gull, test_tnmx = load_or_create_test_set(
        main_auto, gull_auto, labeled_stems, tnmx_auto)

    # 셔플 후 분할
    indices = list(range(len(img_paths)))
    random.shuffle(indices)

    val_size = max(1, int(len(indices) * VAL_RATIO))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_imgs = [img_paths[i] for i in train_indices]
    train_masks = [mask_paths[i] for i in train_indices]
    val_imgs = [img_paths[i] for i in val_indices]
    val_masks = [mask_paths[i] for i in val_indices]

    print(f"Train: {len(train_imgs)}개, Val: {len(val_imgs)}개")
    print(f"Test main: {len(test_main)}개, Test gull: {len(test_gull)}개, "
          f"Test tnmx: {len(test_tnmx)}개")

    return train_imgs, train_masks, val_imgs, val_masks, test_main, test_gull


# ============================================================
# 학습
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    total_iou = 0
    total_dice = 0

    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(preds, masks)
        total_dice += compute_dice(preds, masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0

    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, masks)

        total_loss += loss.item()
        total_iou += compute_iou(preds, masks)
        total_dice += compute_dice(preds, masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


@torch.no_grad()
def save_predictions(model, val_imgs, val_masks, result_dir, transform):
    """검증 이미지의 예측 결과를 시각화하여 저장."""
    model.eval()
    os.makedirs(result_dir, exist_ok=True)
    areas = []

    for img_path, mask_path in zip(val_imgs, val_masks):
        # 원본 이미지
        img_orig = cv.imread(str(img_path))
        img_rgb = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)

        # GT 마스크
        gt_mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        # CLAHE 적용 후 예측
        img_clahe = apply_clahe(img_rgb)
        augmented = transform(image=img_clahe, mask=np.zeros_like(gt_mask, dtype=np.float32))
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
        pred_mask = (pred > 0.5).astype(np.uint8) * 255

        # 원본 크기로 복원
        h, w = img_orig.shape[:2]
        pred_resized = cv.resize(pred_mask, (w, h))
        gt_resized = cv.resize(gt_mask, (w, h))

        # 오버레이 생성
        overlay = img_orig.copy()
        # GT: 녹색
        gt_colored = np.zeros_like(img_orig)
        gt_colored[gt_resized > 127] = [0, 255, 0]
        # Pred: 빨간색
        pred_colored = np.zeros_like(img_orig)
        pred_colored[pred_resized > 127] = [0, 0, 255]

        overlay = cv.addWeighted(overlay, 0.6, gt_colored, 0.2, 0)
        overlay = cv.addWeighted(overlay, 1.0, pred_colored, 0.2, 0)

        # 비교 이미지 생성 (원본 | GT | Pred | 오버레이)
        scale = max(1, 200 // max(h, w))
        img_big = cv.resize(img_orig, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)

        gt_big = cv.resize(gt_resized, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)
        gt_big_3ch = cv.cvtColor(gt_big, cv.COLOR_GRAY2BGR)

        pred_big = cv.resize(pred_resized, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)
        pred_big_3ch = cv.cvtColor(pred_big, cv.COLOR_GRAY2BGR)

        overlay_big = cv.resize(overlay, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)

        comparison = np.hstack([img_big, gt_big_3ch, pred_big_3ch, overlay_big])

        # 면적 계산
        solder_px = np.count_nonzero(pred_resized > 127)
        gt_px = np.count_nonzero(gt_resized > 127)
        area_mm2 = solder_px * PIXEL_AREA_MM2
        gt_area_mm2 = gt_px * PIXEL_AREA_MM2

        text = f"GT:{gt_area_mm2:.4f} Pred:{area_mm2:.4f} mm2"
        cv.putText(comparison, text, (5, comparison.shape[0] - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        stem = img_path.stem
        cv.imwrite(str(result_dir / f"{stem}_comparison.png"), comparison)
        areas.append({"stem": stem, "gt_px": int(gt_px), "pred_px": int(solder_px),
                      "gt_area_mm2": float(gt_area_mm2), "pred_area_mm2": float(area_mm2)})

    # 면적 통계 저장
    if areas:
        with open(result_dir / "area_stats.json", "w", encoding="utf-8") as f:
            json.dump(areas, f, indent=2, ensure_ascii=False)
        avg_gt = np.mean([a["gt_area_mm2"] for a in areas])
        avg_pred = np.mean([a["pred_area_mm2"] for a in areas])
        print(f"  예측 시각화 저장: {result_dir}")
        print(f"  면적 비교 - GT 평균: {avg_gt:.4f} mm², Pred 평균: {avg_pred:.4f} mm²")
    else:
        print(f"  예측 시각화 저장: {result_dir}")


@torch.no_grad()
def save_test_predictions(model, test_imgs, test_dir, transform):
    """테스트 이미지(GT 없음)의 예측 결과를 시각화하여 저장."""
    model.eval()
    os.makedirs(test_dir, exist_ok=True)
    areas = []

    for img_path in test_imgs:
        img_orig = cv.imread(str(img_path))
        img_rgb = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)
        h, w = img_orig.shape[:2]

        # CLAHE 적용 후 예측
        img_clahe = apply_clahe(img_rgb)
        augmented = transform(image=img_clahe, mask=np.zeros((h, w), dtype=np.float32))
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
        pred_mask = (pred > 0.5).astype(np.uint8) * 255
        pred_resized = cv.resize(pred_mask, (w, h))

        # 오버레이 (예측: 녹색)
        overlay = img_orig.copy()
        pred_colored = np.zeros_like(img_orig)
        pred_colored[pred_resized > 127] = [0, 255, 0]
        overlay = cv.addWeighted(overlay, 0.7, pred_colored, 0.3, 0)

        # 면적 계산
        solder_px = np.count_nonzero(pred_resized)
        area_mm2 = solder_px * PIXEL_AREA_MM2

        # 비교 이미지 (원본 | 예측 마스크 | 오버레이)
        scale = max(1, 200 // max(h, w))
        img_big = cv.resize(img_orig, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)
        pred_big = cv.resize(pred_resized, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)
        pred_big_3ch = cv.cvtColor(pred_big, cv.COLOR_GRAY2BGR)
        overlay_big = cv.resize(overlay, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)

        comparison = np.hstack([img_big, pred_big_3ch, overlay_big])

        # 면적 정보 텍스트 추가
        text = f"{solder_px}px / {area_mm2:.4f}mm2"
        cv.putText(comparison, text, (5, comparison.shape[0] - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv.imwrite(str(test_dir / f"{img_path.stem}_test.png"), comparison)
        areas.append({"stem": img_path.stem, "pixels": int(solder_px), "area_mm2": float(area_mm2)})

    # 면적 통계 저장
    if areas:
        with open(test_dir / "area_stats.json", "w", encoding="utf-8") as f:
            json.dump(areas, f, indent=2, ensure_ascii=False)
        avg_area = np.mean([a["area_mm2"] for a in areas])
        print(f"  테스트 예측 저장: {test_dir} ({len(test_imgs)}개)")
        print(f"  평균 솔더 면적: {avg_area:.4f} mm²")
    else:
        print(f"  테스트 예측 저장: {test_dir} ({len(test_imgs)}개)")


# ============================================================
# 메인
# ============================================================
def make_result_dir():
    """매 학습마다 새로운 결과 폴더 생성 (run_001, run_002, ...)."""
    os.makedirs(RESULT_BASE, exist_ok=True)
    existing = sorted(RESULT_BASE.glob("run_*"))
    if existing:
        last_num = max(int(d.name.split("_")[1]) for d in existing if d.is_dir())
        next_num = last_num + 1
    else:
        next_num = 1
    result_dir = RESULT_BASE / f"run_{next_num:03d}"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def main():
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    RESULT_DIR = make_result_dir()

    print("=" * 60)
    print("솔더 페이스트 U-Net 세그멘테이션 학습")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Results: {RESULT_DIR}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LR}")

    # 데이터 준비
    train_imgs, train_masks, val_imgs, val_masks, test_main, test_gull = prepare_data()

    train_dataset = SolderPasteDataset(train_imgs, train_masks, get_transforms(is_train=True))
    val_dataset = SolderPasteDataset(val_imgs, val_masks, get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 모델
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {param_count:,}")

    # 손실, 옵티마이저, 스케줄러
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 학습 루프
    best_val_iou = 0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [],
               "train_dice": [], "val_dice": [], "lr": []}

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Train IoU':>10} | {'Val IoU':>10} | {'Val Dice':>10} | {'LR':>10}")
    print("-" * 85)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_iou, train_dice = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_iou, val_dice = validate(model, val_loader, criterion)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["lr"].append(lr)

        marker = ""
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_dice": val_dice,
            }, OUTPUT_DIR / "best_model.pth")
            marker = " *"

        if epoch % 10 == 0 or epoch <= 5 or marker:
            print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
                  f"{train_iou:>10.4f} | {val_iou:>10.4f} | {val_dice:>10.4f} | "
                  f"{lr:>10.6f}{marker}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("학습 완료")
    print("=" * 60)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Best Val Dice: {history['val_dice'][best_epoch - 1]:.4f}")
    print(f"OpenCV 최고 IoU: 0.8165 (Adaptive_Lab_L_bs11_c10)")
    print(f"U-Net vs OpenCV: {'+' if best_val_iou > 0.8165 else ''}{(best_val_iou - 0.8165) * 100:.2f}%p")

    # 베스트 모델로 예측 시각화
    checkpoint = torch.load(OUTPUT_DIR / "best_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_predictions(model, val_imgs, val_masks, RESULT_DIR / "val", get_transforms(is_train=False))

    # 테스트셋 예측 (GT 없이 시각화만) - 각 50개씩 하위폴더에
    if test_main:
        save_test_predictions(model, test_main[:50], RESULT_DIR / "test_main", get_transforms(is_train=False))
    if test_gull:
        save_test_predictions(model, test_gull[:50], RESULT_DIR / "test_gull", get_transforms(is_train=False))

    # 학습 기록 저장
    with open(RESULT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "img_size": IMG_SIZE,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "lr": LR,
                "encoder": "resnet18",
                "train_count": len(train_imgs),
                "val_count": len(val_imgs),
                "device": str(DEVICE),
            },
            "best_epoch": best_epoch,
            "best_val_iou": best_val_iou,
            "best_val_dice": history["val_dice"][best_epoch - 1],
            "opencv_best_iou": 0.8165,
            "history": history,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n모델 저장: {OUTPUT_DIR / 'best_model.pth'}")
    print(f"학습 기록: {RESULT_DIR / 'training_history.json'}")
    print(f"예측 시각화: {RESULT_DIR}")


if __name__ == "__main__":
    main()
