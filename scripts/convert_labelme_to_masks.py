"""
LabelMe JSON → 바이너리 마스크 변환 스크립트
=============================================
LabelMe에서 생성한 폴리곤 어노테이션(JSON)을
바이너리 마스크 PNG(0=배경, 255=솔더)로 변환합니다.

사용법:
    python scripts/convert_labelme_to_masks.py

입력: test/*.json (LabelMe 어노테이션)
출력: annotations/gt_masks/*.png (바이너리 마스크)
"""

import cv2 as cv
import numpy as np
import json
import os
import sys
from pathlib import Path

# 프로젝트 루트 기준 경로
PROJECT_ROOT = Path(__file__).parent.parent
JSON_DIR = PROJECT_ROOT / "test"
MASK_DIR = PROJECT_ROOT / "annotations" / "gt_masks"
DEBUG_DIR = MASK_DIR / "debug"

# Windows cp949 인코딩 문제 방지
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def labelme_json_to_mask(json_path):
    """
    LabelMe JSON 파일에서 바이너리 마스크 생성.

    Args:
        json_path: LabelMe JSON 파일 경로

    Returns:
        mask: uint8 바이너리 마스크 (0=배경, 255=솔더)
        None: JSON 파싱 실패 시
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    h = data['imageHeight']
    w = data['imageWidth']
    mask = np.zeros((h, w), dtype=np.uint8)

    shape_count = 0
    for shape in data['shapes']:
        if shape['label'] == 'solder_paste' and shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            cv.fillPoly(mask, [points], 255)
            shape_count += 1

    if shape_count == 0:
        print(f"  [!] 'solder_paste' 폴리곤 없음: {json_path}")
        return None

    return mask


def create_debug_overlay(img_path, mask, output_path):
    """
    원본 이미지 위에 마스크를 오버레이한 디버그 이미지 생성.
    초록색 = 마스크 영역, 빨간 윤곽선 = 마스크 경계
    """
    img = cv.imread(str(img_path))
    if img is None:
        return

    overlay = img.copy()

    # 초록색 반투명 오버레이
    colored = np.zeros_like(img)
    colored[mask > 0] = (0, 255, 0)
    overlay = cv.addWeighted(overlay, 0.6, colored, 0.4, 0)

    # 빨간 윤곽선
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(overlay, contours, -1, (0, 0, 255), 1)

    # 면적 정보
    area_pct = np.count_nonzero(mask) / mask.size * 100
    cv.putText(overlay, f"Area: {area_pct:.1f}%", (2, 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    cv.imwrite(str(output_path), overlay)


def batch_convert():
    """test/ 디렉토리의 모든 LabelMe JSON을 바이너리 마스크로 변환."""
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(JSON_DIR.glob("*.json"))

    if not json_files:
        print(f"JSON 파일 없음: {JSON_DIR}")
        print(f"\nLabelMe로 먼저 라벨링하세요:")
        print(f"  pip install labelme")
        print(f"  labelme test/")
        return

    print(f"LabelMe JSON -> 바이너리 마스크 변환")
    print(f"입력: {JSON_DIR}")
    print(f"출력: {MASK_DIR}")
    print(f"{'=' * 50}")

    converted = 0
    for json_file in json_files:
        name = json_file.stem
        img_path = JSON_DIR / f"{name}.png"

        print(f"\n  {name}:")

        # JSON → 마스크 변환
        mask = labelme_json_to_mask(json_file)
        if mask is None:
            continue

        # 마스크 저장
        mask_path = MASK_DIR / f"{name}.png"
        cv.imwrite(str(mask_path), mask)

        area_pct = np.count_nonzero(mask) / mask.size * 100
        pixel_count = np.count_nonzero(mask)
        print(f"    크기: {mask.shape[1]}x{mask.shape[0]}")
        print(f"    솔더 영역: {pixel_count}px ({area_pct:.1f}%)")
        print(f"    저장: {mask_path}")

        # 디버그 오버레이 생성
        if img_path.exists():
            debug_path = DEBUG_DIR / f"{name}_overlay.png"
            create_debug_overlay(img_path, mask, debug_path)
            print(f"    디버그: {debug_path}")

        converted += 1

    print(f"\n{'=' * 50}")
    print(f"변환 완료: {converted}/{len(json_files)}개")
    print(f"\n다음 단계: python improved_analysis.py")


if __name__ == "__main__":
    batch_convert()
