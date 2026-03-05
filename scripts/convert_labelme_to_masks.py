"""
LabelMe JSON → 바이너리 마스크 변환 스크립트
=============================================
LabelMe에서 생성한 폴리곤 어노테이션(JSON)을
바이너리 마스크 PNG(0=배경, 255=솔더)로 변환합니다.

사용법:
    python scripts/convert_labelme_to_masks.py              # test/ 폴더
    python scripts/convert_labelme_to_masks.py images_main   # images_main/ 폴더
    python scripts/convert_labelme_to_masks.py /path/to/dir  # 임의 경로
    python scripts/convert_labelme_to_masks.py images_TNMX --recursive  # TNMX 하위폴더
"""

import cv2 as cv
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트 기준 경로
PROJECT_ROOT = Path(__file__).parent.parent

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
        # shapes가 비어있으면 네거티브 샘플 (올-블랙 마스크)
        if len(data['shapes']) == 0:
            print(f"  [NEG] 빈 shapes → 네거티브 마스크: {Path(json_path).name}")
            return mask  # 올-블랙 (전부 0)
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


def _resolve_mask_dir(source_dir):
    """소스 디렉토리에 따라 마스크 출력 경로 결정."""
    dir_name = source_dir.name
    if dir_name == "images_gull":
        return PROJECT_ROOT / "annotations" / "gt_masks_gull"
    elif dir_name == "images_TNMX":
        return PROJECT_ROOT / "annotations" / "gt_masks_tnmx"
    else:
        return PROJECT_ROOT / "annotations" / "gt_masks"


def _convert_folder(source_dir, mask_dir, debug_dir):
    """단일 폴더의 JSON → 마스크 변환. 변환 개수 반환."""
    json_files = sorted(j for j in source_dir.glob("*.json")
                        if not j.name.startswith("."))

    if not json_files:
        return 0, 0

    converted = 0
    for json_file in json_files:
        name = json_file.stem
        img_path = source_dir / f"{name}.png"

        mask = labelme_json_to_mask(json_file)
        if mask is None:
            continue

        mask_path = mask_dir / f"{name}.png"
        cv.imwrite(str(mask_path), mask)

        pixel_count = np.count_nonzero(mask)
        if pixel_count == 0:
            print(f"    {name}: {mask.shape[1]}x{mask.shape[0]}, "
                  f"네거티브 (필렛 없음)")
        else:
            area_pct = pixel_count / mask.size * 100
            print(f"    {name}: {mask.shape[1]}x{mask.shape[0]}, "
                  f"{pixel_count}px ({area_pct:.1f}%)")

        if img_path.exists() and debug_dir:
            debug_path = debug_dir / f"{name}_overlay.png"
            create_debug_overlay(img_path, mask, debug_path)

        converted += 1

    return converted, len(json_files)


def batch_convert(source_dir=None, recursive=False):
    """지정 디렉토리의 모든 LabelMe JSON을 바이너리 마스크로 변환."""
    if source_dir is None:
        source_dir = PROJECT_ROOT / "test"
    else:
        source_dir = Path(source_dir)
        if not source_dir.is_absolute():
            source_dir = PROJECT_ROOT / source_dir

    base_mask_dir = _resolve_mask_dir(source_dir)

    print(f"LabelMe JSON -> 바이너리 마스크 변환")
    print(f"입력: {source_dir}")
    print(f"출력: {base_mask_dir}")
    if recursive:
        print(f"모드: --recursive (하위폴더 재귀)")
    print(f"{'=' * 50}")

    if recursive:
        # TNMX config 기반 하위폴더 처리
        config_path = source_dir / ".tnmx_folder_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            folders = [source_dir / e["folder"] for e in config["folders"]
                       if e["status"] == "include"]
            print(f"Config: {len(folders)}개 include 폴더")
        else:
            folders = sorted([d for d in source_dir.iterdir() if d.is_dir()])
            print(f"Config 없음: 모든 하위폴더 ({len(folders)}개)")

        total_converted = 0
        total_json = 0
        folders_with_labels = 0

        for folder in folders:
            # 하위폴더별 마스크 디렉토리 미러링
            subfolder_mask = base_mask_dir / folder.name
            subfolder_debug = subfolder_mask / "debug"

            # JSON이 있는 폴더만 처리
            json_count = len(list(folder.glob("*.json")))
            if json_count == 0:
                continue

            subfolder_mask.mkdir(parents=True, exist_ok=True)
            subfolder_debug.mkdir(parents=True, exist_ok=True)

            print(f"\n  📁 {folder.name} ({json_count}개 JSON)")
            converted, found = _convert_folder(folder, subfolder_mask, subfolder_debug)
            total_converted += converted
            total_json += found
            if converted > 0:
                folders_with_labels += 1

        print(f"\n{'=' * 50}")
        print(f"변환 완료: {total_converted}/{total_json}개 "
              f"({folders_with_labels}개 폴더)")

    else:
        # 기존 flat 모드
        mask_dir = base_mask_dir
        debug_dir = mask_dir / "debug"
        mask_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(j for j in source_dir.glob("*.json")
                            if not j.name.startswith("."))

        if not json_files:
            print(f"JSON 파일 없음: {source_dir}")
            print(f"\nLabelMe로 먼저 라벨링하세요:")
            print(f"  labelme {source_dir}")
            return

        converted, found = _convert_folder(source_dir, mask_dir, debug_dir)

        print(f"\n{'=' * 50}")
        print(f"변환 완료: {converted}/{found}개")
        print(f"\n다음 단계: python improved_analysis.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LabelMe JSON → 바이너리 마스크 변환")
    parser.add_argument("source_dir", nargs="?", default=None,
                        help="소스 디렉토리 (기본: test/)")
    parser.add_argument("--recursive", action="store_true",
                        help="하위폴더 재귀 처리 (TNMX config 기반)")
    args = parser.parse_args()

    batch_convert(args.source_dir, args.recursive)
