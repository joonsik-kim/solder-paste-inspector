"""
CLAHE 라벨링 준비 스크립트 (v2 - 원본 교체 방식)
=================================================
원본 이미지를 _orig/ 서브폴더로 백업하고,
CLAHE 전처리된 이미지를 원본 파일명으로 저장.

- LabelMe에서 CLAHE 이미지만 보임 (원본 숨김)
- JSON이 없는 이미지도 CLAHE 적용 → 수동 라벨링 가능
- 학습 시 CLAHE 재적용은 near-idempotent → 영향 무시 가능
- --cleanup: 원본 복원 + _orig/ 삭제

사용법:
    python scripts/prepare_clahe_labeling.py images_TNMX/811100-28410730-A2
    python scripts/prepare_clahe_labeling.py images_TNMX --recursive
    python scripts/prepare_clahe_labeling.py images_TNMX --recursive --cleanup
"""

import os
import sys
import json
import shutil
import argparse
import cv2 as cv
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIG_BACKUP_DIR = "_orig"
OLD_CLAHE_SUFFIX = "_clahe"


def apply_clahe(img_bgr, clip_limit=3.0, grid_size=(8, 8)):
    """LAB L채널에 CLAHE 적용 (학습/추론과 동일)."""
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def _cleanup_old_clahe(folder_path):
    """이전 방식(_clahe.png 파일 + JSON imagePath) 정리."""
    old_files = sorted(folder_path.glob(f"*{OLD_CLAHE_SUFFIX}.png"))
    if not old_files:
        return 0

    cleaned = 0
    for clahe_img in old_files:
        stem = clahe_img.stem.replace(OLD_CLAHE_SUFFIX, "")
        json_path = folder_path / f"{stem}.json"

        # JSON imagePath 원복
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if OLD_CLAHE_SUFFIX in data.get("imagePath", ""):
                data["imagePath"] = f"{stem}.png"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

        clahe_img.unlink()
        cleaned += 1

    return cleaned


def process_folder(folder_path):
    """원본을 _orig/로 백업, CLAHE를 원본 파일명으로 저장."""
    orig_dir = folder_path / ORIG_BACKUP_DIR

    # 이미 처리됨 (원본 백업 존재)
    if orig_dir.exists() and any(orig_dir.glob("*.png")):
        print(f"    이미 처리됨 (스킵): {folder_path.name}")
        return 0

    # 이전 방식 _clahe.png 파일 정리
    old_cleaned = _cleanup_old_clahe(folder_path)
    if old_cleaned > 0:
        print(f"    이전 _clahe 파일 정리: {old_cleaned}개")

    # 모든 PNG 수집 (JSON 유무와 무관)
    img_files = sorted([
        p for p in folder_path.glob("*.png")
        if not p.name.startswith(".")
        and OLD_CLAHE_SUFFIX not in p.stem
    ])

    if not img_files:
        return 0

    # _orig/ 디렉토리 생성
    orig_dir.mkdir(exist_ok=True)

    processed = 0
    for img_path in img_files:
        img = cv.imread(str(img_path))
        if img is None:
            continue

        # 원본을 _orig/로 이동
        backup_path = orig_dir / img_path.name
        shutil.move(str(img_path), str(backup_path))

        # CLAHE 적용 후 원본 파일명으로 저장
        clahe_result = apply_clahe(img)
        cv.imwrite(str(img_path), clahe_result)

        processed += 1

    return processed


def cleanup_folder(folder_path):
    """원본 복원: _orig/에서 원본 되돌리기."""
    orig_dir = folder_path / ORIG_BACKUP_DIR

    # 이전 방식 _clahe.png 정리도 같이 수행
    old_cleaned = _cleanup_old_clahe(folder_path)

    if not orig_dir.exists():
        return old_cleaned

    backup_files = sorted(orig_dir.glob("*.png"))
    restored = 0
    for backup_img in backup_files:
        target = folder_path / backup_img.name
        # CLAHE 버전 덮어쓰기 → 원본 복원
        shutil.move(str(backup_img), str(target))
        restored += 1

    # _orig/ 디렉토리 삭제
    try:
        orig_dir.rmdir()
    except OSError:
        pass

    return restored + old_cleaned


def main():
    parser = argparse.ArgumentParser(
        description="CLAHE 라벨링 준비 (원본 교체 방식)")
    parser.add_argument("source_dir",
                        help="이미지 디렉토리")
    parser.add_argument("--recursive", action="store_true",
                        help="하위폴더 재귀 처리 (TNMX config 기반)")
    parser.add_argument("--cleanup", action="store_true",
                        help="원본 복원 + _orig/ 삭제")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir

    action = "원본 복원" if args.cleanup else "CLAHE 교체"
    print(f"{'=' * 50}")
    print(f"CLAHE 라벨링 준비 — {action}")
    print(f"{'=' * 50}")

    if args.recursive:
        config_path = source_dir / ".tnmx_folder_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            folders = [source_dir / e["folder"] for e in config["folders"]
                       if e["status"] == "include"]
        else:
            folders = sorted([d for d in source_dir.iterdir()
                             if d.is_dir() and d.name != ORIG_BACKUP_DIR])

        total = 0
        processed_folders = 0
        for i, folder in enumerate(folders):
            if args.cleanup:
                count = cleanup_folder(folder)
            else:
                count = process_folder(folder)

            if count > 0:
                processed_folders += 1
                total += count

            if (i + 1) % 20 == 0:
                print(f"  진행: {i+1}/{len(folders)} 폴더 ({total}개 이미지)")

        print(f"\n{action} 완료: {total}개 이미지 ({processed_folders}개 폴더)")
    else:
        if args.cleanup:
            count = cleanup_folder(source_dir)
        else:
            count = process_folder(source_dir)
        print(f"\n{action} 완료: {count}개 이미지")

    if not args.cleanup:
        print(f"\n이제 LabelMe에서 CLAHE 이미지로 라벨링하세요:")
        if args.recursive:
            print(f"  labelme {source_dir}/[폴더명] --autosave")
        else:
            print(f"  labelme {source_dir} --autosave")
        print(f"\n라벨링 완료 후 원본 복원:")
        cleanup_cmd = f"python scripts/prepare_clahe_labeling.py {args.source_dir}"
        if args.recursive:
            cleanup_cmd += " --recursive"
        print(f"  {cleanup_cmd} --cleanup")
        print(f"\n참고: 원본은 각 폴더의 _orig/ 에 보관됩니다.")


if __name__ == "__main__":
    main()
