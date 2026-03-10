"""
TNMX 폴더 서베이 및 필터링 도구
================================
images_TNMX 하위 147개 폴더를 자동 분류하고,
수동 리뷰를 위한 그리드 뷰어를 제공.

사용법:
    python scripts/review_tnmx_folders.py                  # 서베이 + config 생성
    python scripts/review_tnmx_folders.py --review          # 'review' 폴더 수동 확인
    python scripts/review_tnmx_folders.py --stats           # 현재 config 통계만 출력

자동 분류 기준:
    - include: 모든 이미지가 소형~중형 ROI (높이맵)
    - exclude: 대형 원본 사진 비율 > 30%
    - review: 혼합 또는 판단 어려운 폴더
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TNMX_DIR = PROJECT_ROOT / "images_TNMX"
CONFIG_PATH = TNMX_DIR / ".tnmx_folder_config.json"

# 분류 기준 (픽셀)
MAX_ROI_AREA = 500 * 500       # 이보다 크면 원본 사진 의심
MIN_VALID_DIM = 5              # 최소 유효 크기 (너무 작으면 무효)


def survey_folder(folder_path, sample_n=5):
    """폴더 내 이미지 크기를 샘플링하여 분류 정보 반환."""
    imgs = sorted(folder_path.glob("*.png"))
    if not imgs:
        return None

    # 샘플링 (최대 sample_n개)
    if len(imgs) <= sample_n:
        samples = imgs
    else:
        step = len(imgs) // sample_n
        samples = [imgs[i * step] for i in range(sample_n)]

    sizes = []
    for img_path in samples:
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                sizes.append((w, h))
        except Exception:
            continue

    if not sizes:
        return None

    areas = [w * h for w, h in sizes]
    widths = [w for w, h in sizes]
    heights = [h for w, h in sizes]

    large_count = sum(1 for a in areas if a > MAX_ROI_AREA)
    tiny_count = sum(1 for w, h in sizes if w < MIN_VALID_DIM or h < MIN_VALID_DIM)
    large_ratio = large_count / len(sizes)

    # 자동 분류
    if large_ratio > 0.3:
        status = "exclude"
        reason = f"대형 이미지 {large_ratio:.0%} (원본 사진 의심)"
    elif tiny_count > len(sizes) * 0.5:
        status = "exclude"
        reason = f"무효 크기 이미지 {tiny_count}/{len(sizes)}"
    elif large_ratio > 0:
        status = "review"
        reason = f"대형 이미지 {large_count}/{len(sizes)} 혼합"
    else:
        status = "include"
        reason = "정상 ROI"

    return {
        "folder": folder_path.name,
        "image_count": len(imgs),
        "sampled": len(sizes),
        "avg_width": int(np.mean(widths)),
        "avg_height": int(np.mean(heights)),
        "min_size": f"{min(widths)}x{min(heights)}",
        "max_size": f"{max(widths)}x{max(heights)}",
        "large_ratio": round(large_ratio, 2),
        "status": status,
        "reason": reason,
    }


def run_survey():
    """전체 TNMX 하위 폴더 서베이 + config 생성."""
    if not TNMX_DIR.exists():
        print(f"디렉토리 없음: {TNMX_DIR}")
        return

    folders = sorted([d for d in TNMX_DIR.iterdir() if d.is_dir()])
    print(f"{'=' * 60}")
    print(f"TNMX 폴더 서베이 ({len(folders)}개 폴더)")
    print(f"{'=' * 60}")

    results = []
    include_count = 0
    exclude_count = 0
    review_count = 0
    total_images = 0

    for i, folder in enumerate(folders):
        info = survey_folder(folder)
        if info is None:
            print(f"  [{i+1}/{len(folders)}] {folder.name}: 이미지 없음 → SKIP")
            continue

        results.append(info)
        total_images += info["image_count"]

        if info["status"] == "include":
            include_count += 1
        elif info["status"] == "exclude":
            exclude_count += 1
        else:
            review_count += 1

        marker = {"include": "✓", "exclude": "✗", "review": "?"}[info["status"]]
        if (i + 1) % 10 == 0 or info["status"] != "include":
            print(f"  [{i+1}/{len(folders)}] {marker} {folder.name}: "
                  f"{info['image_count']}장, {info['avg_width']}x{info['avg_height']}avg, "
                  f"{info['reason']}")

    # 기존 config 로드 (수동 변경 보존)
    existing = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            old_config = json.load(f)
        for entry in old_config.get("folders", []):
            if entry.get("manual_override"):
                existing[entry["folder"]] = entry["status"]

    # 수동 오버라이드 적용
    for r in results:
        if r["folder"] in existing:
            r["status"] = existing[r["folder"]]
            r["reason"] += " (수동 설정 유지)"
            r["manual_override"] = True

    # config 저장
    config = {
        "created": str(Path(__file__).name),
        "total_folders": len(results),
        "total_images": total_images,
        "summary": {
            "include": include_count,
            "exclude": exclude_count,
            "review": review_count,
        },
        "folders": results,
    }

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"결과 요약:")
    print(f"  include (학습 대상): {include_count}개")
    print(f"  exclude (제외):      {exclude_count}개")
    print(f"  review  (수동 확인): {review_count}개")
    print(f"  총 이미지:           {total_images}장")
    print(f"\nConfig 저장: {CONFIG_PATH}")
    if review_count > 0:
        print(f"\n다음 단계: python scripts/review_tnmx_folders.py --review")


def review_folders():
    """'review' 상태 폴더를 OpenCV 그리드 뷰어로 수동 확인."""
    try:
        import cv2 as cv
    except ImportError:
        print("OpenCV 필요: pip install opencv-python")
        return

    if not CONFIG_PATH.exists():
        print("먼저 서베이를 실행하세요: python scripts/review_tnmx_folders.py")
        return

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    review_list = [e for e in config["folders"] if e["status"] == "review"]
    if not review_list:
        print("리뷰할 폴더가 없습니다.")
        return

    print(f"리뷰 대상: {len(review_list)}개 폴더")
    print("조작: [i] include / [e] exclude / [n] 다음 / [q] 종료\n")

    changed = False
    for idx, entry in enumerate(review_list):
        folder_path = TNMX_DIR / entry["folder"]
        imgs = sorted(folder_path.glob("*.png"))
        if not imgs:
            continue

        # 최대 9장 샘플 (3x3 그리드)
        sample = imgs[:9] if len(imgs) <= 9 else [imgs[i * len(imgs) // 9] for i in range(9)]

        # 그리드 생성
        cell_size = 200
        grid_cols = min(3, len(sample))
        grid_rows = (len(sample) + grid_cols - 1) // grid_cols
        grid = np.zeros((grid_rows * cell_size, grid_cols * cell_size, 3), dtype=np.uint8)

        for i, img_path in enumerate(sample):
            img = cv.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = min(cell_size / w, cell_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv.resize(img, (new_w, new_h))

            row, col = divmod(i, grid_cols)
            y_off = row * cell_size + (cell_size - new_h) // 2
            x_off = col * cell_size + (cell_size - new_w) // 2
            grid[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # 폴더 정보 텍스트
        info_text = (f"{entry['folder']} ({entry['image_count']}imgs, "
                     f"{entry['avg_width']}x{entry['avg_height']}avg)")
        cv.putText(grid, info_text, (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv.putText(grid, f"[{idx+1}/{len(review_list)}] i=include e=exclude n=skip q=quit",
                   (5, grid.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv.imshow("TNMX Folder Review", grid)
        while True:
            key = cv.waitKey(0) & 0xFF
            if key == ord('i'):
                entry["status"] = "include"
                entry["manual_override"] = True
                entry["reason"] = "수동 확인 → include"
                changed = True
                print(f"  ✓ {entry['folder']} → include")
                break
            elif key == ord('e'):
                entry["status"] = "exclude"
                entry["manual_override"] = True
                entry["reason"] = "수동 확인 → exclude"
                changed = True
                print(f"  ✗ {entry['folder']} → exclude")
                break
            elif key == ord('n'):
                print(f"  - {entry['folder']} → skip (review 유지)")
                break
            elif key == ord('q'):
                cv.destroyAllWindows()
                if changed:
                    _save_config(config)
                return

    cv.destroyAllWindows()
    if changed:
        _save_config(config)


def _save_config(config):
    """config 갱신 저장."""
    # summary 재계산
    folders = config["folders"]
    config["summary"] = {
        "include": sum(1 for f in folders if f["status"] == "include"),
        "exclude": sum(1 for f in folders if f["status"] == "exclude"),
        "review": sum(1 for f in folders if f["status"] == "review"),
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nConfig 갱신 저장: {CONFIG_PATH}")


def print_stats():
    """현재 config 통계 출력."""
    if not CONFIG_PATH.exists():
        print("Config 없음. 먼저 서베이 실행: python scripts/review_tnmx_folders.py")
        return

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    summary = config["summary"]
    print(f"TNMX 폴더 Config 현황:")
    print(f"  include: {summary['include']}개 폴더")
    print(f"  exclude: {summary['exclude']}개 폴더")
    print(f"  review:  {summary['review']}개 폴더")
    print(f"  총:      {config['total_folders']}개 폴더, {config['total_images']}장")

    # include 폴더 이미지 수 합계
    include_imgs = sum(f["image_count"] for f in config["folders"]
                       if f["status"] == "include")
    print(f"\n학습 대상 이미지: {include_imgs}장")


def get_included_folders():
    """다른 스크립트에서 사용: include 상태 폴더 경로 목록 반환."""
    if not CONFIG_PATH.exists():
        return []
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return [TNMX_DIR / e["folder"] for e in config["folders"]
            if e["status"] == "include"]


def get_all_tnmx_images():
    """다른 스크립트에서 사용: include 폴더 내 모든 이미지 경로 반환.

    Returns:
        list of (img_path, subfolder_name) tuples
    """
    result = []
    for folder in get_included_folders():
        for img in sorted(folder.glob("*.png")):
            result.append((img, folder.name))
    return result


def main():
    parser = argparse.ArgumentParser(description="TNMX 폴더 서베이 및 리뷰")
    parser.add_argument("--review", action="store_true",
                        help="review 폴더 수동 확인 (OpenCV 뷰어)")
    parser.add_argument("--stats", action="store_true",
                        help="현재 config 통계만 출력")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.review:
        review_folders()
    else:
        run_survey()


if __name__ == "__main__":
    main()
