"""
프리라벨 리뷰 도우미
====================
자동 생성된 라벨만 LabelMe에서 열어서 리뷰.

사용법:
    python scripts/review_prelabels.py                    # 자동생성 전체
    python scripts/review_prelabels.py --start 100        # 100번째부터
    python scripts/review_prelabels.py --count 50         # 50개만

LabelMe 단축키:
    D       = 다음 이미지 (자동 저장됨)
    A       = 이전 이미지
    Ctrl+S  = 수동 저장
    Delete  = 선택한 폴리곤 삭제
    Ctrl+Z  = 실행 취소

수정 팁:
    - 잘된 건 → 그냥 D (다음) 누르면 자동 저장
    - 틀린 건 → 폴리곤 우클릭 → Delete → 새로 그리기
    - 꼭짓점 조정보다 삭제 후 새로 그리는 게 더 빠름
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_auto_generated_images(source_dir):
    """자동 생성된 JSON이 있는 이미지만 필터링."""
    auto_images = []

    for json_path in sorted(source_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        is_auto = any(
            "auto-generated" in (s.get("description", "") or "")
            for s in shapes
        )

        if is_auto:
            img_path = json_path.with_suffix(".png")
            if img_path.exists():
                auto_images.append(img_path)

    return auto_images


def main():
    parser = argparse.ArgumentParser(description="프리라벨 리뷰 도우미")
    parser.add_argument("source_dir", nargs="?", default="images_main",
                        help="소스 이미지 디렉토리 (기본: images_main)")
    parser.add_argument("--start", type=int, default=0,
                        help="시작 인덱스 (기본: 0)")
    parser.add_argument("--count", type=int, default=0,
                        help="리뷰할 이미지 수 (0=전체)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir

    print("=" * 60)
    print("프리라벨 리뷰 도우미")
    print("=" * 60)

    auto_images = get_auto_generated_images(source_dir)
    print(f"\n자동 생성 라벨: {len(auto_images)}개")

    if args.start > 0:
        auto_images = auto_images[args.start:]
        print(f"시작 위치: {args.start}번째부터")

    if args.count > 0:
        auto_images = auto_images[:args.count]
        print(f"리뷰 대상: {args.count}개")

    if not auto_images:
        print("리뷰할 이미지가 없습니다.")
        return

    print(f"\n리뷰할 이미지: {len(auto_images)}개")
    print(f"첫 번째: {auto_images[0].name}")
    print(f"마지막: {auto_images[-1].name}")

    # 리뷰할 파일 목록 저장
    review_list_path = source_dir / "_review_list.txt"
    with open(review_list_path, "w", encoding="utf-8") as f:
        for img in auto_images:
            f.write(f"{img.name}\n")
    print(f"\n리뷰 목록 저장: {review_list_path}")

    print(f"\n{'=' * 60}")
    print("LabelMe 실행 방법:")
    print(f"  labelme {source_dir} --autosave")
    print()
    print("단축키:")
    print("  D = 다음 (자동 저장)  |  A = 이전")
    print("  Delete = 폴리곤 삭제  |  Ctrl+Z = 취소")
    print()
    print("수정 팁:")
    print("  잘된 건 → D (다음) 누르면 자동 저장")
    print("  틀린 건 → 폴리곤 우클릭 → Delete → 새로 그리기")
    print(f"{'=' * 60}")

    # LabelMe 실행 여부 확인
    answer = input("\nLabelMe를 바로 실행할까요? (y/n): ").strip().lower()
    if answer == 'y':
        first_img = str(auto_images[0])
        print(f"\nLabelMe 실행 중... (첫 이미지: {auto_images[0].name})")
        subprocess.run([
            "labelme", str(source_dir),
            "--autosave",
            "--filename", first_img,
        ])


if __name__ == "__main__":
    main()
