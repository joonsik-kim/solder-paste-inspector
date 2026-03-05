"""
프리라벨 리뷰 도구
==================
자동 생성된 라벨을 빠르게 검토하는 OpenCV 기반 뷰어.

사용법:
    python scripts/review_tool.py                # 미리뷰 전체
    python scripts/review_tool.py --start 100    # 100번째부터

단축키:
    Space/D  = 승인 (reviewed 마킹) → 다음
    X        = 거부 (폴리곤 삭제) → 다음
    E        = LabelMe로 열기 (수정용)
    A        = 이전 이미지
    Q/ESC    = 종료
"""

import cv2 as cv
import numpy as np
import json
import sys
import subprocess
import argparse
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_image_with_overlay(img_path, json_path):
    """이미지 + 폴리곤 오버레이 표시."""
    img = cv.imread(str(img_path))
    if img is None:
        return None

    # 크게 확대 (작은 이미지라서)
    h, w = img.shape[:2]
    scale = max(1, 500 // max(h, w))
    img_large = cv.resize(img, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)

    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for shape in data.get('shapes', []):
            points = shape.get('points', [])
            if len(points) < 3:
                continue
            pts = np.array(points, dtype=np.float32) * scale
            pts = pts.astype(np.int32)

            # 반투명 오버레이
            overlay = img_large.copy()
            cv.fillPoly(overlay, [pts], (0, 255, 0, 80))
            cv.addWeighted(overlay, 0.3, img_large, 0.7, 0, img_large)

            # 폴리곤 외곽선
            cv.polylines(img_large, [pts], True, (0, 255, 0), 2)

            # 꼭짓점 표시
            for pt in pts:
                cv.circle(img_large, tuple(pt), 4, (0, 0, 255), -1)

    return img_large


def mark_reviewed(json_path):
    """JSON의 description을 reviewed로 변경."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for shape in data.get('shapes', []):
        desc = shape.get('description', '') or ''
        if 'auto-generated' in desc:
            shape['description'] = 'reviewed'

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_polygon(json_path):
    """JSON에서 폴리곤 삭제 (빈 shapes)."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data['shapes'] = []

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_unreviewed_images(source_dir):
    """아직 리뷰 안 된 auto-generated 이미지 목록."""
    images = []
    for json_path in sorted(source_dir.glob('*.json')):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shapes = data.get('shapes', [])
        is_auto = any(
            'auto-generated' in (s.get('description', '') or '')
            for s in shapes
        )

        if is_auto:
            img_path = json_path.with_suffix('.png')
            if img_path.exists():
                images.append((img_path, json_path))

    return images


def main():
    parser = argparse.ArgumentParser(description="프리라벨 리뷰 도구")
    parser.add_argument("source_dir", nargs="?", default="images_main")
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir

    images = get_unreviewed_images(source_dir)
    if not images:
        print("리뷰할 이미지가 없습니다.")
        return

    total = len(images)
    idx = args.start
    approved = 0
    rejected = 0

    print(f"리뷰 대상: {total}개")
    print(f"Space/D=승인  X=거부  E=LabelMe편집  A=이전  Q=종료")
    print()

    cv.namedWindow("Review", cv.WINDOW_NORMAL)

    while 0 <= idx < total:
        img_path, json_path = images[idx]

        # 상태 표시
        title = f"[{idx+1}/{total}] {img_path.name} | Space=OK  X=Del  E=Edit  Q=Quit"
        cv.setWindowTitle("Review", title)

        # 이미지 표시
        display = load_image_with_overlay(img_path, json_path)
        if display is None:
            idx += 1
            continue

        # 상단에 정보 추가
        info_bar = np.zeros((40, display.shape[1], 3), dtype=np.uint8)
        cv.putText(info_bar, f"{img_path.name}  [{idx+1}/{total}]  OK:{approved} Del:{rejected}",
                   (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        display = np.vstack([info_bar, display])

        cv.imshow("Review", display)
        key = cv.waitKey(0) & 0xFF

        if key in [ord(' '), ord('d')]:  # 승인
            mark_reviewed(json_path)
            approved += 1
            print(f"  ✅ {img_path.name} → reviewed")
            idx += 1

        elif key == ord('x'):  # 거부
            delete_polygon(json_path)
            rejected += 1
            print(f"  ❌ {img_path.name} → 폴리곤 삭제")
            idx += 1

        elif key == ord('e'):  # LabelMe 편집
            print(f"  ✏️  {img_path.name} → LabelMe 열기...")
            subprocess.run(["labelme", str(img_path), "--autosave"])
            # LabelMe 닫으면 reviewed 마킹
            mark_reviewed(json_path)
            approved += 1
            print(f"  ✅ {img_path.name} → reviewed (편집 완료)")
            idx += 1

        elif key == ord('a'):  # 이전
            idx = max(0, idx - 1)

        elif key in [ord('q'), 27]:  # 종료
            break

    cv.destroyAllWindows()
    print(f"\n{'='*50}")
    print(f"리뷰 결과: 승인 {approved}개, 거부 {rejected}개")
    print(f"남은 미리뷰: {total - approved - rejected}개")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
