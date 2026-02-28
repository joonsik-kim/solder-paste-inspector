"""
GT 마스크 추출 디버깅 및 개선
라벨 이미지의 보라색 윤곽선을 정확히 추출
"""
import cv2 as cv
import numpy as np
import os

TEST_DIR = "test"
LABEL_DIR = "test_label"
OUTPUT_DIR = "analysis_results/gt_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in sorted(os.listdir(TEST_DIR)):
    test_path = os.path.join(TEST_DIR, fname)
    label_path = os.path.join(LABEL_DIR, fname)

    img = cv.imread(test_path)
    label = cv.imread(label_path)

    if img is None or label is None:
        continue

    if img.shape != label.shape:
        label = cv.resize(label, (img.shape[1], img.shape[0]))

    base = os.path.splitext(fname)[0]
    print(f"\n{'='*60}")
    print(f"이미지: {fname} (크기: {img.shape})")

    # 1. 픽셀 단위 차이
    diff = cv.absdiff(label, img)
    diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

    print(f"  차이 이미지 - min={diff_gray.min()}, max={diff_gray.max()}, "
          f"mean={diff_gray.mean():.1f}, nonzero={np.count_nonzero(diff_gray)}")

    # 차이가 있는 픽셀의 색상 분석
    changed_mask = diff_gray > 5
    changed_pixels = np.count_nonzero(changed_mask)
    print(f"  변경된 픽셀 수: {changed_pixels} / {diff_gray.size}")

    if changed_pixels > 0:
        # 변경된 픽셀의 라벨 이미지 색상 분석
        changed_colors_label = label[changed_mask]
        changed_colors_orig = img[changed_mask]

        print(f"  변경 픽셀 (라벨) BGR 평균: B={changed_colors_label[:,0].mean():.1f}, "
              f"G={changed_colors_label[:,1].mean():.1f}, R={changed_colors_label[:,2].mean():.1f}")
        print(f"  변경 픽셀 (원본) BGR 평균: B={changed_colors_orig[:,0].mean():.1f}, "
              f"G={changed_colors_orig[:,1].mean():.1f}, R={changed_colors_orig[:,2].mean():.1f}")

        # HSV에서 분석
        label_hsv = cv.cvtColor(label, cv.COLOR_BGR2HSV)
        changed_hsv = label_hsv[changed_mask]
        print(f"  변경 픽셀 HSV: H={changed_hsv[:,0].mean():.1f}({changed_hsv[:,0].min()}-{changed_hsv[:,0].max()}), "
              f"S={changed_hsv[:,1].mean():.1f}, V={changed_hsv[:,2].mean():.1f}")

    # 다양한 임계값으로 차이 마스크 생성
    for thresh in [5, 10, 15, 20, 30, 50]:
        _, mask = cv.threshold(diff_gray, thresh, 255, cv.THRESH_BINARY)
        pix_count = np.count_nonzero(mask)
        pct = pix_count / diff_gray.size * 100
        print(f"    thresh={thresh}: {pix_count} pixels ({pct:.1f}%)")

    # 시각화 저장
    # 차이 이미지 (강화)
    diff_enhanced = cv.normalize(diff_gray, None, 0, 255, cv.NORM_MINMAX)
    cv.imwrite(os.path.join(OUTPUT_DIR, f"{base}_diff_gray.png"), diff_enhanced)

    # 차이 컬러
    diff_color = cv.normalize(diff, None, 0, 255, cv.NORM_MINMAX)
    cv.imwrite(os.path.join(OUTPUT_DIR, f"{base}_diff_color.png"), diff_color)

    # 다양한 임계값 마스크
    for thresh in [10, 20, 30]:
        _, mask = cv.threshold(diff_gray, thresh, 255, cv.THRESH_BINARY)
        # 형태학적 연산
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)

        # 채우기
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        for cnt in contours:
            if cv.contourArea(cnt) > 10:
                cv.drawContours(filled, [cnt], -1, 255, -1)

        # 오버레이
        overlay = img.copy()
        colored = np.zeros_like(img)
        colored[filled > 0] = (0, 255, 0)
        overlay = cv.addWeighted(overlay, 0.6, colored, 0.4, 0)

        cv.imwrite(os.path.join(OUTPUT_DIR, f"{base}_gt_thresh{thresh}.png"), filled)
        cv.imwrite(os.path.join(OUTPUT_DIR, f"{base}_overlay_thresh{thresh}.png"), overlay)

    # 나란히 비교: 원본 / 라벨 / 차이
    h, w = img.shape[:2]
    comparison = np.hstack([img, label, cv.cvtColor(diff_enhanced, cv.COLOR_GRAY2BGR)])
    cv.imwrite(os.path.join(OUTPUT_DIR, f"{base}_comparison.png"), comparison)

print("\n완료!")
