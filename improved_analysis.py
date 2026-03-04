"""
개선된 솔더 페이스트 검출 분석 v4
===============================
1단계: 개선된 GT 마스크 추출 (마젠타 색상 정밀 검출)
2단계: 330+ 검출 방법 IoU 평가
3단계: 결과 분석 및 최적 방법 도출
"""

import cv2 as cv
import numpy as np
import os
import json
import time

TEST_DIR = "test"
LABEL_DIR = "test_label"
OUTPUT_DIR = "analysis_results_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 평가 함수
# ============================================================
def calc_iou(pred, gt):
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return inter / union if union > 0 else 0.0

def calc_dice(pred, gt):
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    inter = np.logical_and(p, g).sum()
    total = p.sum() + g.sum()
    return 2.0 * inter / total if total > 0 else 0.0

def calc_precision_recall(pred, gt):
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, np.logical_not(g)).sum()
    fn = np.logical_and(np.logical_not(p), g).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec

def clean_mask(mask, min_area=5, morph_k=3):
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_k, morph_k))
    m = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
    m = cv.morphologyEx(m, cv.MORPH_CLOSE, k)
    contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(m)
    for cnt in contours:
        if cv.contourArea(cnt) >= min_area:
            cv.drawContours(result, [cnt], -1, 255, -1)
    return result


# ============================================================
# 개선된 GT 마스크 추출
# ============================================================
def extract_gt_improved(label_img, orig_img, debug_dir=None, name=""):
    """
    개선된 GT 마스크 추출 v4 (마젠타 색상 감지):
    1. 라벨 이미지에서 마젠타(255,0,255) 색상 정밀 추출
    2. 원본과의 diff + 마젠타 색상 조건으로 이중 검증
    3. 윤곽선 내부 채우기 + 두꺼운 선 보정(erode)
    """
    h, w = label_img.shape[:2]
    img_area = h * w

    # === 1단계: 차이 계산 ===
    diff = cv.absdiff(label_img, orig_img)
    diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

    # === 2단계: 마젠타 색상 정밀 검출 ===
    label_hsv = cv.cvtColor(label_img, cv.COLOR_BGR2HSV)

    # 라벨 BGR 채널 (float)
    lb = label_img[:,:,0].astype(np.float32)  # Blue
    lg = label_img[:,:,1].astype(np.float32)  # Green
    lr = label_img[:,:,2].astype(np.float32)  # Red

    # 방법 A: HSV에서 마젠타 범위
    # 마젠타(255,0,255) → HSV에서 H≈150 (OpenCV 0-180 스케일)
    # 넓은 범위: H=140~180 (wrap-around 고려), S>50, V>50
    magenta_hsv1 = cv.inRange(label_hsv, np.array([140, 50, 50]), np.array([180, 255, 255]))
    # H=0~10 영역도 체크 (wrap-around: 핑크/마젠타 경계)
    magenta_hsv2 = cv.inRange(label_hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    magenta_hsv = cv.bitwise_or(magenta_hsv1, magenta_hsv2)

    # 방법 B: BGR에서 마젠타 조건
    # 마젠타: B 높고, R 높고, G 매우 낮음
    magenta_bgr = np.zeros((h, w), dtype=np.uint8)
    # 순수 마젠타: B>120, R>120, G<100, G < min(B,R)*0.6
    min_br = np.minimum(lb, lr)
    magenta_cond = (lb > 120) & (lr > 120) & (lg < 100) & (lg < min_br * 0.6)
    magenta_bgr[magenta_cond] = 255

    # 방법 C: 차이 이미지에서 마젠타 검출
    diff_hsv = cv.cvtColor(diff, cv.COLOR_BGR2HSV)
    magenta_diff1 = cv.inRange(diff_hsv, np.array([140, 30, 20]), np.array([180, 255, 255]))
    magenta_diff2 = cv.inRange(diff_hsv, np.array([0, 30, 20]), np.array([10, 255, 255]))
    magenta_diff = cv.bitwise_or(magenta_diff1, magenta_diff2)

    # 방법 D: BGR diff에서 마젠타 특성 (B,R 변화 크고 G 변화 작음)
    diff_b = diff[:,:,0].astype(np.float32)
    diff_g = diff[:,:,1].astype(np.float32)
    diff_r = diff[:,:,2].astype(np.float32)
    # 마젠타 추가: B,R이 크게 증가, G는 적게 변함
    magenta_diff_bgr = np.zeros((h, w), dtype=np.uint8)
    magenta_diff_cond = (diff_b > 30) & (diff_r > 30) & (diff_b + diff_r > diff_g * 2.5)
    magenta_diff_bgr[magenta_diff_cond] = 255

    # 원본과 차이가 있는 곳만 (압축 아티팩트 제거)
    diff_significant = (diff_gray > 15).astype(np.uint8) * 255

    # 모든 방법 결합 (OR)
    magenta_combined = np.zeros((h, w), dtype=np.uint8)
    for pm in [magenta_hsv, magenta_bgr, magenta_diff, magenta_diff_bgr]:
        filtered = cv.bitwise_and(pm, diff_significant)
        magenta_combined = cv.bitwise_or(magenta_combined, filtered)

    # === 3단계: 압축 아티팩트 필터링 ===
    # 마젠타는 B,R 변화가 크고 G 변화가 작음
    color_diff_ratio = (diff_b + diff_r + 1) / (diff_g + 1)
    color_filter = (color_diff_ratio > 1.8).astype(np.uint8) * 255
    magenta_combined = cv.bitwise_and(magenta_combined, color_filter)

    # === 4단계: 형태학적 연산으로 윤곽선 연결 ===
    img_diag = np.sqrt(h*h + w*w)
    close_k = max(3, int(img_diag * 0.06))
    if close_k % 2 == 0:
        close_k += 1

    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_k, close_k))
    closed = cv.morphologyEx(magenta_combined, cv.MORPH_CLOSE, kernel_close)

    # dilate로 끊긴 부분 연결
    dilate_k = max(3, int(img_diag * 0.04))
    if dilate_k % 2 == 0:
        dilate_k += 1
    kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate_k, dilate_k))
    dilated = cv.dilate(closed, kernel_dilate, iterations=1)

    # === 5단계: 윤곽선 내부 채우기 ===
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max(10, img_area * 0.005):
            cv.drawContours(filled, [cnt], -1, 255, -1)

    # === 6단계: 두꺼운 마킹 보정 (erode) ===
    # 마젠타 선이 두꺼우므로 약간 축소하여 실제 영역에 가깝게 조정
    filled_area = np.count_nonzero(filled)
    if filled_area > img_area * 0.05:  # 충분히 큰 영역이 검출된 경우만
        erode_k = max(1, int(img_diag * 0.015))
        if erode_k % 2 == 0:
            erode_k += 1
        if erode_k >= 3:
            kernel_erode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erode_k, erode_k))
            filled = cv.erode(filled, kernel_erode, iterations=1)

    # === 7단계: 폴백 - 검출 실패 시 diff 기반 ===
    filled_area = np.count_nonzero(filled)
    if filled_area < img_area * 0.01:
        # 마젠타 검출 실패 → diff 기반 폴백
        if diff_gray.max() > 0:
            sorted_vals = np.sort(diff_gray[diff_gray > 0])
            if len(sorted_vals) > 0:
                thresh_val = sorted_vals[max(0, int(len(sorted_vals) * 0.3))]
                thresh_val = max(thresh_val, 25)
                _, filled = cv.threshold(diff_gray, thresh_val, 255, cv.THRESH_BINARY)
                filled = clean_mask(filled, min_area=max(5, int(img_area * 0.005)))

    # 디버그 저장
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv.imwrite(os.path.join(debug_dir, f"{name}_magenta_hsv.png"), magenta_hsv)
        cv.imwrite(os.path.join(debug_dir, f"{name}_magenta_bgr.png"), magenta_bgr)
        cv.imwrite(os.path.join(debug_dir, f"{name}_magenta_diff.png"), magenta_diff)
        cv.imwrite(os.path.join(debug_dir, f"{name}_magenta_diff_bgr.png"), magenta_diff_bgr)
        cv.imwrite(os.path.join(debug_dir, f"{name}_magenta_combined.png"), magenta_combined)
        cv.imwrite(os.path.join(debug_dir, f"{name}_closed.png"), closed)
        cv.imwrite(os.path.join(debug_dir, f"{name}_dilated.png"), dilated)
        cv.imwrite(os.path.join(debug_dir, f"{name}_gt_final.png"), filled)

        # 오버레이 시각화
        overlay = orig_img.copy()
        colored = np.zeros_like(orig_img)
        colored[filled > 0] = (0, 255, 0)
        overlay = cv.addWeighted(overlay, 0.6, colored, 0.4, 0)
        cv.imwrite(os.path.join(debug_dir, f"{name}_gt_overlay.png"), overlay)

        gt_pct = np.count_nonzero(filled) / img_area * 100
        print(f"  {name}: GT 영역 = {np.count_nonzero(filled)}px ({gt_pct:.1f}%)")

    return filled


# ============================================================
# 검출 방법들 (200+ 변형)
# ============================================================
def run_all_methods(img, gt_mask, fname):
    """모든 검출 방법 실행 및 평가"""
    results = {}
    h, w = img.shape[:2]
    img_area = h * w

    # 최소 면적 (이미지 면적의 0.5%)
    min_a = max(3, int(img_area * 0.003))

    # 채널 분리
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 색상공간 변환
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    l_ch, a_ch, b_lab = lab[:,:,0], lab[:,:,1], lab[:,:,2]

    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y_ch, cr_ch, cb_ch = ycrcb[:,:,0], ycrcb[:,:,1], ycrcb[:,:,2]

    def evaluate(mask, method_name):
        cleaned = clean_mask(mask, min_area=min_a)
        iou = calc_iou(cleaned, gt_mask)
        dice = calc_dice(cleaned, gt_mask)
        prec, rec = calc_precision_recall(cleaned, gt_mask)
        results[method_name] = {
            'iou': iou, 'dice': dice,
            'precision': prec, 'recall': rec,
            'pred_area': int(np.count_nonzero(cleaned)),
            'gt_area': int(np.count_nonzero(gt_mask)),
            'mask': cleaned
        }

    # ==============================
    # 카테고리 1: Blue 채널 임계값
    # ==============================
    for lo in [50, 80, 100, 120, 150, 170, 200]:
        mask = cv.inRange(b, lo, 255)
        evaluate(mask, f"Blue_{lo}-255")
    for lo, hi in [(50,150), (80,180), (100,200), (120,220), (150,255)]:
        mask = cv.inRange(b, lo, hi)
        evaluate(mask, f"Blue_{lo}-{hi}")

    # ==============================
    # 카테고리 2: Green 채널 임계값
    # ==============================
    for lo in [50, 80, 100, 120, 150, 170, 200]:
        mask = cv.inRange(g, lo, 255)
        evaluate(mask, f"Green_{lo}-255")

    # ==============================
    # 카테고리 3: Red 채널 임계값
    # ==============================
    for lo in [50, 80, 100, 120, 150]:
        mask = cv.inRange(r, lo, 255)
        evaluate(mask, f"Red_{lo}-255")

    # ==============================
    # 카테고리 4: 채널 차이/비율
    # ==============================
    bf, gf, rf = b.astype(np.float32), g.astype(np.float32), r.astype(np.float32)

    # B-R
    diff_br = bf - rf
    for t in [10, 20, 30, 50, 80]:
        mask = (diff_br > t).astype(np.uint8) * 255
        evaluate(mask, f"B-R_gt{t}")

    # B-G
    diff_bg = bf - gf
    for t in [10, 20, 30, 50, 80]:
        mask = (diff_bg > t).astype(np.uint8) * 255
        evaluate(mask, f"B-G_gt{t}")

    # G-R
    diff_gr = gf - rf
    for t in [10, 20, 30, 50]:
        mask = (diff_gr > t).astype(np.uint8) * 255
        evaluate(mask, f"G-R_gt{t}")

    # B / (B+G+R) 비율
    total = bf + gf + rf + 1
    b_ratio = bf / total
    for t in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        mask = (b_ratio > t).astype(np.uint8) * 255
        evaluate(mask, f"Bratio_gt{t}")

    # B / (B+R) 비율
    br_total = bf + rf + 1
    b_br_ratio = bf / br_total
    for t in [0.55, 0.6, 0.65, 0.7]:
        mask = (b_br_ratio > t).astype(np.uint8) * 255
        evaluate(mask, f"B_BR_ratio_gt{t}")

    # (B+G) / (R+1)
    bg_r_ratio = (bf + gf) / (rf + 1)
    for t in [2, 3, 4, 5]:
        mask = (bg_r_ratio > t).astype(np.uint8) * 255
        evaluate(mask, f"BG_R_ratio_gt{t}")

    # ==============================
    # 카테고리 5: HSV 색상공간
    # ==============================
    # Hue 기반 (파란색 영역 H=90~130)
    for lo, hi in [(85,130), (90,125), (95,120), (100,130), (80,140)]:
        mask = cv.inRange(h_ch, lo, hi)
        evaluate(mask, f"Hue_{lo}-{hi}")

    # Saturation 기반
    for lo in [30, 50, 80, 100, 120, 150]:
        mask = cv.inRange(s_ch, lo, 255)
        evaluate(mask, f"Sat_{lo}-255")

    # Value 기반 (어두운 영역)
    for hi in [80, 100, 120, 150, 180]:
        mask = cv.inRange(v_ch, 0, hi)
        evaluate(mask, f"Val_0-{hi}")

    # 밝은 영역
    for lo in [100, 150, 180, 200]:
        mask = cv.inRange(v_ch, lo, 255)
        evaluate(mask, f"Val_{lo}-255")

    # HSV 조합 (파란색 + 채도 + 밝기)
    for h_lo, h_hi, s_lo, v_lo, v_hi in [
        (85, 130, 30, 30, 255), (90, 125, 50, 50, 255),
        (85, 140, 30, 0, 200), (80, 130, 20, 0, 150),
        (90, 130, 50, 50, 200), (85, 125, 40, 30, 180),
        (80, 140, 20, 20, 255), (95, 125, 60, 60, 200),
    ]:
        lower = np.array([h_lo, s_lo, v_lo])
        upper = np.array([h_hi, 255, v_hi])
        mask = cv.inRange(hsv, lower, upper)
        evaluate(mask, f"HSV_h{h_lo}-{h_hi}_s{s_lo}_v{v_lo}-{v_hi}")

    # ==============================
    # 카테고리 6: Lab 색상공간
    # ==============================
    # L 채널 (밝기)
    for hi in [80, 100, 120, 150]:
        mask = cv.inRange(l_ch, 0, hi)
        evaluate(mask, f"Lab_L_0-{hi}")

    # a 채널 (Green-Red, 128 기준)
    for hi in [115, 120, 125, 128]:
        mask = cv.inRange(a_ch, 0, hi)
        evaluate(mask, f"Lab_a_0-{hi}")
    for lo in [128, 130, 135, 140]:
        mask = cv.inRange(a_ch, lo, 255)
        evaluate(mask, f"Lab_a_{lo}-255")

    # b 채널 (Blue-Yellow, 128 기준)
    for hi in [100, 110, 115, 120, 125, 128]:
        mask = cv.inRange(b_lab, 0, hi)
        evaluate(mask, f"Lab_b_0-{hi}")

    # ==============================
    # 카테고리 7: YCrCb 색상공간
    # ==============================
    for lo in [128, 135, 140, 145, 150, 160]:
        mask = cv.inRange(cb_ch, lo, 255)
        evaluate(mask, f"Cb_{lo}-255")

    for hi in [110, 115, 120, 125, 128]:
        mask = cv.inRange(cr_ch, 0, hi)
        evaluate(mask, f"Cr_0-{hi}")

    # ==============================
    # 카테고리 8: Otsu 자동 임계값
    # ==============================
    for ch, ch_name in [(b, "Blue"), (g, "Green"), (gray, "Gray"),
                         (s_ch, "Sat"), (v_ch, "Val"), (l_ch, "Lab_L"),
                         (b_lab, "Lab_b"), (cb_ch, "Cb")]:
        _, mask = cv.threshold(ch, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        evaluate(mask, f"Otsu_{ch_name}")
        _, mask_inv = cv.threshold(ch, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        evaluate(mask_inv, f"Otsu_{ch_name}_inv")

    # ==============================
    # 카테고리 9: 적응형 임계값
    # ==============================
    for ch, ch_name in [(b, "Blue"), (gray, "Gray"), (l_ch, "Lab_L")]:
        for bs in [7, 11, 15, 21]:
            for c_val in [2, 5, 10]:
                if bs > min(h, w):
                    continue
                mask = cv.adaptiveThreshold(ch, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv.THRESH_BINARY, bs, c_val)
                evaluate(mask, f"Adaptive_{ch_name}_bs{bs}_c{c_val}")
                mask_inv = cv.adaptiveThreshold(ch, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv.THRESH_BINARY_INV, bs, c_val)
                evaluate(mask_inv, f"Adaptive_{ch_name}_inv_bs{bs}_c{c_val}")

    # ==============================
    # 카테고리 10: CLAHE + Otsu
    # ==============================
    for cl in [2.0, 4.0, 8.0]:
        for tile in [4, 8]:
            tg = (tile, tile)
            clahe = cv.createCLAHE(clipLimit=cl, tileGridSize=tg)
            for ch, ch_name in [(b, "Blue"), (gray, "Gray")]:
                enhanced = clahe.apply(ch)
                _, mask = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                evaluate(mask, f"CLAHE_{ch_name}_cl{cl}_t{tile}")
                _, mask_inv = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
                evaluate(mask_inv, f"CLAHE_{ch_name}_inv_cl{cl}_t{tile}")

    # ==============================
    # 카테고리 11: 엣지 기반
    # ==============================
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    for lo_t, hi_t in [(20,60), (30,90), (50,150), (20,100)]:
        edges = cv.Canny(blurred, lo_t, hi_t)
        # 엣지를 확장하여 영역으로 변환
        k_e = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        dilated = cv.dilate(edges, k_e, iterations=2)
        k_c = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, k_c)
        contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        edge_mask = np.zeros_like(gray)
        for cnt in contours:
            if cv.contourArea(cnt) > min_a:
                cv.drawContours(edge_mask, [cnt], -1, 255, -1)
        evaluate(edge_mask, f"Canny_{lo_t}-{hi_t}")

    # Gradient magnitude (Sobel)
    for ksize in [3, 5]:
        sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=ksize)
        sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=ksize)
        mag = np.sqrt(sobelx**2 + sobely**2)
        mag_norm = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        for t in [20, 30, 50, 80]:
            mask = (mag_norm > t).astype(np.uint8) * 255
            evaluate(mask, f"Gradient_k{ksize}_t{t}")

    # ==============================
    # 카테고리 12: K-means 클러스터링
    # ==============================
    for k in [2, 3, 4]:
        for color_space, cs_name in [(img, "BGR"), (hsv, "HSV"), (lab, "Lab")]:
            data = color_space.reshape((-1, 3)).astype(np.float32)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1.0)
            _, labels_km, centers = cv.kmeans(data, k, None, criteria, 3, cv.KMEANS_PP_CENTERS)
            labels_img = labels_km.reshape(h, w)

            # 각 클러스터를 후보로 평가
            for ci in range(k):
                mask = (labels_img == ci).astype(np.uint8) * 255
                # 이 클러스터가 전체 면적의 80% 이상이면 반전
                if np.count_nonzero(mask) > img_area * 0.8:
                    continue
                evaluate(mask, f"KMeans_{cs_name}_k{k}_c{ci}")

    # ==============================
    # 카테고리 13: 색상 거리 기반
    # ==============================
    # BGR 공간에서 특정 색상과의 거리
    target_colors_bgr = [
        (255, 0, 0, "PureBlue"),     # 순수 파랑
        (200, 50, 0, "DarkBlue"),    # 어두운 파랑
        (150, 100, 0, "BlueGreen"),  # 파랑-초록
        (100, 0, 0, "DeepBlue"),     # 깊은 파랑
        (0, 0, 0, "Black"),          # 검정
        (50, 50, 0, "DarkCyan"),     # 어두운 시안
    ]
    for tb, tg, tr, tname in target_colors_bgr:
        target = np.array([tb, tg, tr], dtype=np.float32)
        dist = np.sqrt(np.sum((img.astype(np.float32) - target) ** 2, axis=2))
        for d_thresh in [50, 80, 100, 120, 150]:
            mask = (dist < d_thresh).astype(np.uint8) * 255
            if np.count_nonzero(mask) > img_area * 0.85:
                continue
            evaluate(mask, f"ColorDist_{tname}_{d_thresh}")

    # Lab 공간에서 색상 거리
    target_colors_lab = [
        (30, 128, 80, "DarkBlue_Lab"),
        (50, 128, 90, "MidBlue_Lab"),
        (20, 128, 70, "DeepBlue_Lab"),
    ]
    for tl, ta, tb_val, tname in target_colors_lab:
        target = np.array([tl, ta, tb_val], dtype=np.float32)
        dist = np.sqrt(np.sum((lab.astype(np.float32) - target) ** 2, axis=2))
        for d_thresh in [30, 50, 70, 100]:
            mask = (dist < d_thresh).astype(np.uint8) * 255
            if np.count_nonzero(mask) > img_area * 0.85:
                continue
            evaluate(mask, f"LabDist_{tname}_{d_thresh}")

    # ==============================
    # 카테고리 14: FloodFill
    # ==============================
    # Blue 채널에서 가장 높은 값의 위치에서 시작
    for ch, ch_name in [(b, "Blue"), (gray, "Gray")]:
        # 최대값 위치
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(ch)
        for lo_diff, hi_diff in [(10,10), (20,20), (30,30), (40,40), (50,50)]:
            flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)
            ch_copy = ch.copy()
            cv.floodFill(ch_copy, flood_mask, max_loc, 255,
                         loDiff=lo_diff, upDiff=hi_diff,
                         flags=cv.FLOODFILL_MASK_ONLY | (255 << 8))
            result_mask = flood_mask[1:-1, 1:-1] * 255
            if np.count_nonzero(result_mask) > img_area * 0.85:
                continue
            evaluate(result_mask, f"FloodFill_{ch_name}_max_d{lo_diff}")

        # 최소값 위치 (어두운 영역)
        for lo_diff, hi_diff in [(10,10), (20,20), (30,30), (40,40)]:
            flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)
            ch_copy = ch.copy()
            cv.floodFill(ch_copy, flood_mask, min_loc, 255,
                         loDiff=lo_diff, upDiff=hi_diff,
                         flags=cv.FLOODFILL_MASK_ONLY | (255 << 8))
            result_mask = flood_mask[1:-1, 1:-1] * 255
            if np.count_nonzero(result_mask) > img_area * 0.85:
                continue
            evaluate(result_mask, f"FloodFill_{ch_name}_min_d{lo_diff}")

    # ==============================
    # 카테고리 15: Connected Components
    # ==============================
    for ch, ch_name in [(b, "Blue"), (gray, "Gray")]:
        _, binary = cv.threshold(ch, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        num_labels, labels_cc, stats, centroids = cv.connectedComponentsWithStats(binary)
        # 가장 큰 컴포넌트 선택 (배경 제외)
        if num_labels > 1:
            areas = stats[1:, cv.CC_STAT_AREA]
            largest_idx = np.argmax(areas) + 1
            mask = (labels_cc == largest_idx).astype(np.uint8) * 255
            evaluate(mask, f"CC_largest_{ch_name}")

            # 두 번째로 큰 컴포넌트
            if num_labels > 2:
                areas_sorted = np.argsort(areas)[::-1]
                if len(areas_sorted) > 1:
                    second_idx = areas_sorted[1] + 1
                    mask2 = (labels_cc == second_idx).astype(np.uint8) * 255
                    evaluate(mask2, f"CC_2nd_{ch_name}")

    # ==============================
    # 카테고리 16: 다중 채널 조합
    # ==============================
    # Blue 높고 + Red 낮음
    for b_lo, r_hi in [(100, 100), (120, 80), (80, 120), (150, 100)]:
        mask = ((b > b_lo) & (r < r_hi)).astype(np.uint8) * 255
        evaluate(mask, f"Blue_hi_Red_lo_{b_lo}_{r_hi}")

    # Blue 높고 + Green 중간
    for b_lo, g_lo, g_hi in [(100, 50, 180), (120, 80, 200), (80, 30, 150)]:
        mask = ((b > b_lo) & (g > g_lo) & (g < g_hi)).astype(np.uint8) * 255
        evaluate(mask, f"Blue_Green_range_{b_lo}_{g_lo}_{g_hi}")

    # HSV + Blue channel 조합
    for h_lo, h_hi, b_lo in [(85, 130, 80), (90, 125, 100), (80, 140, 60)]:
        mask_hue = cv.inRange(h_ch, h_lo, h_hi)
        mask_blue = cv.inRange(b, b_lo, 255)
        mask = cv.bitwise_and(mask_hue, mask_blue)
        evaluate(mask, f"HSV_Blue_h{h_lo}-{h_hi}_b{b_lo}")

    # Lab_b 낮음 + Blue 채널 높음
    for lb_hi, b_lo in [(120, 80), (115, 100), (125, 60), (110, 120)]:
        mask_lb = cv.inRange(b_lab, 0, lb_hi)
        mask_b = cv.inRange(b, b_lo, 255)
        mask = cv.bitwise_and(mask_lb, mask_b)
        evaluate(mask, f"Lab_b_Blue_{lb_hi}_{b_lo}")

    # 투표 기반 (여러 방법의 합)
    vote_methods = [
        cv.inRange(b, 100, 255),
        cv.inRange(h_ch, 85, 130),
        cv.inRange(s_ch, 50, 255),
        cv.inRange(b_lab, 0, 120),
        cv.inRange(cb_ch, 135, 255),
    ]
    for min_votes in [2, 3, 4]:
        vote_sum = sum((m > 0).astype(np.float32) for m in vote_methods)
        mask = (vote_sum >= min_votes).astype(np.uint8) * 255
        evaluate(mask, f"Vote_{min_votes}of5")

    # ==============================
    # 카테고리 17: 가중 조합
    # ==============================
    b_norm = b.astype(np.float32) / 255
    s_norm = s_ch.astype(np.float32) / 255
    lb_norm = (255 - b_lab).astype(np.float32) / 255  # Lab_b 반전 (낮을수록 파랑)
    cb_norm = cb_ch.astype(np.float32) / 255

    for wb, ws, wlb, wcb in [(0.3, 0.2, 0.3, 0.2), (0.4, 0.1, 0.3, 0.2),
                              (0.25, 0.25, 0.25, 0.25), (0.5, 0.1, 0.2, 0.2)]:
        weighted = wb * b_norm + ws * s_norm + wlb * lb_norm + wcb * cb_norm
        w_norm = cv.normalize(weighted, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        _, mask = cv.threshold(w_norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        evaluate(mask, f"Weighted_{wb}_{ws}_{wlb}_{wcb}")

    # ==============================
    # 카테고리 18: 히스토그램 역투영
    # ==============================
    # Blue 채널이 높은 영역의 HSV 히스토그램 기반
    blue_region = b > np.percentile(b, 70)
    if np.count_nonzero(blue_region) > 10:
        roi_hsv = hsv[blue_region]
        if len(roi_hsv) > 0:
            roi_hsv_2d = roi_hsv.reshape(-1, 1, 3)
            roi_hist = cv.calcHist([roi_hsv_2d], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
            backproj = cv.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
            for t in [30, 50, 80, 100]:
                mask = (backproj > t).astype(np.uint8) * 255
                evaluate(mask, f"BackProj_t{t}")

    # ==============================
    # 카테고리 19: Normalized RGB (산업용 AOI 표준)
    # ==============================
    total_f = b.astype(np.float32) + g.astype(np.float32) + r.astype(np.float32)
    total_f = np.maximum(total_f, 1.0)
    norm_b = b.astype(np.float32) / total_f
    norm_g = g.astype(np.float32) / total_f
    norm_r = r.astype(np.float32) / total_f

    # b 비율 임계값
    for t in [0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55]:
        mask = (norm_b >= t).astype(np.uint8) * 255
        evaluate(mask, f"NormRGB_b_gt_{t}")

    # r 비율 임계값 (Red가 낮은 = 솔더 아닌 영역 제외)
    for t in [0.25, 0.30, 0.35]:
        mask = (norm_r < t).astype(np.uint8) * 255
        evaluate(mask, f"NormRGB_r_lt_{t}")

    # b-r 차이 (Blue 비율이 Red보다 높은 영역)
    diff_br = norm_b - norm_r
    for t in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        mask = (diff_br >= t).astype(np.uint8) * 255
        evaluate(mask, f"NormRGB_b-r_gt_{t}")

    # b-g 차이 (Blue 비율이 Green보다 높은 영역)
    diff_bg = norm_b - norm_g
    for t in [0.05, 0.08, 0.10, 0.15]:
        mask = (diff_bg >= t).astype(np.uint8) * 255
        evaluate(mask, f"NormRGB_b-g_gt_{t}")

    # ==============================
    # 카테고리 20: 채널 비율 (B/G, B/R 등)
    # ==============================
    g_f = np.maximum(g.astype(np.float32), 1.0)
    r_f = np.maximum(r.astype(np.float32), 1.0)
    b_f = np.maximum(b.astype(np.float32), 1.0)

    ratio_bg = b.astype(np.float32) / g_f
    for t in [1.2, 1.3, 1.5, 1.8, 2.0]:
        mask = (ratio_bg >= t).astype(np.uint8) * 255
        evaluate(mask, f"Ratio_BG_gt_{t}")

    ratio_br = b.astype(np.float32) / r_f
    for t in [1.2, 1.5, 2.0]:
        mask = (ratio_br >= t).astype(np.uint8) * 255
        evaluate(mask, f"Ratio_BR_gt_{t}")

    # B/G와 B/R 동시 조건
    for t in [1.1, 1.2, 1.3]:
        mask = ((ratio_bg >= t) & (ratio_br >= t)).astype(np.uint8) * 255
        evaluate(mask, f"Ratio_BG_and_BR_gt_{t}")

    # ==============================
    # 카테고리 21: Bilateral Filter + 기존 방법
    # ==============================
    for d_val in [5, 9]:
        denoised = cv.bilateralFilter(img, d_val, 75, 75)
        den_b = denoised[:,:,0]
        den_hsv = cv.cvtColor(denoised, cv.COLOR_BGR2HSV)
        den_s = den_hsv[:,:,1]

        # Bilateral + Otsu Blue inv
        _, m = cv.threshold(den_b, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        evaluate(m, f"Bilateral_d{d_val}_Otsu_Blue_inv")

        # Bilateral + CLAHE Blue inv
        for cl in [4.0, 8.0]:
            clahe = cv.createCLAHE(clipLimit=cl, tileGridSize=(4, 4))
            enhanced = clahe.apply(den_b)
            _, m = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            evaluate(m, f"Bilateral_d{d_val}_CLAHE_Blue_inv_cl{cl}")

        # Bilateral + Saturation
        for lo in [30, 50]:
            m = cv.inRange(den_s, lo, 255)
            evaluate(m, f"Bilateral_d{d_val}_Sat_{lo}-255")

        # Bilateral + Normalized RGB
        den_total = denoised[:,:,0].astype(np.float32) + denoised[:,:,1].astype(np.float32) + denoised[:,:,2].astype(np.float32)
        den_total = np.maximum(den_total, 1.0)
        den_norm_b = denoised[:,:,0].astype(np.float32) / den_total
        for t in [0.38, 0.40, 0.42, 0.45]:
            m = (den_norm_b >= t).astype(np.uint8) * 255
            evaluate(m, f"Bilateral_d{d_val}_NormRGB_b_gt_{t}")

    # ==============================
    # 카테고리 22: NLM + 기존 방법
    # ==============================
    try:
        nlm = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        nlm_b = nlm[:,:,0]
        nlm_hsv = cv.cvtColor(nlm, cv.COLOR_BGR2HSV)
        nlm_s = nlm_hsv[:,:,1]

        _, m = cv.threshold(nlm_b, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        evaluate(m, "NLM_Otsu_Blue_inv")

        for cl in [4.0, 8.0]:
            clahe = cv.createCLAHE(clipLimit=cl, tileGridSize=(4, 4))
            enhanced = clahe.apply(nlm_b)
            _, m = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            evaluate(m, f"NLM_CLAHE_Blue_inv_cl{cl}")

        m = cv.inRange(nlm_s, 30, 255)
        evaluate(m, "NLM_Sat_30-255")

        nlm_total = nlm[:,:,0].astype(np.float32) + nlm[:,:,1].astype(np.float32) + nlm[:,:,2].astype(np.float32)
        nlm_total = np.maximum(nlm_total, 1.0)
        nlm_norm_b = nlm[:,:,0].astype(np.float32) / nlm_total
        for t in [0.38, 0.40, 0.42]:
            m = (nlm_norm_b >= t).astype(np.uint8) * 255
            evaluate(m, f"NLM_NormRGB_b_gt_{t}")
    except Exception:
        pass  # NLM이 실패하면 건너뜀

    # ==============================
    # 카테고리 23: 그라데이션 패턴
    # ==============================
    # Blue gradient 크기 (필렛 전환 영역)
    grad_b_x = cv.Sobel(norm_b, cv.CV_32F, 1, 0, ksize=3)
    grad_b_y = cv.Sobel(norm_b, cv.CV_32F, 0, 1, ksize=3)
    grad_b_mag = np.sqrt(grad_b_x**2 + grad_b_y**2)
    grad_b_norm = cv.normalize(grad_b_mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    for t in [20, 30, 50]:
        base = cv.threshold(grad_b_norm, t, 255, cv.THRESH_BINARY)[1]
        kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        m = cv.dilate(base, kern, iterations=2)
        evaluate(m, f"GradPattern_NormB_t{t}")

    # Hue gradient (색상 전환 패턴)
    h_f = h_ch.astype(np.float32)
    grad_h_x = cv.Sobel(h_f, cv.CV_32F, 1, 0, ksize=3)
    grad_h_y = cv.Sobel(h_f, cv.CV_32F, 0, 1, ksize=3)
    grad_h_mag = np.sqrt(grad_h_x**2 + grad_h_y**2)
    grad_h_norm = cv.normalize(grad_h_mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    for t in [20, 40]:
        base = cv.threshold(grad_h_norm, t, 255, cv.THRESH_BINARY)[1]
        kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        m = cv.dilate(base, kern, iterations=2)
        evaluate(m, f"GradPattern_Hue_t{t}")

    return results


# ============================================================
# 메인 실행
# ============================================================
def main():
    print("=" * 70)
    print("개선된 솔더 페이스트 검출 분석 v4 (마젠타 GT)")
    print("=" * 70)

    all_results = {}
    gt_debug_dir = os.path.join(OUTPUT_DIR, "gt_debug")

    fnames = sorted(os.listdir(TEST_DIR))
    print(f"\n테스트 이미지: {len(fnames)}개")

    # ============================================================
    # Phase 1: GT 마스크 추출
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 1: GT 마스크 추출 (마젠타 색상 검출)")
    print("=" * 70)

    gt_masks = {}
    for fname in fnames:
        test_path = os.path.join(TEST_DIR, fname)
        label_path = os.path.join(LABEL_DIR, fname)

        img = cv.imread(test_path)
        label = cv.imread(label_path)

        if img is None or label is None:
            continue

        if img.shape != label.shape:
            label = cv.resize(label, (img.shape[1], img.shape[0]))

        base = os.path.splitext(fname)[0]
        print(f"\n  처리: {fname} (크기: {img.shape})")

        gt_mask = extract_gt_improved(label, img, debug_dir=gt_debug_dir, name=base)
        gt_masks[fname] = gt_mask

    # ============================================================
    # Phase 2: 모든 검출 방법 실행
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 2: 330+ 검출 방법 IoU 평가")
    print("=" * 70)

    for fname in fnames:
        test_path = os.path.join(TEST_DIR, fname)
        img = cv.imread(test_path)
        if img is None or fname not in gt_masks:
            continue

        base = os.path.splitext(fname)[0]
        gt_mask = gt_masks[fname]

        print(f"\n  분석 중: {fname}")
        start = time.time()

        results = run_all_methods(img, gt_mask, fname)
        elapsed = time.time() - start

        print(f"    {len(results)}개 방법 완료 ({elapsed:.1f}초)")
        all_results[fname] = results

        # 이미지별 Top 10 출력
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['iou'], reverse=True)
        print(f"\n    Top 10 (IoU 기준):")
        for i, (method, res) in enumerate(sorted_methods[:10]):
            print(f"      {i+1:2d}. {method:<40s} IoU={res['iou']:.4f}  "
                  f"Dice={res['dice']:.4f}  P={res['precision']:.3f}  R={res['recall']:.3f}")

        # Top 10 시각화 저장
        img_dir = os.path.join(OUTPUT_DIR, base)
        os.makedirs(img_dir, exist_ok=True)

        for i, (method, res) in enumerate(sorted_methods[:10]):
            overlay = img.copy()
            mask_vis = res['mask']
            colored = np.zeros_like(img)
            colored[mask_vis > 0] = (0, 255, 0)
            overlay = cv.addWeighted(overlay, 0.5, colored, 0.5, 0)

            # GT 윤곽선 빨간색으로 표시
            gt_contours, _ = cv.findContours(gt_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(overlay, gt_contours, -1, (0, 0, 255), 1)

            info = f"IoU={res['iou']:.3f}"
            cv.putText(overlay, f"{method}", (2, 10), cv.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 1)
            cv.putText(overlay, info, (2, 20), cv.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 1)

            cv.imwrite(os.path.join(img_dir, f"rank{i+1:02d}_{method}.png"), overlay)

    # ============================================================
    # Phase 3: 종합 분석
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 3: 종합 분석")
    print("=" * 70)

    # 모든 이미지에 대한 평균 IoU 계산
    method_avg_iou = {}
    all_method_names = set()
    for fname, results in all_results.items():
        for method in results:
            all_method_names.add(method)

    for method in all_method_names:
        ious = []
        for fname, results in all_results.items():
            if method in results:
                ious.append(results[method]['iou'])
        if len(ious) == len(all_results):  # 모든 이미지에서 실행된 방법만
            method_avg_iou[method] = {
                'avg_iou': np.mean(ious),
                'min_iou': min(ious),
                'max_iou': max(ious),
                'std_iou': np.std(ious),
                'per_image': {fname: results[method]['iou']
                              for fname, results in all_results.items()
                              if method in results}
            }

    # 평균 IoU 기준 정렬
    sorted_avg = sorted(method_avg_iou.items(), key=lambda x: x[1]['avg_iou'], reverse=True)

    print(f"\n총 {len(sorted_avg)}개 방법 (모든 이미지에서 실행)")
    print(f"\n{'='*90}")
    print(f"{'순위':<5} {'방법':<45} {'평균IoU':<10} {'최소':<8} {'최대':<8} {'표준편차':<8}")
    print(f"{'='*90}")

    for i, (method, stats) in enumerate(sorted_avg[:30]):
        per_img_str = "  ".join([f"{os.path.splitext(fn)[0][-5:]}={iou:.3f}"
                                  for fn, iou in stats['per_image'].items()])
        print(f"{i+1:3d}.  {method:<45s} {stats['avg_iou']:.4f}    "
              f"{stats['min_iou']:.4f}  {stats['max_iou']:.4f}  {stats['std_iou']:.4f}  "
              f"| {per_img_str}")

    # ============================================================
    # Phase 4: 최적 방법 안정성 분석
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 4: 안정성 분석 (분산이 낮은 방법)")
    print("=" * 70)

    # 평균 IoU > 0.3이면서 표준편차가 낮은 방법
    stable_methods = [(m, s) for m, s in sorted_avg if s['avg_iou'] > 0.3]
    stable_methods.sort(key=lambda x: x[1]['std_iou'])

    print(f"\n{'순위':<5} {'방법':<45} {'평균IoU':<10} {'표준편차':<10} {'최소IoU':<10}")
    for i, (method, stats) in enumerate(stable_methods[:15]):
        print(f"{i+1:3d}.  {method:<45s} {stats['avg_iou']:.4f}    "
              f"{stats['std_iou']:.4f}    {stats['min_iou']:.4f}")

    # ============================================================
    # Phase 5: 카테고리별 최적 방법
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 5: 카테고리별 최적 방법")
    print("=" * 70)

    categories = {
        'Blue 채널': lambda m: m.startswith('Blue_'),
        'Green 채널': lambda m: m.startswith('Green_'),
        'Red 채널': lambda m: m.startswith('Red_'),
        '채널 차이/비율': lambda m: any(m.startswith(p) for p in ['B-R', 'B-G', 'G-R', 'Bratio', 'B_BR', 'BG_R']),
        'HSV': lambda m: m.startswith('Hue_') or m.startswith('Sat_') or m.startswith('Val_') or m.startswith('HSV_'),
        'Lab': lambda m: m.startswith('Lab_'),
        'YCrCb': lambda m: m.startswith('Cb_') or m.startswith('Cr_'),
        'Otsu': lambda m: m.startswith('Otsu_'),
        '적응형': lambda m: m.startswith('Adaptive_'),
        'CLAHE': lambda m: m.startswith('CLAHE_'),
        '엣지': lambda m: m.startswith('Canny_') or m.startswith('Gradient_'),
        'K-means': lambda m: m.startswith('KMeans_'),
        '색상 거리': lambda m: m.startswith('ColorDist_') or m.startswith('LabDist_'),
        'FloodFill': lambda m: m.startswith('FloodFill_'),
        'Connected Comp.': lambda m: m.startswith('CC_'),
        '다중 채널': lambda m: any(m.startswith(p) for p in ['Blue_hi', 'Blue_Green', 'HSV_Blue', 'Lab_b_Blue']),
        '투표/가중': lambda m: m.startswith('Vote_') or m.startswith('Weighted_'),
        '히스토그램': lambda m: m.startswith('BackProj_'),
    }

    for cat_name, cat_filter in categories.items():
        cat_methods = [(m, s) for m, s in sorted_avg if cat_filter(m)]
        if cat_methods:
            best = cat_methods[0]
            print(f"\n  [{cat_name}]")
            print(f"    최적: {best[0]} (avg IoU={best[1]['avg_iou']:.4f}, "
                  f"std={best[1]['std_iou']:.4f})")
            if len(cat_methods) > 1:
                second = cat_methods[1]
                print(f"    차선: {second[0]} (avg IoU={second[1]['avg_iou']:.4f})")

    # ============================================================
    # 결과 저장
    # ============================================================
    # JSON 저장
    json_results = {}
    for fname, results in all_results.items():
        json_results[fname] = {}
        for method, res in results.items():
            json_results[fname][method] = {
                'iou': float(res['iou']),
                'dice': float(res['dice']),
                'precision': float(res['precision']),
                'recall': float(res['recall']),
                'pred_area': int(res['pred_area']),
                'gt_area': int(res['gt_area']),
            }

    # 종합 순위
    rankings = []
    for method, stats in sorted_avg:
        rankings.append({
            'rank': len(rankings) + 1,
            'method': method,
            'avg_iou': float(stats['avg_iou']),
            'min_iou': float(stats['min_iou']),
            'max_iou': float(stats['max_iou']),
            'std_iou': float(stats['std_iou']),
            'per_image': {k: float(v) for k, v in stats['per_image'].items()},
        })

    output_data = {
        'summary': {
            'total_methods': len(sorted_avg),
            'images': list(all_results.keys()),
            'top_10': rankings[:10],
        },
        'rankings': rankings,
        'per_image_results': json_results,
    }

    with open(os.path.join(OUTPUT_DIR, "analysis_v4_results.json"), 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 텍스트 리포트 저장
    report_path = os.path.join(OUTPUT_DIR, "analysis_v4_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("개선된 솔더 페이스트 검출 분석 리포트 v4 (마젠타 GT)\n")
        f.write("=" * 90 + "\n\n")

        f.write("Top 30 방법 (평균 IoU 순)\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'순위':<5} {'방법':<45} {'평균IoU':<10} {'최소':<8} {'최대':<8} {'표준편차':<8}\n")
        f.write("-" * 90 + "\n")
        for i, (method, stats) in enumerate(sorted_avg[:30]):
            f.write(f"{i+1:3d}.  {method:<45s} {stats['avg_iou']:.4f}    "
                    f"{stats['min_iou']:.4f}  {stats['max_iou']:.4f}  {stats['std_iou']:.4f}\n")

        f.write("\n\n이미지별 Top 10\n")
        f.write("=" * 90 + "\n")
        for fname, results in all_results.items():
            sorted_m = sorted(results.items(), key=lambda x: x[1]['iou'], reverse=True)
            f.write(f"\n{fname}:\n")
            for i, (method, res) in enumerate(sorted_m[:10]):
                f.write(f"  {i+1:2d}. {method:<40s} IoU={res['iou']:.4f}  "
                        f"Dice={res['dice']:.4f}  P={res['precision']:.3f}  R={res['recall']:.3f}\n")

        f.write("\n\n안정성 분석 (avg IoU > 0.3, std 낮은 순)\n")
        f.write("-" * 90 + "\n")
        for i, (method, stats) in enumerate(stable_methods[:15]):
            f.write(f"{i+1:3d}.  {method:<45s} avg={stats['avg_iou']:.4f}  "
                    f"std={stats['std_iou']:.4f}  min={stats['min_iou']:.4f}\n")

    print(f"\n결과 저장: {OUTPUT_DIR}/")
    print("완료!")


if __name__ == '__main__':
    main()
