"""
종합 솔더 페이스트 검출 분석 스크립트
===========================================
모든 가능한 영상처리 방법을 시도하고, 라벨(보라색 윤곽선)과 비교하여
IoU(Intersection over Union) 점수로 정량 평가합니다.

사용법: python comprehensive_analysis.py
"""

import cv2 as cv
import numpy as np
import os
import json
import time
from collections import OrderedDict

# ============================================================
# 설정
# ============================================================
TEST_DIR = "test"
LABEL_DIR = "test_label"
OUTPUT_DIR = "analysis_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 유틸리티 함수
# ============================================================
def calc_iou(mask_pred, mask_gt):
    """IoU (Intersection over Union) 계산"""
    pred = (mask_pred > 0).astype(np.uint8)
    gt = (mask_gt > 0).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return intersection / union


def calc_dice(mask_pred, mask_gt):
    """Dice coefficient 계산"""
    pred = (mask_pred > 0).astype(np.uint8)
    gt = (mask_gt > 0).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 0.0
    return 2.0 * intersection / total


def calc_precision_recall(mask_pred, mask_gt):
    """Precision, Recall 계산"""
    pred = (mask_pred > 0).astype(np.uint8)
    gt = (mask_gt > 0).astype(np.uint8)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def clean_mask(mask, min_area=10, morph_kernel=3):
    """마스크 정리 (형태학적 연산 + 소면적 제거)"""
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    cleaned = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)

    # 소면적 컨투어 제거
    contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(cleaned)
    for cnt in contours:
        if cv.contourArea(cnt) >= min_area:
            cv.drawContours(result, [cnt], -1, 255, -1)
    return result


def extract_ground_truth(label_img, original_img):
    """
    라벨 이미지에서 Ground Truth 마스크 추출
    보라색(purple/violet) 윤곽선으로 그려진 영역을 검출

    전략:
    1. 라벨과 원본의 차이(diff)에서 보라색 윤곽선 검출
    2. 윤곽선 내부를 채워 마스크 생성
    """
    # 방법 1: 라벨-원본 차이 기반
    diff = cv.absdiff(label_img, original_img)
    diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

    # 보라색은 HSV에서 H=120~160 범위
    diff_hsv = cv.cvtColor(label_img, cv.COLOR_BGR2HSV)
    orig_hsv = cv.cvtColor(original_img, cv.COLOR_BGR2HSV)

    # 보라색 범위 (여러 범위 시도)
    purple_masks = []

    # 보라색 HSV 범위 1: 일반적인 보라색
    lower_purple1 = np.array([120, 30, 30])
    upper_purple1 = np.array([170, 255, 255])
    pm1 = cv.inRange(diff_hsv, lower_purple1, upper_purple1)
    purple_masks.append(pm1)

    # 보라색 HSV 범위 2: 밝은 보라/마젠타
    lower_purple2 = np.array([140, 50, 50])
    upper_purple2 = np.array([180, 255, 255])
    pm2 = cv.inRange(diff_hsv, lower_purple2, upper_purple2)
    purple_masks.append(pm2)

    # 보라색 HSV 범위 3: 넓은 범위
    lower_purple3 = np.array([100, 20, 20])
    upper_purple3 = np.array([180, 255, 255])
    pm3 = cv.inRange(diff_hsv, lower_purple3, upper_purple3)
    purple_masks.append(pm3)

    # 방법 2: 차이 이미지에서 변경된 부분 감지
    _, diff_thresh = cv.threshold(diff_gray, 15, 255, cv.THRESH_BINARY)
    purple_masks.append(diff_thresh)

    # 방법 3: BGR에서 직접 보라색 감지 (B>100, R>100, G<B, G<R)
    b, g, r = cv.split(label_img)
    purple_bgr = np.zeros_like(diff_gray)
    purple_cond = (b.astype(int) > 80) & (r.astype(int) > 80) & \
                  (g.astype(int) < b.astype(int) - 20) & \
                  (g.astype(int) < r.astype(int) - 20)

    # 원본에는 없고 라벨에만 있는 보라색 픽셀
    orig_b, orig_g, orig_r = cv.split(original_img)
    diff_significant = diff_gray > 10
    purple_bgr[purple_cond & diff_significant] = 255
    purple_masks.append(purple_bgr)

    # 모든 보라색 마스크 합치기
    combined_purple = np.zeros_like(diff_gray)
    for pm in purple_masks:
        combined_purple = cv.bitwise_or(combined_purple, pm)

    # 차이가 있는 부분으로 제한 (원본과 다른 부분에서만)
    combined_purple = cv.bitwise_and(combined_purple, diff_thresh)

    # 형태학적 연산으로 윤곽선 연결
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    closed = cv.morphologyEx(combined_purple, cv.MORPH_CLOSE, kernel_close)

    # dilate로 윤곽선 두께 확장 후 fill
    kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilated = cv.dilate(closed, kernel_dilate, iterations=2)

    # 윤곽선 찾아서 내부 채우기
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(diff_gray)
    for cnt in contours:
        if cv.contourArea(cnt) > 20:  # 노이즈 제거
            cv.drawContours(filled_mask, [cnt], -1, 255, -1)

    # 추가: closed 마스크도 fill 시도
    contours2, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours2:
        if cv.contourArea(cnt) > 20:
            cv.drawContours(filled_mask, [cnt], -1, 255, -1)

    return filled_mask, combined_purple


def save_comparison(original, gt_mask, pred_mask, method_name, filename, output_dir):
    """비교 결과 시각화 저장"""
    h, w = original.shape[:2]

    # 원본에 GT 오버레이 (초록)
    gt_overlay = original.copy()
    gt_colored = np.zeros_like(original)
    gt_colored[gt_mask > 0] = (0, 255, 0)
    gt_overlay = cv.addWeighted(gt_overlay, 0.7, gt_colored, 0.3, 0)

    # 원본에 예측 오버레이 (빨강)
    pred_overlay = original.copy()
    pred_colored = np.zeros_like(original)
    pred_colored[pred_mask > 0] = (0, 0, 255)
    pred_overlay = cv.addWeighted(pred_overlay, 0.7, pred_colored, 0.3, 0)

    # TP/FP/FN 시각화
    comparison = original.copy()
    pred_bin = (pred_mask > 0)
    gt_bin = (gt_mask > 0)
    tp = pred_bin & gt_bin
    fp = pred_bin & ~gt_bin
    fn = ~pred_bin & gt_bin

    comp_colored = np.zeros_like(original)
    comp_colored[tp] = (0, 255, 0)   # TP = 초록
    comp_colored[fp] = (0, 0, 255)   # FP = 빨강
    comp_colored[fn] = (255, 0, 0)   # FN = 파랑
    comparison = cv.addWeighted(comparison, 0.5, comp_colored, 0.5, 0)

    # 4분할 이미지 생성
    top = np.hstack([gt_overlay, pred_overlay])
    # 마스크를 3채널로
    gt_3ch = cv.cvtColor(gt_mask, cv.COLOR_GRAY2BGR)
    pred_3ch = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
    bottom = np.hstack([gt_3ch, comparison])

    combined = np.vstack([top, bottom])

    # 텍스트 추가
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(combined, "GT (green)", (5, 20), font, 0.5, (0, 255, 0), 1)
    cv.putText(combined, f"Pred: {method_name}", (w + 5, 20), font, 0.5, (0, 0, 255), 1)
    cv.putText(combined, "GT Mask", (5, h + 20), font, 0.5, (255, 255, 255), 1)
    cv.putText(combined, "TP/FP/FN", (w + 5, h + 20), font, 0.5, (255, 255, 255), 1)

    base = os.path.splitext(filename)[0]
    safe_method = method_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_path = os.path.join(output_dir, f"{base}_{safe_method}.png")
    cv.imwrite(out_path, combined)


# ============================================================
# 검출 방법들
# ============================================================

def method_blue_channel(img, min_val, max_val):
    """Blue 채널 임계값 (현재 방법)"""
    b = img[:, :, 0]
    return cv.inRange(b, min_val, max_val)


def method_green_channel(img, min_val, max_val):
    """Green 채널 임계값"""
    g = img[:, :, 1]
    return cv.inRange(g, min_val, max_val)


def method_red_channel(img, min_val, max_val):
    """Red 채널 임계값"""
    r = img[:, :, 2]
    return cv.inRange(r, min_val, max_val)


def method_blue_minus_red(img, threshold):
    """B-R 차이 기반: 파란색이 빨간색보다 강한 영역"""
    b = img[:, :, 0].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    diff = b - r
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[diff > threshold] = 255
    return mask


def method_blue_minus_green(img, threshold):
    """B-G 차이 기반"""
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    diff = b - g
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[diff > threshold] = 255
    return mask


def method_blue_ratio(img, threshold):
    """B/(B+G+R) 비율 기반"""
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    total = b + g + r + 1e-6
    ratio = b / total
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[ratio > threshold] = 255
    return mask


def method_br_ratio(img, threshold):
    """B/(B+R) 비율 기반 (Green 무시)"""
    b = img[:, :, 0].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    total = b + r + 1e-6
    ratio = b / total
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[ratio > threshold] = 255
    return mask


def method_hsv_hue(img, h_min, h_max, s_min=0, v_min=0):
    """HSV Hue 범위 기반"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, 255, 255])
    return cv.inRange(hsv, lower, upper)


def method_hsv_saturation(img, s_min, s_max):
    """HSV Saturation 범위 기반"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, s_min, 0])
    upper = np.array([180, s_max, 255])
    return cv.inRange(hsv, lower, upper)


def method_hsv_value(img, v_min, v_max):
    """HSV Value (밝기) 범위 기반"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, v_min])
    upper = np.array([180, 255, v_max])
    return cv.inRange(hsv, lower, upper)


def method_hsv_combined(img, h_min, h_max, s_min, s_max, v_min, v_max):
    """HSV 결합 범위"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    return cv.inRange(hsv, lower, upper)


def method_lab_a(img, a_min, a_max):
    """Lab 색상공간 - a 채널 (green-red)"""
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    a = lab[:, :, 1]
    return cv.inRange(a, a_min, a_max)


def method_lab_b(img, b_min, b_max):
    """Lab 색상공간 - b 채널 (blue-yellow)"""
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    b_ch = lab[:, :, 2]
    return cv.inRange(b_ch, b_min, b_max)


def method_lab_l(img, l_min, l_max):
    """Lab 색상공간 - L 채널 (밝기)"""
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    l = lab[:, :, 0]
    return cv.inRange(l, l_min, l_max)


def method_lab_combined(img, l_min, l_max, a_min, a_max, b_min, b_max):
    """Lab 결합"""
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    lower = np.array([l_min, a_min, b_min])
    upper = np.array([l_max, a_max, b_max])
    return cv.inRange(lab, lower, upper)


def method_ycrcb_cb(img, cb_min, cb_max):
    """YCrCb - Cb 채널 (blue difference)"""
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    cb = ycrcb[:, :, 2]
    return cv.inRange(cb, cb_min, cb_max)


def method_ycrcb_cr(img, cr_min, cr_max):
    """YCrCb - Cr 채널 (red difference)"""
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    return cv.inRange(cr, cr_min, cr_max)


def method_otsu_blue(img):
    """Otsu 자동 임계값 - Blue 채널"""
    b = img[:, :, 0]
    blurred = cv.GaussianBlur(b, (5, 5), 0)
    _, mask = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return mask


def method_otsu_gray(img):
    """Otsu 자동 임계값 - Grayscale"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return mask


def method_otsu_inv_gray(img):
    """Otsu 자동 임계값 - Grayscale 반전 (어두운 영역 검출)"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return mask


def method_otsu_saturation(img):
    """Otsu - HSV Saturation 채널"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    blurred = cv.GaussianBlur(s, (5, 5), 0)
    _, mask = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return mask


def method_adaptive_blue(img, block_size=11, c=2):
    """적응형 임계값 - Blue 채널"""
    b = img[:, :, 0]
    blurred = cv.GaussianBlur(b, (5, 5), 0)
    mask = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv.THRESH_BINARY, block_size, c)
    return mask


def method_adaptive_gray(img, block_size=11, c=2):
    """적응형 임계값 - Grayscale"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    mask = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv.THRESH_BINARY, block_size, c)
    return mask


def method_adaptive_inv_gray(img, block_size=11, c=2):
    """적응형 임계값 반전 - Grayscale (어두운 영역)"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    mask = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv.THRESH_BINARY_INV, block_size, c)
    return mask


def method_clahe_otsu(img, clip_limit=2.0):
    """CLAHE 대비 향상 + Otsu"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, mask = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return mask


def method_clahe_blue_otsu(img, clip_limit=2.0):
    """CLAHE 대비 향상 (Blue채널) + Otsu"""
    b = img[:, :, 0]
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(b)
    _, mask = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return mask


def method_canny_fill(img, low=50, high=150):
    """Canny 에지 검출 + 내부 채우기"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, low, high)

    # 에지 연결
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dilated = cv.dilate(edges, kernel, iterations=2)

    # 내부 채우기
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for cnt in contours:
        if cv.contourArea(cnt) > 50:
            cv.drawContours(mask, [cnt], -1, 255, -1)
    return mask


def method_canny_blue_fill(img, low=30, high=100):
    """Canny 에지 (Blue 채널) + 내부 채우기"""
    b = img[:, :, 0]
    blurred = cv.GaussianBlur(b, (5, 5), 0)
    edges = cv.Canny(blurred, low, high)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dilated = cv.dilate(edges, kernel, iterations=2)

    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(b)
    for cnt in contours:
        if cv.contourArea(cnt) > 50:
            cv.drawContours(mask, [cnt], -1, 255, -1)
    return mask


def method_gradient_magnitude(img, threshold=30):
    """그래디언트 크기 기반"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))

    _, mask = cv.threshold(magnitude, threshold, 255, cv.THRESH_BINARY)

    # 내부 채우기
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    for cnt in contours:
        if cv.contourArea(cnt) > 50:
            cv.drawContours(result, [cnt], -1, 255, -1)
    return result


def method_kmeans(img, k=2, target_cluster='darkest'):
    """K-means 클러스터링"""
    # BGR을 float32로
    pixel_values = img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv.kmeans(pixel_values, k, None, criteria, 5, cv.KMEANS_PP_CENTERS)

    labels = labels.reshape(img.shape[:2])
    centers = np.uint8(centers)

    # 타겟 클러스터 선택
    if target_cluster == 'darkest':
        # 가장 어두운 클러스터
        brightness = [np.mean(c) for c in centers]
        target_idx = np.argmin(brightness)
    elif target_cluster == 'bluest':
        # Blue 값이 가장 높은 클러스터
        blue_vals = [c[0] for c in centers]
        target_idx = np.argmax(blue_vals)
    elif target_cluster == 'blue_dominant':
        # B/(B+G+R) 비율이 가장 높은 클러스터
        ratios = [c[0] / (sum(c) + 1e-6) for c in centers]
        target_idx = np.argmax(ratios)
    else:
        target_idx = 0

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[labels == target_idx] = 255
    return mask


def method_kmeans_hsv(img, k=3, target='high_sat'):
    """K-means (HSV 공간)"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    pixel_values = hsv.reshape((-1, 3)).astype(np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv.kmeans(pixel_values, k, None, criteria, 5, cv.KMEANS_PP_CENTERS)

    labels = labels.reshape(img.shape[:2])

    if target == 'high_sat':
        # 채도가 가장 높은 클러스터
        sat_vals = [c[1] for c in centers]
        target_idx = np.argmax(sat_vals)
    elif target == 'blue_hue':
        # Hue가 blue 범위 (100-130)에 가장 가까운 클러스터
        blue_dist = [abs(c[0] - 115) for c in centers]
        target_idx = np.argmin(blue_dist)
    elif target == 'darkest':
        val_vals = [c[2] for c in centers]
        target_idx = np.argmin(val_vals)
    else:
        target_idx = 0

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[labels == target_idx] = 255
    return mask


def method_color_distance(img, ref_color_bgr, threshold=50):
    """기준 색상까지의 거리 기반 검출"""
    ref = np.array(ref_color_bgr, dtype=np.float32)
    diff = np.sqrt(np.sum((img.astype(np.float32) - ref) ** 2, axis=2))
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[diff < threshold] = 255
    return mask


def method_color_distance_lab(img, ref_color_bgr, threshold=30):
    """Lab 색상공간에서 기준 색상까지의 거리"""
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab).astype(np.float32)
    ref_bgr = np.uint8([[ref_color_bgr]])
    ref_lab = cv.cvtColor(ref_bgr, cv.COLOR_BGR2Lab).astype(np.float32)[0, 0]

    diff = np.sqrt(np.sum((lab - ref_lab) ** 2, axis=2))
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[diff < threshold] = 255
    return mask


def method_background_subtract(img, bg_color_bgr, threshold=40):
    """배경색 제거: 배경에서 멀리 떨어진 픽셀 검출"""
    ref = np.array(bg_color_bgr, dtype=np.float32)
    diff = np.sqrt(np.sum((img.astype(np.float32) - ref) ** 2, axis=2))
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[diff > threshold] = 255
    return mask


def method_watershed(img):
    """Watershed 세그멘테이션"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Otsu로 초기 전경/배경 분리
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # 확실한 배경
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # 확실한 전경
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 미확인 영역
    unknown = cv.subtract(sure_bg, sure_fg)

    # 마커
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = cv.watershed(img, markers)

    # 마커에서 마스크 생성 (경계=-1 제외, 배경=1 제외)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[markers > 1] = 255
    return mask


def method_watershed_blue(img):
    """Watershed (Blue 채널 기반)"""
    b = img[:, :, 0]
    blurred = cv.GaussianBlur(b, (5, 5), 0)

    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[markers > 1] = 255
    return mask


def method_grabcut(img, iterations=5):
    """GrabCut 세그멘테이션"""
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    h, w = img.shape[:2]
    # 중심 영역을 초기 전경으로 설정
    margin_x = max(w // 6, 2)
    margin_y = max(h // 6, 2)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    try:
        cv.grabCut(img, mask, rect, bgd_model, fgd_model, iterations, cv.GC_INIT_WITH_RECT)
        result_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        result_mask[(mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD)] = 255
        return result_mask
    except Exception:
        return np.zeros(img.shape[:2], dtype=np.uint8)


def method_grabcut_blue_init(img, iterations=5):
    """GrabCut (Blue 채널 임계값으로 초기화)"""
    b = img[:, :, 0]
    _, init_mask = cv.threshold(b, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    gc_mask = np.zeros(img.shape[:2], np.uint8)
    gc_mask[:] = cv.GC_BGD
    gc_mask[init_mask > 0] = cv.GC_PR_FGD

    # 확실한 전경: Blue 값이 매우 높은 영역
    gc_mask[b > 200] = cv.GC_FGD
    # 확실한 배경: Blue 값이 매우 낮은 영역
    gc_mask[b < 30] = cv.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv.grabCut(img, gc_mask, None, bgd_model, fgd_model, iterations, cv.GC_INIT_WITH_MASK)
        result_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        result_mask[(gc_mask == cv.GC_FGD) | (gc_mask == cv.GC_PR_FGD)] = 255
        return result_mask
    except Exception:
        return np.zeros(img.shape[:2], dtype=np.uint8)


def method_mean_shift(img, sp=10, sr=30):
    """Mean Shift 필터링 + Otsu"""
    shifted = cv.pyrMeanShiftFiltering(img, sp, sr)
    gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return mask


def method_local_variance(img, ksize=15, threshold=500):
    """로컬 분산 (텍스처) 기반"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float64)

    # 로컬 평균
    mean = cv.blur(gray, (ksize, ksize))
    # 로컬 제곱 평균
    sqr_mean = cv.blur(gray ** 2, (ksize, ksize))
    # 분산
    variance = sqr_mean - mean ** 2

    norm_var = np.uint8(np.clip(variance / variance.max() * 255, 0, 255))
    _, mask = cv.threshold(norm_var, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return mask


def method_multi_channel_vote(img, blue_thresh=100, green_thresh_max=120, red_thresh_max=100):
    """멀티채널 투표: B>thresh AND G<thresh AND R<thresh"""
    b = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    r = img[:, :, 2].astype(int)

    cond = (b > blue_thresh) & (g < green_thresh_max) & (r < red_thresh_max)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[cond] = 255
    return mask


def method_multi_channel_weighted(img, wb=1.0, wg=-0.5, wr=-0.5, threshold=50):
    """가중 멀티채널 조합"""
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)

    combined = wb * b + wg * g + wr * r
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[combined > threshold] = 255
    return mask


def method_hsv_blue_range(img):
    """HSV에서 파란색 영역만 검출"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # 파란색 범위
    lower = np.array([90, 50, 30])
    upper = np.array([140, 255, 255])
    return cv.inRange(hsv, lower, upper)


def method_hsv_dark_blue(img):
    """HSV에서 어두운 파란색/남색 영역"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([90, 30, 10])
    upper = np.array([145, 255, 180])
    return cv.inRange(hsv, lower, upper)


def method_hsv_blue_green(img):
    """HSV에서 파란색~초록색 영역 (솔더페이스트의 중~고 높이)"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([60, 30, 20])
    upper = np.array([140, 255, 255])
    return cv.inRange(hsv, lower, upper)


def method_hsv_not_red_orange(img):
    """HSV에서 빨강/주황을 제외한 모든 영역"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # 빨강/주황 범위
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(red_mask1, red_mask2)

    # 반전: 빨강이 아닌 영역
    return cv.bitwise_not(red_mask)


def method_luv_u(img, u_min, u_max):
    """Luv 색상공간 - u 채널"""
    luv = cv.cvtColor(img, cv.COLOR_BGR2Luv)
    u = luv[:, :, 1]
    return cv.inRange(u, u_min, u_max)


def method_luv_v(img, v_min, v_max):
    """Luv 색상공간 - v 채널"""
    luv = cv.cvtColor(img, cv.COLOR_BGR2Luv)
    v = luv[:, :, 2]
    return cv.inRange(v, v_min, v_max)


def method_multi_colorspace(img, use_lab=True, use_hsv=True, use_ycrcb=True):
    """다중 색상공간 결합 (AND/OR 투표)"""
    masks = []

    if use_hsv:
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # 파란색 계열
        m_hsv = cv.inRange(hsv, np.array([80, 30, 20]), np.array([140, 255, 255]))
        masks.append(m_hsv)

    if use_lab:
        lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
        # b채널이 낮은 영역 (파란색 쪽)
        b_ch = lab[:, :, 2]
        m_lab = np.zeros_like(b_ch)
        m_lab[b_ch < 128] = 255
        masks.append(m_lab)

    if use_ycrcb:
        ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        cb = ycrcb[:, :, 2]
        m_ycrcb = np.zeros_like(cb)
        m_ycrcb[cb > 128] = 255
        masks.append(m_ycrcb)

    if not masks:
        return np.zeros(img.shape[:2], dtype=np.uint8)

    # 과반수 투표
    vote = np.zeros(img.shape[:2], dtype=np.float32)
    for m in masks:
        vote += (m > 0).astype(np.float32)

    result = np.zeros(img.shape[:2], dtype=np.uint8)
    result[vote >= len(masks) / 2] = 255
    return result


def method_bilateral_otsu(img, d=9, sigma_color=75, sigma_space=75):
    """Bilateral 필터 (에지 보존 스무딩) + Otsu"""
    filtered = cv.bilateralFilter(img, d, sigma_color, sigma_space)
    gray = cv.cvtColor(filtered, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return mask


def method_superpixel_like(img, region_size=20, ruler=10.0):
    """SLIC 유사 방법: 작은 영역별 평균 색상 기반"""
    # blur로 초과세그먼트 모방
    blurred = cv.GaussianBlur(img, (region_size | 1, region_size | 1), 0)
    # Blue 우세 영역
    b = blurred[:, :, 0].astype(float)
    g = blurred[:, :, 1].astype(float)
    r = blurred[:, :, 2].astype(float)
    total = b + g + r + 1e-6
    blue_ratio = b / total
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[blue_ratio > 0.4] = 255
    return mask


def method_floodfill_from_blue(img, seed_threshold=180):
    """Blue 채널 고값 시드에서 FloodFill"""
    b = img[:, :, 0]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 시드 포인트: Blue 채널이 높은 위치
    seeds = np.where(b > seed_threshold)
    if len(seeds[0]) == 0:
        # 임계값을 낮춰서 재시도
        seeds = np.where(b > np.percentile(b, 80))
    if len(seeds[0]) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # 중앙에 가장 가까운 시드 선택
    center_y, center_x = h // 2, w // 2
    distances = (seeds[0] - center_y) ** 2 + (seeds[1] - center_x) ** 2
    best_idx = np.argmin(distances)
    seed_point = (int(seeds[1][best_idx]), int(seeds[0][best_idx]))

    # FloodFill
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    flood_img = gray.copy()

    lo_diff = 20
    up_diff = 20
    cv.floodFill(flood_img, mask_ff, seed_point, 255,
                 loDiff=(lo_diff,), upDiff=(up_diff,),
                 flags=cv.FLOODFILL_MASK_ONLY | (255 << 8))

    result = mask_ff[1:-1, 1:-1]
    return result


def method_connected_components_blue(img, threshold=80):
    """Blue 채널 임계값 + Connected Components 필터링"""
    b = img[:, :, 0]
    _, binary = cv.threshold(b, threshold, 255, cv.THRESH_BINARY)

    # Connected Components
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary)

    # 가장 큰 연결 요소 선택 (배경=0 제외)
    mask = np.zeros_like(binary)
    if num_labels > 1:
        # 배경(0) 제외한 최대 영역
        areas = stats[1:, cv.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        mask[labels == largest_label] = 255
    return mask


def method_heatmap_peak(img):
    """Blue 채널을 히트맵으로 해석하고 피크 영역 검출"""
    b = img[:, :, 0].astype(np.float32)

    # 가우시안 블러로 스무딩
    smoothed = cv.GaussianBlur(b, (7, 7), 0)

    # 피크 = 로컬 최대값 영역
    # dilate와 비교하여 로컬 최대값 찾기
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    dilated = cv.dilate(smoothed, kernel)

    # 피크 주변 영역: dilated와의 차이가 작은 영역
    diff = dilated - smoothed
    peak_region = np.zeros(img.shape[:2], dtype=np.uint8)
    peak_region[diff < 10] = 255

    # Blue값이 평균 이상인 영역으로 제한
    mean_b = np.mean(b)
    peak_region[b < mean_b] = 0

    return peak_region


# ============================================================
# 채널별 분포 분석
# ============================================================
def analyze_pixel_distribution(img, name=""):
    """이미지의 채널별 픽셀 분포 분석"""
    print(f"\n{'='*60}")
    print(f"픽셀 분포 분석: {name}")
    print(f"{'='*60}")
    print(f"이미지 크기: {img.shape}")

    b, g, r = cv.split(img)

    print(f"\n[BGR 채널]")
    print(f"  B: min={b.min()}, max={b.max()}, mean={b.mean():.1f}, std={b.std():.1f}, median={np.median(b):.0f}")
    print(f"  G: min={g.min()}, max={g.max()}, mean={g.mean():.1f}, std={g.std():.1f}, median={np.median(g):.0f}")
    print(f"  R: min={r.min()}, max={r.max()}, mean={r.mean():.1f}, std={r.std():.1f}, median={np.median(r):.0f}")

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    print(f"\n[HSV 채널]")
    print(f"  H: min={h.min()}, max={h.max()}, mean={h.mean():.1f}, std={h.std():.1f}")
    print(f"  S: min={s.min()}, max={s.max()}, mean={s.mean():.1f}, std={s.std():.1f}")
    print(f"  V: min={v.min()}, max={v.max()}, mean={v.mean():.1f}, std={v.std():.1f}")

    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    l, a, b_ch = cv.split(lab)
    print(f"\n[Lab 채널]")
    print(f"  L: min={l.min()}, max={l.max()}, mean={l.mean():.1f}, std={l.std():.1f}")
    print(f"  a: min={a.min()}, max={a.max()}, mean={a.mean():.1f}, std={a.std():.1f}")
    print(f"  b: min={b_ch.min()}, max={b_ch.max()}, mean={b_ch.mean():.1f}, std={b_ch.std():.1f}")

    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb)
    print(f"\n[YCrCb 채널]")
    print(f"  Y:  min={y.min()}, max={y.max()}, mean={y.mean():.1f}")
    print(f"  Cr: min={cr.min()}, max={cr.max()}, mean={cr.mean():.1f}")
    print(f"  Cb: min={cb.min()}, max={cb.max()}, mean={cb.mean():.1f}")

    return {
        'bgr': {'b': b, 'g': g, 'r': r},
        'hsv': {'h': h, 's': s, 'v': v},
        'lab': {'l': l, 'a': a, 'b': b_ch},
        'ycrcb': {'y': y, 'cr': cr, 'cb': cb}
    }


def analyze_gt_region(img, gt_mask, name=""):
    """GT 영역 내부/외부 픽셀 분포 비교"""
    gt_bin = gt_mask > 0

    if gt_bin.sum() == 0:
        print(f"  [!] GT 마스크가 비어있음: {name}")
        return

    print(f"\n[GT 영역 내부 vs 외부 비교: {name}]")
    print(f"  GT 픽셀 수: {gt_bin.sum()} / {gt_bin.size} ({gt_bin.sum()/gt_bin.size*100:.1f}%)")

    b, g, r = cv.split(img)
    for ch_name, ch in [('B', b), ('G', g), ('R', r)]:
        inside = ch[gt_bin]
        outside = ch[~gt_bin]
        print(f"  {ch_name} 채널:")
        print(f"    내부: mean={inside.mean():.1f}, std={inside.std():.1f}, "
              f"min={inside.min()}, max={inside.max()}")
        if outside.size > 0:
            print(f"    외부: mean={outside.mean():.1f}, std={outside.std():.1f}, "
                  f"min={outside.min()}, max={outside.max()}")

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    for ch_name, ch in [('H', h), ('S', s), ('V', v)]:
        inside = ch[gt_bin]
        outside = ch[~gt_bin]
        print(f"  {ch_name} 채널:")
        print(f"    내부: mean={inside.mean():.1f}, std={inside.std():.1f}")
        if outside.size > 0:
            print(f"    외부: mean={outside.mean():.1f}, std={outside.std():.1f}")

    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    l, a, b_ch = cv.split(lab)
    for ch_name, ch in [('L', l), ('a', a), ('b_lab', b_ch)]:
        inside = ch[gt_bin]
        outside = ch[~gt_bin]
        print(f"  {ch_name} 채널:")
        print(f"    내부: mean={inside.mean():.1f}, std={inside.std():.1f}")
        if outside.size > 0:
            print(f"    외부: mean={outside.mean():.1f}, std={outside.std():.1f}")


# ============================================================
# 메인 실행
# ============================================================
def run_all_methods(img, gt_mask, filename):
    """모든 방법 실행 및 평가"""
    results = OrderedDict()

    h, w = img.shape[:2]
    total_pixels = h * w

    # 블러 적용 버전
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    blurred3 = cv.GaussianBlur(img, (3, 3), 0)
    blurred7 = cv.GaussianBlur(img, (7, 7), 0)

    # ===========================================================
    # 카테고리 1: Blue 채널 단일 임계값 (현재 방법)
    # ===========================================================
    print("\n--- 카테고리 1: Blue 채널 단일 임계값 ---")
    for min_val in [30, 50, 70, 80, 90, 100, 120, 140, 150, 160, 180]:
        name = f"Blue_ch_{min_val}-255"
        mask = method_blue_channel(img, min_val, 255)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # Blurred 버전
    for min_val in [50, 80, 100, 120, 150]:
        name = f"Blue_ch_blur_{min_val}-255"
        mask = method_blue_channel(blurred, min_val, 255)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # ===========================================================
    # 카테고리 2: Green/Red 채널 임계값
    # ===========================================================
    print("\n--- 카테고리 2: Green/Red 채널 임계값 ---")
    for min_val in [50, 80, 100, 120]:
        name = f"Green_ch_{min_val}-255"
        mask = method_green_channel(img, min_val, 255)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # Red 채널 반전 (Red가 낮은 = 솔더페이스트 가능)
    for max_val in [50, 80, 100, 120, 150]:
        name = f"Red_ch_inv_0-{max_val}"
        mask = method_red_channel(img, 0, max_val)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # ===========================================================
    # 카테고리 3: 채널 차이 / 비율 기반
    # ===========================================================
    print("\n--- 카테고리 3: 채널 차이/비율 기반 ---")
    for thresh in [-30, -10, 0, 10, 20, 30, 50, 70]:
        name = f"B-R_diff_{thresh}"
        mask = method_blue_minus_red(img, thresh)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    for thresh in [-30, -10, 0, 10, 20, 30, 50]:
        name = f"B-G_diff_{thresh}"
        mask = method_blue_minus_green(img, thresh)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    for thresh in [0.33, 0.35, 0.38, 0.40, 0.42, 0.45, 0.50, 0.55]:
        name = f"B_ratio_{thresh}"
        mask = method_blue_ratio(img, thresh)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        name = f"BR_ratio_{thresh}"
        mask = method_br_ratio(img, thresh)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # ===========================================================
    # 카테고리 4: HSV 색상 공간
    # ===========================================================
    print("\n--- 카테고리 4: HSV 색상 공간 ---")

    # Hue 범위별
    hue_ranges = [
        (80, 130, "blue"), (90, 140, "blue_wide"),
        (60, 130, "cyan_blue"), (100, 145, "deep_blue"),
        (60, 160, "blue_purple"), (0, 60, "red_yellow"),
    ]
    for h_min, h_max, label in hue_ranges:
        for s_min in [20, 50]:
            name = f"HSV_H{h_min}-{h_max}_S{s_min}+_{label}"
            mask = method_hsv_hue(img, h_min, h_max, s_min=s_min)
            mask = clean_mask(mask, min_area=10)
            iou = calc_iou(mask, gt_mask)
            dice = calc_dice(mask, gt_mask)
            prec, rec = calc_precision_recall(mask, gt_mask)
            results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
            print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # Saturation 범위
    for s_min, s_max in [(50, 255), (80, 255), (100, 255), (0, 80), (0, 50)]:
        name = f"HSV_S{s_min}-{s_max}"
        mask = method_hsv_saturation(img, s_min, s_max)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # Value 범위
    for v_min, v_max in [(0, 80), (0, 100), (0, 120), (0, 150), (50, 150), (100, 200)]:
        name = f"HSV_V{v_min}-{v_max}"
        mask = method_hsv_value(img, v_min, v_max)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # HSV 결합 (유망한 조합)
    hsv_combos = [
        (80, 140, 30, 255, 20, 200, "blue_mid"),
        (80, 140, 50, 255, 30, 180, "blue_sat"),
        (90, 130, 30, 255, 10, 150, "blue_dark"),
        (60, 140, 20, 255, 10, 255, "wide_blue"),
        (80, 145, 20, 255, 0, 120, "dark_blue"),
    ]
    for h1, h2, s1, s2, v1, v2, label in hsv_combos:
        name = f"HSV_comb_{label}_H{h1}-{h2}_S{s1}-{s2}_V{v1}-{v2}"
        mask = method_hsv_combined(img, h1, h2, s1, s2, v1, v2)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # 특수 HSV 방법들
    for method_func, name in [
        (method_hsv_blue_range, "HSV_blue_range"),
        (method_hsv_dark_blue, "HSV_dark_blue"),
        (method_hsv_blue_green, "HSV_blue_green"),
        (method_hsv_not_red_orange, "HSV_not_red_orange"),
    ]:
        mask = method_func(img)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # ===========================================================
    # 카테고리 5: Lab 색상 공간
    # ===========================================================
    print("\n--- 카테고리 5: Lab 색상 공간 ---")

    # L 채널 (밝기)
    for l_min, l_max in [(0, 50), (0, 80), (0, 100), (0, 120), (50, 150)]:
        name = f"Lab_L{l_min}-{l_max}"
        mask = method_lab_l(img, l_min, l_max)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # a 채널 (green-red, 128=중립)
    for a_min, a_max in [(0, 110), (0, 120), (0, 128), (100, 128), (128, 180)]:
        name = f"Lab_a{a_min}-{a_max}"
        mask = method_lab_a(img, a_min, a_max)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # b 채널 (blue-yellow, 128=중립, <128=파란쪽)
    for b_min, b_max in [(0, 100), (0, 110), (0, 120), (0, 128), (0, 90), (80, 128)]:
        name = f"Lab_b{b_min}-{b_max}"
        mask = method_lab_b(img, b_min, b_max)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # Lab 결합
    lab_combos = [
        (0, 120, 0, 128, 0, 120, "dark_blue"),
        (0, 100, 0, 120, 0, 110, "very_dark_blue"),
        (0, 150, 80, 140, 0, 115, "mid_blue"),
    ]
    for l1, l2, a1, a2, b1, b2, label in lab_combos:
        name = f"Lab_comb_{label}"
        mask = method_lab_combined(img, l1, l2, a1, a2, b1, b2)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # ===========================================================
    # 카테고리 6: YCrCb 색상 공간
    # ===========================================================
    print("\n--- 카테고리 6: YCrCb 색상 공간 ---")
    for cb_min, cb_max in [(128, 255), (135, 255), (140, 255), (145, 255), (150, 255)]:
        name = f"YCrCb_Cb{cb_min}-{cb_max}"
        mask = method_ycrcb_cb(img, cb_min, cb_max)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    for cr_min, cr_max in [(0, 110), (0, 120), (0, 128), (0, 100)]:
        name = f"YCrCb_Cr{cr_min}-{cr_max}"
        mask = method_ycrcb_cr(img, cr_min, cr_max)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # ===========================================================
    # 카테고리 7: Luv 색상 공간
    # ===========================================================
    print("\n--- 카테고리 7: Luv 색상 공간 ---")
    for u_min, u_max in [(0, 80), (0, 90), (0, 100), (0, 110)]:
        name = f"Luv_u{u_min}-{u_max}"
        mask = method_luv_u(img, u_min, u_max)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    for v_min, v_max in [(0, 80), (0, 90), (0, 100), (0, 110)]:
        name = f"Luv_v{v_min}-{v_max}"
        mask = method_luv_v(img, v_min, v_max)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # ===========================================================
    # 카테고리 8: Otsu 자동 임계값
    # ===========================================================
    print("\n--- 카테고리 8: Otsu 자동 임계값 ---")
    for method_func, name in [
        (method_otsu_blue, "Otsu_blue"),
        (method_otsu_gray, "Otsu_gray"),
        (method_otsu_inv_gray, "Otsu_inv_gray"),
        (method_otsu_saturation, "Otsu_saturation"),
    ]:
        mask = method_func(img)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        prec, rec = calc_precision_recall(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
        print(f"  {name}: IoU={iou:.4f} Dice={dice:.4f} P={prec:.4f} R={rec:.4f}")

    # 블러 + Otsu
    for blur_ver, blur_name in [(blurred, "blur5"), (blurred3, "blur3"), (blurred7, "blur7")]:
        mask = method_otsu_blue(blur_ver)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        results[f"Otsu_blue_{blur_name}"] = {'iou': iou, 'dice': dice, 'mask': mask,
                                              'precision': calc_precision_recall(mask, gt_mask)[0],
                                              'recall': calc_precision_recall(mask, gt_mask)[1]}
        print(f"  Otsu_blue_{blur_name}: IoU={iou:.4f} Dice={dice:.4f}")

    # ===========================================================
    # 카테고리 9: 적응형 임계값
    # ===========================================================
    print("\n--- 카테고리 9: 적응형 임계값 ---")
    for block_size in [11, 21, 31, 51]:
        for c in [2, 5, 10]:
            name = f"Adaptive_blue_bs{block_size}_c{c}"
            mask = method_adaptive_blue(img, block_size, c)
            mask = clean_mask(mask, min_area=10)
            iou = calc_iou(mask, gt_mask)
            dice = calc_dice(mask, gt_mask)
            prec, rec = calc_precision_recall(mask, gt_mask)
            results[name] = {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'mask': mask}
            print(f"  {name}: IoU={iou:.4f}")

    for block_size in [11, 21, 31]:
        for c in [2, 5]:
            name = f"Adaptive_inv_gray_bs{block_size}_c{c}"
            mask = method_adaptive_inv_gray(img, block_size, c)
            mask = clean_mask(mask, min_area=10)
            iou = calc_iou(mask, gt_mask)
            dice = calc_dice(mask, gt_mask)
            results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                              'precision': calc_precision_recall(mask, gt_mask)[0],
                              'recall': calc_precision_recall(mask, gt_mask)[1]}
            print(f"  {name}: IoU={iou:.4f}")

    # ===========================================================
    # 카테고리 10: CLAHE + Otsu
    # ===========================================================
    print("\n--- 카테고리 10: CLAHE + Otsu ---")
    for clip in [1.0, 2.0, 3.0, 5.0]:
        name = f"CLAHE_Otsu_clip{clip}"
        mask = method_clahe_otsu(img, clip)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                          'precision': calc_precision_recall(mask, gt_mask)[0],
                          'recall': calc_precision_recall(mask, gt_mask)[1]}
        print(f"  {name}: IoU={iou:.4f}")

    for clip in [1.0, 2.0, 3.0, 5.0]:
        name = f"CLAHE_blue_Otsu_clip{clip}"
        mask = method_clahe_blue_otsu(img, clip)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                          'precision': calc_precision_recall(mask, gt_mask)[0],
                          'recall': calc_precision_recall(mask, gt_mask)[1]}
        print(f"  {name}: IoU={iou:.4f}")

    # ===========================================================
    # 카테고리 11: 에지 기반
    # ===========================================================
    print("\n--- 카테고리 11: 에지 기반 ---")
    for low, high in [(30, 80), (50, 100), (50, 150), (80, 200)]:
        name = f"Canny_fill_{low}-{high}"
        mask = method_canny_fill(img, low, high)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                          'precision': calc_precision_recall(mask, gt_mask)[0],
                          'recall': calc_precision_recall(mask, gt_mask)[1]}
        print(f"  {name}: IoU={iou:.4f}")

    for low, high in [(20, 60), (30, 80), (50, 120)]:
        name = f"Canny_blue_fill_{low}-{high}"
        mask = method_canny_blue_fill(img, low, high)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                          'precision': calc_precision_recall(mask, gt_mask)[0],
                          'recall': calc_precision_recall(mask, gt_mask)[1]}
        print(f"  {name}: IoU={iou:.4f}")

    for thresh in [20, 30, 40, 50]:
        name = f"Gradient_mag_{thresh}"
        mask = method_gradient_magnitude(img, thresh)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                          'precision': calc_precision_recall(mask, gt_mask)[0],
                          'recall': calc_precision_recall(mask, gt_mask)[1]}
        print(f"  {name}: IoU={iou:.4f}")

    # ===========================================================
    # 카테고리 12: K-means 클러스터링
    # ===========================================================
    print("\n--- 카테고리 12: K-means 클러스터링 ---")
    for k in [2, 3, 4]:
        for target in ['darkest', 'bluest', 'blue_dominant']:
            name = f"KMeans_k{k}_{target}"
            mask = method_kmeans(img, k, target)
            mask = clean_mask(mask, min_area=10)
            iou = calc_iou(mask, gt_mask)
            dice = calc_dice(mask, gt_mask)
            results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                              'precision': calc_precision_recall(mask, gt_mask)[0],
                              'recall': calc_precision_recall(mask, gt_mask)[1]}
            print(f"  {name}: IoU={iou:.4f}")

    for k in [2, 3]:
        for target in ['high_sat', 'blue_hue', 'darkest']:
            name = f"KMeans_HSV_k{k}_{target}"
            mask = method_kmeans_hsv(img, k, target)
            mask = clean_mask(mask, min_area=10)
            iou = calc_iou(mask, gt_mask)
            dice = calc_dice(mask, gt_mask)
            results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                              'precision': calc_precision_recall(mask, gt_mask)[0],
                              'recall': calc_precision_recall(mask, gt_mask)[1]}
            print(f"  {name}: IoU={iou:.4f}")

    # ===========================================================
    # 카테고리 13: 색상 거리 기반
    # ===========================================================
    print("\n--- 카테고리 13: 색상 거리 기반 ---")
    # 다양한 파란색 기준
    blue_refs = [
        ([255, 0, 0], "pure_blue"),
        ([200, 50, 0], "dark_blue"),
        ([150, 80, 30], "mid_blue"),
        ([100, 50, 20], "navy"),
        ([180, 100, 50], "medium_blue"),
    ]
    for ref, label in blue_refs:
        for thresh in [40, 60, 80, 100]:
            name = f"ColorDist_{label}_t{thresh}"
            mask = method_color_distance(img, ref, thresh)
            mask = clean_mask(mask, min_area=10)
            iou = calc_iou(mask, gt_mask)
            dice = calc_dice(mask, gt_mask)
            results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                              'precision': calc_precision_recall(mask, gt_mask)[0],
                              'recall': calc_precision_recall(mask, gt_mask)[1]}
            print(f"  {name}: IoU={iou:.4f}")

    # Lab 거리
    for ref, label in blue_refs[:3]:
        for thresh in [20, 30, 40, 50]:
            name = f"LabDist_{label}_t{thresh}"
            mask = method_color_distance_lab(img, ref, thresh)
            mask = clean_mask(mask, min_area=10)
            iou = calc_iou(mask, gt_mask)
            dice = calc_dice(mask, gt_mask)
            results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                              'precision': calc_precision_recall(mask, gt_mask)[0],
                              'recall': calc_precision_recall(mask, gt_mask)[1]}
            print(f"  {name}: IoU={iou:.4f}")

    # 배경 제거
    print("\n--- 배경색 제거 ---")
    # 코너 픽셀을 배경으로 추정
    corners = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
    avg_corner = np.mean(corners, axis=0).astype(int).tolist()
    for thresh in [30, 40, 50, 60, 80]:
        name = f"BgSubtract_corner_t{thresh}"
        mask = method_background_subtract(img, avg_corner, thresh)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                          'precision': calc_precision_recall(mask, gt_mask)[0],
                          'recall': calc_precision_recall(mask, gt_mask)[1]}
        print(f"  {name}: IoU={iou:.4f}")

    # ===========================================================
    # 카테고리 14: Watershed / GrabCut
    # ===========================================================
    print("\n--- 카테고리 14: Watershed / GrabCut ---")
    for method_func, name in [
        (method_watershed, "Watershed"),
        (method_watershed_blue, "Watershed_blue"),
        (method_grabcut, "GrabCut"),
        (method_grabcut_blue_init, "GrabCut_blue_init"),
    ]:
        try:
            mask = method_func(img)
            mask = clean_mask(mask, min_area=10)
            iou = calc_iou(mask, gt_mask)
            dice = calc_dice(mask, gt_mask)
            results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                              'precision': calc_precision_recall(mask, gt_mask)[0],
                              'recall': calc_precision_recall(mask, gt_mask)[1]}
            print(f"  {name}: IoU={iou:.4f}")
        except Exception as e:
            print(f"  {name}: 실패 - {e}")

    # ===========================================================
    # 카테고리 15: 기타 고급 방법
    # ===========================================================
    print("\n--- 카테고리 15: 기타 고급 방법 ---")

    # Mean shift
    for sp, sr in [(10, 20), (10, 30), (15, 40), (20, 50)]:
        name = f"MeanShift_sp{sp}_sr{sr}"
        try:
            mask = method_mean_shift(img, sp, sr)
            mask = clean_mask(mask, min_area=10)
            iou = calc_iou(mask, gt_mask)
            dice = calc_dice(mask, gt_mask)
            results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                              'precision': calc_precision_recall(mask, gt_mask)[0],
                              'recall': calc_precision_recall(mask, gt_mask)[1]}
            print(f"  {name}: IoU={iou:.4f}")
        except Exception as e:
            print(f"  {name}: 실패 - {e}")

    # Local variance
    for ksize in [9, 15, 21]:
        name = f"LocalVar_k{ksize}"
        mask = method_local_variance(img, ksize)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                          'precision': calc_precision_recall(mask, gt_mask)[0],
                          'recall': calc_precision_recall(mask, gt_mask)[1]}
        print(f"  {name}: IoU={iou:.4f}")

    # Bilateral + Otsu
    name = "Bilateral_Otsu"
    mask = method_bilateral_otsu(img)
    mask = clean_mask(mask, min_area=10)
    iou = calc_iou(mask, gt_mask)
    dice = calc_dice(mask, gt_mask)
    results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                      'precision': calc_precision_recall(mask, gt_mask)[0],
                      'recall': calc_precision_recall(mask, gt_mask)[1]}
    print(f"  {name}: IoU={iou:.4f}")

    # Superpixel-like
    name = "Superpixel_like"
    mask = method_superpixel_like(img)
    mask = clean_mask(mask, min_area=10)
    iou = calc_iou(mask, gt_mask)
    dice = calc_dice(mask, gt_mask)
    results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                      'precision': calc_precision_recall(mask, gt_mask)[0],
                      'recall': calc_precision_recall(mask, gt_mask)[1]}
    print(f"  {name}: IoU={iou:.4f}")

    # FloodFill
    name = "FloodFill_blue"
    mask = method_floodfill_from_blue(img)
    mask = clean_mask(mask, min_area=10)
    iou = calc_iou(mask, gt_mask)
    dice = calc_dice(mask, gt_mask)
    results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                      'precision': calc_precision_recall(mask, gt_mask)[0],
                      'recall': calc_precision_recall(mask, gt_mask)[1]}
    print(f"  {name}: IoU={iou:.4f}")

    # Connected Components
    for thresh in [60, 80, 100, 120]:
        name = f"ConnComp_blue_t{thresh}"
        mask = method_connected_components_blue(img, thresh)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                          'precision': calc_precision_recall(mask, gt_mask)[0],
                          'recall': calc_precision_recall(mask, gt_mask)[1]}
        print(f"  {name}: IoU={iou:.4f}")

    # Heatmap peak
    name = "Heatmap_peak"
    mask = method_heatmap_peak(img)
    mask = clean_mask(mask, min_area=10)
    iou = calc_iou(mask, gt_mask)
    dice = calc_dice(mask, gt_mask)
    results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                      'precision': calc_precision_recall(mask, gt_mask)[0],
                      'recall': calc_precision_recall(mask, gt_mask)[1]}
    print(f"  {name}: IoU={iou:.4f}")

    # ===========================================================
    # 카테고리 16: 멀티채널 조합
    # ===========================================================
    print("\n--- 카테고리 16: 멀티채널 조합 ---")
    # 투표 방식
    mc_combos = [
        (80, 150, 120, "B80_Gmax150_Rmax120"),
        (100, 120, 100, "B100_Gmax120_Rmax100"),
        (60, 180, 150, "B60_Gmax180_Rmax150"),
        (50, 200, 180, "B50_Gmax200_Rmax180"),
    ]
    for blue_t, green_max, red_max, label in mc_combos:
        name = f"MultiCh_vote_{label}"
        mask = method_multi_channel_vote(img, blue_t, green_max, red_max)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                          'precision': calc_precision_recall(mask, gt_mask)[0],
                          'recall': calc_precision_recall(mask, gt_mask)[1]}
        print(f"  {name}: IoU={iou:.4f}")

    # 가중 조합
    weight_combos = [
        (1.0, -0.5, -0.5, 30, "B-0.5G-0.5R_t30"),
        (1.0, -0.5, -0.5, 50, "B-0.5G-0.5R_t50"),
        (1.0, -0.3, -0.7, 30, "B-0.3G-0.7R_t30"),
        (1.5, -0.5, -1.0, 30, "1.5B-0.5G-1R_t30"),
        (2.0, -1.0, -1.0, 0, "2B-G-R_t0"),
        (1.0, 0, -1.0, 0, "B-R_t0"),
        (1.0, -1.0, 0, 0, "B-G_t0"),
    ]
    for wb, wg, wr, t, label in weight_combos:
        name = f"MultiCh_weight_{label}"
        mask = method_multi_channel_weighted(img, wb, wg, wr, t)
        mask = clean_mask(mask, min_area=10)
        iou = calc_iou(mask, gt_mask)
        dice = calc_dice(mask, gt_mask)
        results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                          'precision': calc_precision_recall(mask, gt_mask)[0],
                          'recall': calc_precision_recall(mask, gt_mask)[1]}
        print(f"  {name}: IoU={iou:.4f}")

    # 다중 색상공간 투표
    name = "Multi_colorspace_vote"
    mask = method_multi_colorspace(img)
    mask = clean_mask(mask, min_area=10)
    iou = calc_iou(mask, gt_mask)
    dice = calc_dice(mask, gt_mask)
    results[name] = {'iou': iou, 'dice': dice, 'mask': mask,
                      'precision': calc_precision_recall(mask, gt_mask)[0],
                      'recall': calc_precision_recall(mask, gt_mask)[1]}
    print(f"  {name}: IoU={iou:.4f}")

    return results


# ============================================================
# 종합 결과 리포트
# ============================================================
def generate_report(all_results, output_dir):
    """종합 결과 리포트 생성"""
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("종합 솔더 페이스트 검출 분석 리포트")
    report_lines.append("=" * 100)
    report_lines.append("")

    # 이미지별 결과
    for filename, results in all_results.items():
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"이미지: {filename}")
        report_lines.append(f"{'='*80}")

        # IoU 기준 정렬
        sorted_results = sorted(results.items(), key=lambda x: x[1]['iou'], reverse=True)

        report_lines.append(f"\n{'순위':<5} {'방법':<55} {'IoU':<10} {'Dice':<10} {'Precision':<10} {'Recall':<10}")
        report_lines.append("-" * 100)

        for rank, (name, metrics) in enumerate(sorted_results[:50], 1):
            iou = metrics['iou']
            dice = metrics['dice']
            prec = metrics.get('precision', 0)
            rec = metrics.get('recall', 0)
            report_lines.append(f"{rank:<5} {name:<55} {iou:<10.4f} {dice:<10.4f} {prec:<10.4f} {rec:<10.4f}")

    # 전체 이미지 평균 성능
    report_lines.append(f"\n\n{'='*80}")
    report_lines.append("전체 이미지 평균 성능 (IoU 기준)")
    report_lines.append(f"{'='*80}")

    # 모든 방법의 평균 IoU 계산
    all_methods = set()
    for results in all_results.values():
        all_methods.update(results.keys())

    avg_scores = {}
    for method in all_methods:
        ious = []
        for results in all_results.values():
            if method in results:
                ious.append(results[method]['iou'])
        if ious:
            avg_scores[method] = np.mean(ious)

    sorted_avg = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    report_lines.append(f"\n{'순위':<5} {'방법':<55} {'평균 IoU':<10}")
    report_lines.append("-" * 70)

    for rank, (name, avg_iou) in enumerate(sorted_avg[:30], 1):
        marker = " ***" if rank <= 3 else ""
        report_lines.append(f"{rank:<5} {name:<55} {avg_iou:<10.4f}{marker}")

    report = "\n".join(report_lines)

    # 파일 저장
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n리포트 저장: {report_path}")

    # JSON 저장 (마스크 제외)
    json_results = {}
    for filename, results in all_results.items():
        json_results[filename] = {}
        for name, metrics in results.items():
            json_results[filename][name] = {
                'iou': float(metrics['iou']),
                'dice': float(metrics['dice']),
                'precision': float(metrics.get('precision', 0)),
                'recall': float(metrics.get('recall', 0))
            }

    json_path = os.path.join(output_dir, "analysis_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"JSON 저장: {json_path}")

    return sorted_avg


# ============================================================
# 메인
# ============================================================
def main():
    start_time = time.time()

    print("=" * 60)
    print("종합 솔더 페이스트 검출 분석")
    print("=" * 60)

    # 테스트 이미지 목록
    test_files = sorted(os.listdir(TEST_DIR))
    label_files = sorted(os.listdir(LABEL_DIR))

    print(f"테스트 이미지: {test_files}")
    print(f"라벨 이미지: {label_files}")

    all_results = OrderedDict()

    for test_file in test_files:
        if test_file not in label_files:
            print(f"  [!] 라벨 없음: {test_file}")
            continue

        test_path = os.path.join(TEST_DIR, test_file)
        label_path = os.path.join(LABEL_DIR, test_file)

        # 이미지 로드
        img = cv.imread(test_path)
        label_img = cv.imread(label_path)

        if img is None or label_img is None:
            print(f"  [!] 로드 실패: {test_file}")
            continue

        # 크기 맞추기
        if img.shape != label_img.shape:
            label_img = cv.resize(label_img, (img.shape[1], img.shape[0]))

        print(f"\n\n{'#'*60}")
        print(f"이미지: {test_file} (크기: {img.shape})")
        print(f"{'#'*60}")

        # 1. 픽셀 분포 분석
        channels = analyze_pixel_distribution(img, test_file)

        # 2. GT 마스크 추출
        gt_mask, purple_outline = extract_ground_truth(label_img, img)
        gt_pixels = (gt_mask > 0).sum()
        total_pixels = gt_mask.shape[0] * gt_mask.shape[1]
        print(f"\nGT 마스크: {gt_pixels} 픽셀 ({gt_pixels/total_pixels*100:.1f}%)")

        # GT가 비어있으면 경고
        if gt_pixels == 0:
            print("  [!] GT 마스크가 비어있습니다. 보라색 윤곽선 검출 실패.")
            print("  [!] 대체 방법: 차이 이미지 기반 GT 추정")

            # 대체: 원본과 라벨의 차이가 큰 영역
            diff = cv.absdiff(label_img, img)
            diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
            _, gt_mask = cv.threshold(diff_gray, 10, 255, cv.THRESH_BINARY)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            gt_mask = cv.morphologyEx(gt_mask, cv.MORPH_CLOSE, kernel, iterations=3)

            # 재채우기
            contours, _ = cv.findContours(gt_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            gt_mask = np.zeros_like(gt_mask)
            for cnt in contours:
                if cv.contourArea(cnt) > 20:
                    cv.drawContours(gt_mask, [cnt], -1, 255, -1)

            gt_pixels = (gt_mask > 0).sum()
            print(f"  대체 GT 마스크: {gt_pixels} 픽셀 ({gt_pixels/total_pixels*100:.1f}%)")

        # GT 영역 분석
        analyze_gt_region(img, gt_mask, test_file)

        # GT 마스크 저장
        img_dir = os.path.join(OUTPUT_DIR, os.path.splitext(test_file)[0])
        os.makedirs(img_dir, exist_ok=True)
        cv.imwrite(os.path.join(img_dir, "gt_mask.png"), gt_mask)
        cv.imwrite(os.path.join(img_dir, "purple_outline.png"), purple_outline)

        # GT 오버레이 저장
        gt_overlay = img.copy()
        gt_colored = np.zeros_like(img)
        gt_colored[gt_mask > 0] = (0, 255, 0)
        gt_overlay = cv.addWeighted(gt_overlay, 0.7, gt_colored, 0.3, 0)
        cv.imwrite(os.path.join(img_dir, "gt_overlay.png"), gt_overlay)

        # 3. 모든 방법 실행
        results = run_all_methods(img, gt_mask, test_file)
        all_results[test_file] = results

        # 4. Top 10 결과 시각화 저장
        sorted_results = sorted(results.items(), key=lambda x: x[1]['iou'], reverse=True)
        for rank, (name, metrics) in enumerate(sorted_results[:10], 1):
            save_comparison(img, gt_mask, metrics['mask'], f"#{rank}_{name}", test_file, img_dir)

        # 최고 결과 마스크 저장
        if sorted_results:
            best_name, best_metrics = sorted_results[0]
            cv.imwrite(os.path.join(img_dir, "best_mask.png"), best_metrics['mask'])
            print(f"\n  최고 결과: {best_name} (IoU={best_metrics['iou']:.4f})")

    # 5. 종합 리포트
    sorted_avg = generate_report(all_results, OUTPUT_DIR)

    # 6. 요약 출력
    elapsed = time.time() - start_time
    print(f"\n\n{'='*60}")
    print(f"분석 완료! 소요 시간: {elapsed:.1f}초")
    print(f"결과 저장 위치: {OUTPUT_DIR}/")
    print(f"{'='*60}")

    print(f"\n*** 전체 이미지 평균 Top 10 ***")
    for rank, (name, avg_iou) in enumerate(sorted_avg[:10], 1):
        print(f"  #{rank}: {name} (평균 IoU={avg_iou:.4f})")


if __name__ == '__main__':
    main()
