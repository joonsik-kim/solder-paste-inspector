"""
솔더 페이스트 검출 엔진
AOI RGB 조명 이미지 기반 솔더 페이스트 영역 검출

v4: Normalized RGB + 노이즈 제거 기반 검출 개선
- AOI 이미지 색상 혼합 문제 분석 반영
  - 이미지 80%가 혼합색, 파란 영역의 77-93%가 G/R 오염
  - 로컬 노이즈 std=50, 단일 채널 임계값은 본질적 한계
- 산업용 AOI 표준 Normalized RGB(색상비) 검출 도입
- Bilateral/NLM 노이즈 제거 전처리 도입
- 색상 그라데이션 패턴 분석 추가
"""

import cv2 as cv
import numpy as np
from image_processor import (create_mask_from_hsv, apply_morphology,
                              apply_bilateral_filter, apply_nlm_denoise,
                              normalize_rgb)


def detect_solder_paste(img, config):
    """
    솔더 페이스트 영역 검출 메인 함수

    3D 높이 맵 모드: 분석 결과 기반 최적 검출 (Blue Otsu + CLAHE 앙상블)
    2D 색상 모드: HSV 색상 범위 기반 검출

    Args:
        img (numpy.ndarray): BGR 또는 HSV 형식 이미지
        config: 설정 객체 (Config 클래스)

    Returns:
        tuple: (검출된 윤곽선 리스트, 이진 마스크)
    """
    if config.HEIGHT_MAP_MODE:
        # 3D 높이 맵 모드: 분석 결과 기반 최적 검출
        detection_method = getattr(config, 'DETECTION_METHOD', 'ensemble')

        if detection_method == 'otsu_blue':
            mask = detect_otsu_blue_inv(img)
        elif detection_method == 'clahe_blue':
            mask = detect_clahe_blue_inv(img)
        elif detection_method == 'adaptive':
            mask = detect_adaptive_blue(img)
        elif detection_method == 'ensemble':
            mask = detect_ensemble(img)
        elif detection_method == 'ensemble_v4':
            mask = detect_ensemble_v4(img)
        elif detection_method == 'norm_rgb':
            threshold = getattr(config, 'NORM_RGB_THRESHOLD', 0.4)
            mask = detect_normalized_rgb(img, target='blue', threshold=threshold)
        elif detection_method == 'legacy':
            # 기존 방식 (하위 호환)
            mask = create_height_mask(img, config.HEIGHT_THRESHOLD_MIN,
                                       config.HEIGHT_THRESHOLD_MAX)
        else:
            mask = detect_ensemble_v4(img)
    else:
        # 2D 색상 모드: HSV 기반 검출
        mask = create_mask_from_hsv(img, config.LOWER_HSV, config.UPPER_HSV)

    # 형태학적 연산 (노이즈 제거)
    mask = apply_morphology(mask, 'open', config.MORPH_KERNEL_SIZE, cv.MORPH_ELLIPSE)
    mask = apply_morphology(mask, 'close', config.MORPH_KERNEL_SIZE, cv.MORPH_ELLIPSE)

    # 윤곽선 검출
    contours = find_contours(mask)

    # 필터링 (면적, 원형도 기준)
    filtered_contours = filter_contours(contours, config)

    return filtered_contours, mask


# ============================================================
# 분석 결과 기반 최적 검출 방법들
# ============================================================

def detect_otsu_blue_inv(img):
    """
    Blue 채널 Otsu 반전

    Blue 채널에 Otsu 자동 임계값을 적용하고 반전.
    3D 높이 맵에서 Blue=높은 경사이므로, Blue가 낮은 영역이
    솔더 페이스트의 실제 도포 영역에 해당.
    """
    blue = img[:, :, 0]
    _, mask = cv.threshold(blue, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return mask


def detect_clahe_blue_inv(img, clip_limit=4.0, tile_size=4):
    """
    [안정성 1위] CLAHE + Blue 채널 Otsu 반전 (std=0.010)

    CLAHE로 Blue 채널의 국소 대비를 강화한 후 Otsu 반전.
    모든 이미지에서 가장 일관된 결과를 보임 (표준편차 최소).
    """
    blue = img[:, :, 0]
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(blue)
    _, mask = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return mask


def detect_adaptive_blue(img, block_size=7, c_val=10):
    """
    적응형 임계값 Blue 채널

    지역적 밝기 변화에 대응하는 적응형 임계값.
    조명이 균일하지 않은 실제 검사 환경에 유리.
    """
    blue = img[:, :, 0]
    h, w = blue.shape
    bs = min(block_size, min(h, w) - 1)
    if bs % 2 == 0:
        bs -= 1
    if bs < 3:
        bs = 3
    mask = cv.adaptiveThreshold(blue, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv.THRESH_BINARY, bs, c_val)
    return mask


def detect_ensemble(img):
    """
    앙상블 검출 v3: 안정성 + 높은 IoU 방법 결합

    v3 분석 결과 반영:
    1. CLAHE Blue inv cl4.0 (안정성 1위, std=0.010, 가중치 2.0)
    2. CLAHE Blue inv cl8.0 (안정성 2위, std=0.013, 가중치 1.8)
    3. Saturation 30-255 (평균 IoU 1위, 가중치 1.2)
    4. Adaptive Lab_L bs11 c10 (평균 IoU 5위, 가중치 1.0)
    5. Otsu Blue inv (안정적, 가중치 1.0)

    2.5 이상 동의하면 검출로 판정.
    """
    h, w = img.shape[:2]
    votes = np.zeros((h, w), dtype=np.float32)

    # 방법 1: CLAHE Blue inv cl4.0 (안정성 1위, 가중치 2.0)
    mask1 = detect_clahe_blue_inv(img, clip_limit=4.0, tile_size=4)
    votes += (mask1 > 0).astype(np.float32) * 2.0

    # 방법 2: CLAHE Blue inv cl8.0 (안정성 2위, 가중치 1.8)
    mask2 = detect_clahe_blue_inv(img, clip_limit=8.0, tile_size=4)
    votes += (mask2 > 0).astype(np.float32) * 1.8

    # 방법 3: Saturation 30-255 (평균 IoU 1위, 가중치 1.2)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]
    mask3 = cv.inRange(s_ch, 30, 255)
    votes += (mask3 > 0).astype(np.float32) * 1.2

    # 방법 4: Adaptive Lab_L bs11 c10 (평균 IoU 5위, 가중치 1.0)
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    l_ch = lab[:, :, 0]
    bs = min(11, min(h, w) - 1)
    if bs % 2 == 0:
        bs -= 1
    if bs < 3:
        bs = 3
    mask4 = cv.adaptiveThreshold(l_ch, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY, bs, 10)
    votes += (mask4 > 0).astype(np.float32) * 1.0

    # 방법 5: Otsu Blue inv (가중치 1.0)
    mask5 = detect_otsu_blue_inv(img)
    votes += (mask5 > 0).astype(np.float32) * 1.0

    # 투표 임계값: 가중합 >= 2.5
    mask = (votes >= 2.5).astype(np.uint8) * 255

    return mask


# ============================================================
# v4: Normalized RGB + 노이즈 제거 기반 검출
# ============================================================

def detect_normalized_rgb(img, target='blue', threshold=0.4):
    """
    Normalized RGB 검출 (산업용 AOI 표준 방식)

    r = R/(R+G+B), g = G/(R+G+B), b = B/(R+G+B)
    조명 밝기에 무관하게 색상 비율만으로 영역 판단.

    Args:
        img: BGR 이미지
        target: 검출 대상 ('blue', 'red', 'green')
        threshold: 비율 임계값 (0.0~1.0)

    Returns:
        numpy.ndarray: 이진 마스크
    """
    norm_b, norm_g, norm_r = normalize_rgb(img)

    if target == 'blue':
        mask = (norm_b >= threshold).astype(np.uint8) * 255
    elif target == 'red':
        mask = (norm_r >= threshold).astype(np.uint8) * 255
    elif target == 'green':
        mask = (norm_g >= threshold).astype(np.uint8) * 255
    else:
        mask = (norm_b >= threshold).astype(np.uint8) * 255

    return mask


def detect_norm_rgb_diff(img, pair='b-r', threshold=0.1):
    """
    Normalized RGB 차이 기반 검출

    특정 색상이 다른 색상보다 얼마나 우세한지로 판단.
    예: b-r > 0.1 → Blue 비율이 Red보다 10% 이상 높은 영역

    Args:
        img: BGR 이미지
        pair: 'b-r', 'b-g', 'r-g', 'r-b' 등
        threshold: 차이 임계값

    Returns:
        numpy.ndarray: 이진 마스크
    """
    norm_b, norm_g, norm_r = normalize_rgb(img)

    ch_map = {'b': norm_b, 'g': norm_g, 'r': norm_r}
    parts = pair.split('-')
    if len(parts) == 2 and parts[0] in ch_map and parts[1] in ch_map:
        diff = ch_map[parts[0]] - ch_map[parts[1]]
        mask = (diff >= threshold).astype(np.uint8) * 255
    else:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

    return mask


def detect_color_ratio(img, pair='bg', threshold=1.5):
    """
    채널 비율 검출: B/G, R/G, B/R 등 두 채널의 직접 비율로 판단

    AOI에서 경사 방향 판별에 사용. Normalized RGB와 달리
    두 채널만의 상대적 관계에 집중.

    Args:
        img: BGR 이미지
        pair: 'bg'=B/G, 'br'=B/R, 'rg'=R/G 등
        threshold: 비율 임계값 (1.0 이상)

    Returns:
        numpy.ndarray: 이진 마스크
    """
    b, g, r = cv.split(img)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    pair_map = {
        'bg': (b, g), 'br': (b, r), 'rg': (r, g),
        'gb': (g, b), 'rb': (r, b), 'gr': (g, r)
    }

    if pair in pair_map:
        num, den = pair_map[pair]
        den = np.maximum(den, 1.0)
        ratio = num / den
        mask = (ratio >= threshold).astype(np.uint8) * 255
    else:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

    return mask


def detect_gradient_pattern(img, direction='horizontal'):
    """
    색상 그라데이션 방향 분석

    솔더 필렛은 Blue→Green→Red 순차적 색상 전환이 나타남.
    이 전환 패턴의 gradient 방향으로 필렛 영역 식별.

    Args:
        img: BGR 이미지
        direction: 'horizontal' or 'vertical'

    Returns:
        numpy.ndarray: 이진 마스크
    """
    norm_b, norm_g, norm_r = normalize_rgb(img)

    # Blue의 gradient (Blue가 감소하는 방향 = 필렛 곡면)
    if direction == 'horizontal':
        grad_b = cv.Sobel(norm_b, cv.CV_32F, 1, 0, ksize=3)
        grad_r = cv.Sobel(norm_r, cv.CV_32F, 1, 0, ksize=3)
    else:
        grad_b = cv.Sobel(norm_b, cv.CV_32F, 0, 1, ksize=3)
        grad_r = cv.Sobel(norm_r, cv.CV_32F, 0, 1, ksize=3)

    # Blue 감소 + Red 증가 = 필렛 전환 영역 (또는 반대)
    transition = np.abs(grad_b) + np.abs(grad_r)

    # 전환이 강한 영역 주변이 필렛
    transition_norm = cv.normalize(transition, None, 0, 255, cv.NORM_MINMAX)
    _, mask = cv.threshold(transition_norm.astype(np.uint8), 30, 255, cv.THRESH_BINARY)

    # 전환 영역을 팽창하여 필렛 면적으로 확장
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.dilate(mask, kernel, iterations=2)

    return mask


def detect_denoised(img, denoise_method='bilateral', detect_func=None, **kwargs):
    """
    노이즈 제거 후 검출 (wrapper)

    AOI 이미지의 표면 거칠기 노이즈를 먼저 제거한 후
    기존 검출 방법을 적용.

    Args:
        img: BGR 이미지
        denoise_method: 'bilateral' or 'nlm'
        detect_func: 검출 함수 (mask = func(denoised_img, **kwargs))
        **kwargs: 검출 함수에 전달할 인자

    Returns:
        numpy.ndarray: 이진 마스크
    """
    if denoise_method == 'bilateral':
        denoised = apply_bilateral_filter(img)
    elif denoise_method == 'nlm':
        denoised = apply_nlm_denoise(img)
    else:
        denoised = img

    if detect_func is not None:
        return detect_func(denoised, **kwargs)
    else:
        return detect_otsu_blue_inv(denoised)


def detect_ensemble_v4(img):
    """
    v4 앙상블: Normalized RGB + 노이즈 제거 + 기존 안정 방법 결합

    v3 대비 변경:
    - Normalized RGB 검출 추가 (산업용 AOI 표준)
    - Bilateral Filter 전처리 적용 방법 추가
    - 색상 비율 기반 검출 추가

    가중치는 v4 분석 결과로 업데이트 예정.
    현재는 초기 설정.
    """
    h, w = img.shape[:2]
    votes = np.zeros((h, w), dtype=np.float32)

    # 노이즈 제거된 이미지 준비
    denoised = apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75)

    # 방법 1: Bilateral + CLAHE Blue inv (노이즈 제거 + 안정성 1위)
    mask1 = detect_clahe_blue_inv(denoised, clip_limit=4.0, tile_size=4)
    votes += (mask1 > 0).astype(np.float32) * 2.0

    # 방법 2: Normalized RGB blue (산업용 AOI 표준)
    mask2 = detect_normalized_rgb(denoised, target='blue', threshold=0.40)
    votes += (mask2 > 0).astype(np.float32) * 1.8

    # 방법 3: 채널 비율 B/G (Blue가 Green보다 우세한 영역)
    mask3 = detect_color_ratio(denoised, pair='bg', threshold=1.3)
    votes += (mask3 > 0).astype(np.float32) * 1.5

    # 방법 4: Bilateral + Saturation (기존 IoU 1위에 노이즈 제거)
    hsv = cv.cvtColor(denoised, cv.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]
    mask4 = cv.inRange(s_ch, 30, 255)
    votes += (mask4 > 0).astype(np.float32) * 1.0

    # 방법 5: Norm RGB 차이 b-r (Blue 비율이 Red보다 높은 영역)
    mask5 = detect_norm_rgb_diff(denoised, pair='b-r', threshold=0.05)
    votes += (mask5 > 0).astype(np.float32) * 1.0

    # 투표 임계값: 가중합 >= 3.0 (5가지 방법 중 충분한 동의)
    mask = (votes >= 3.0).astype(np.uint8) * 255

    return mask


# ============================================================
# 기존 방법 (하위 호환)
# ============================================================

def create_height_mask(img, min_height, max_height):
    """
    기존 3D 높이 맵 높이 기반 마스크 생성 (하위 호환용)

    Blue channel 값을 높이로 해석:
    - 파란색 = 높음 (높은 경사)
    - 초록색 = 중간
    - 빨강색 = 낮음 (낮은 경사)

    Args:
        img (numpy.ndarray): BGR 이미지 (3D 높이 맵)
        min_height (int): 최소 높이 임계값 (0-255)
        max_height (int): 최대 높이 임계값 (0-255)

    Returns:
        numpy.ndarray: 이진 마스크 (높이 범위 내 = 255, 외부 = 0)
    """
    blue_channel = img[:, :, 0]
    mask = cv.inRange(blue_channel, min_height, max_height)
    return mask


# ============================================================
# 공통 유틸리티
# ============================================================

def find_contours(binary_img):
    """
    이진 이미지에서 윤곽선 검출

    Args:
        binary_img (numpy.ndarray): 이진 이미지 (마스크)

    Returns:
        list: 검출된 윤곽선 리스트
    """
    contours, hierarchies = cv.findContours(
        binary_img,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )
    return contours


def filter_contours(contours, config):
    """
    윤곽선 필터링 (면적, 원형도 기준)

    Args:
        contours (list): 윤곽선 리스트
        config: 설정 객체

    Returns:
        list: 필터링된 윤곽선 리스트
    """
    filtered = []

    for cnt in contours:
        area = cv.contourArea(cnt)

        if area < config.MIN_AREA or area > config.MAX_AREA:
            continue

        circularity = calculate_circularity(cnt)

        if circularity < config.MIN_CIRCULARITY or circularity > config.MAX_CIRCULARITY:
            continue

        filtered.append(cnt)

    return filtered


def calculate_circularity(contour):
    """
    윤곽선의 원형도 계산
    원형도 = 4π × (면적 / 둘레²)
    1.0에 가까울수록 원에 가까움

    Args:
        contour (numpy.ndarray): 윤곽선

    Returns:
        float: 원형도 (0.0 ~ 1.0)
    """
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)

    if perimeter == 0:
        return 0.0

    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return min(circularity, 1.0)


def get_contour_properties(contour):
    """
    윤곽선의 기하학적 특성 계산

    Args:
        contour (numpy.ndarray): 윤곽선

    Returns:
        dict: 윤곽선 특성
    """
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    circularity = calculate_circularity(contour)

    M = cv.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0

    x, y, w, h = cv.boundingRect(contour)

    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'center': (cx, cy),
        'bounding_rect': (x, y, w, h)
    }


def detect_with_adaptive_threshold(gray_img, config):
    """
    적응형 임계값을 사용한 검출 (대체 방법)
    조명이 불균일한 경우 유용

    Args:
        gray_img (numpy.ndarray): 그레이스케일 이미지
        config: 설정 객체

    Returns:
        tuple: (검출된 윤곽선 리스트, 이진 이미지)
    """
    blurred = cv.GaussianBlur(gray_img, config.BLUR_KERNEL_SIZE, 0)

    binary = cv.adaptiveThreshold(
        blurred, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        11, 2
    )

    binary = apply_morphology(binary, 'open', config.MORPH_KERNEL_SIZE, cv.MORPH_ELLIPSE)
    binary = apply_morphology(binary, 'close', config.MORPH_KERNEL_SIZE, cv.MORPH_ELLIPSE)

    contours = find_contours(binary)
    filtered_contours = filter_contours(contours, config)

    return filtered_contours, binary


def sort_contours_by_area(contours, descending=True):
    """윤곽선을 면적 순으로 정렬"""
    return sorted(contours, key=cv.contourArea, reverse=descending)


def sort_contours_by_position(contours, axis='x'):
    """윤곽선을 위치 순으로 정렬 (좌→우 또는 상→하)"""
    def get_position(cnt):
        M = cv.moments(cnt)
        if M['m00'] != 0:
            if axis == 'x':
                return int(M['m10'] / M['m00'])
            else:
                return int(M['m01'] / M['m00'])
        return 0

    return sorted(contours, key=get_position)


if __name__ == '__main__':
    from config import Config
    from image_processor import load_image, preprocess_image

    config = Config()

    # 테스트 이미지 실행
    import os
    test_dir = 'test'
    if os.path.exists(test_dir):
        for fname in sorted(os.listdir(test_dir)):
            fpath = os.path.join(test_dir, fname)
            img = load_image(fpath)
            if img is None:
                continue

            print(f"\n{'='*50}")
            print(f"이미지: {fname} (크기: {img.shape})")

            # 각 방법별 테스트
            for method in ['otsu_blue', 'clahe_blue', 'adaptive', 'ensemble']:
                config.DETECTION_METHOD = method
                contours, mask = detect_solder_paste(img, config)
                area_pct = np.count_nonzero(mask) / (img.shape[0] * img.shape[1]) * 100
                print(f"  {method:<15s}: 윤곽선 {len(contours)}개, "
                      f"마스크 영역 {area_pct:.1f}%")
