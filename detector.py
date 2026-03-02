"""
솔더 페이스트 검출 엔진
3D 높이 맵 기반 검출 또는 HSV 색상 범위 기반 검출

v2: 종합 분석 결과 기반 최적화
- 330개 방법 IoU 평가 결과 반영
- Top 1: Otsu_Blue_inv (avg IoU=0.5699)
- Top 2: CLAHE_Blue_inv (avg IoU=0.5652)
- 앙상블 방식으로 안정성 향상
"""

import cv2 as cv
import numpy as np
from image_processor import create_mask_from_hsv, apply_morphology


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
        elif detection_method == 'legacy':
            # 기존 방식 (하위 호환)
            mask = create_height_mask(img, config.HEIGHT_THRESHOLD_MIN,
                                       config.HEIGHT_THRESHOLD_MAX)
        else:
            mask = detect_ensemble(img)
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
    [Top 1] Blue 채널 Otsu 반전 (avg IoU=0.5699)

    Blue 채널에 Otsu 자동 임계값을 적용하고 반전.
    3D 높이 맵에서 Blue=높은 경사이므로, Blue가 낮은 영역이
    솔더 페이스트의 실제 도포 영역에 해당.

    Otsu의 장점:
    - 히스토그램 기반 자동 임계값 결정
    - 이미지마다 다른 밝기 분포에 적응적으로 동작
    """
    blue = img[:, :, 0]
    _, mask = cv.threshold(blue, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return mask


def detect_clahe_blue_inv(img, clip_limit=2.0, tile_size=4):
    """
    [Top 2] CLAHE + Blue 채널 Otsu 반전 (avg IoU=0.5652)

    CLAHE(대비 제한 적응형 히스토그램 균등화)로 Blue 채널의
    국소 대비를 강화한 후 Otsu 적용.

    장점:
    - 조명이 불균일한 환경에서도 일관된 결과
    - 어두운 영역의 세부 디테일 향상
    """
    blue = img[:, :, 0]
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(blue)
    _, mask = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return mask


def detect_adaptive_blue(img, block_size=7, c_val=10):
    """
    [Top 7] 적응형 임계값 Blue 채널 (avg IoU=0.5415)

    지역적 밝기 변화에 대응하는 적응형 임계값.
    조명이 균일하지 않은 실제 검사 환경에 유리.
    """
    blue = img[:, :, 0]
    h, w = blue.shape
    # block_size가 이미지보다 작아야 함
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
    앙상블 검출: 상위 방법들의 투표 기반 결합

    Top 방법들의 결과를 결합하여 안정성 향상:
    1. Otsu Blue 반전 (Top 1)
    2. CLAHE Blue 반전 (Top 2)
    3. Lab L 임계값 (Top 4)
    4. HSV Value 임계값 (Top 5)
    5. Saturation 임계값 (Top 6)

    2개 이상의 방법이 동의하면 검출로 판정.
    """
    h, w = img.shape[:2]
    votes = np.zeros((h, w), dtype=np.float32)

    # 방법 1: Otsu Blue 반전 (가중치 1.5 - 최고 성능)
    mask1 = detect_otsu_blue_inv(img)
    votes += (mask1 > 0).astype(np.float32) * 1.5

    # 방법 2: CLAHE Blue 반전 (가중치 1.3)
    mask2 = detect_clahe_blue_inv(img)
    votes += (mask2 > 0).astype(np.float32) * 1.3

    # 방법 3: Lab L 채널 임계값 (가중치 1.0)
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    l_ch = lab[:, :, 0]
    mask3 = cv.inRange(l_ch, 0, 150)
    votes += (mask3 > 0).astype(np.float32) * 1.0

    # 방법 4: HSV Value 임계값 (가중치 0.8)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]
    mask4 = cv.inRange(v_ch, 0, 180)
    votes += (mask4 > 0).astype(np.float32) * 0.8

    # 방법 5: Saturation 임계값 (가중치 0.8)
    s_ch = hsv[:, :, 1]
    mask5 = cv.inRange(s_ch, 30, 255)
    votes += (mask5 > 0).astype(np.float32) * 0.8

    # 투표 임계값: 가중합 >= 2.5 (최소 2개 방법 동의)
    mask = (votes >= 2.5).astype(np.uint8) * 255

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
