"""
솔더 페이스트 검출 엔진
3D 높이 맵 기반 검출 또는 HSV 색상 범위 기반 검출
"""

import cv2 as cv
import numpy as np
from image_processor import create_mask_from_hsv, apply_morphology


def detect_solder_paste(img, config):
    """
    솔더 페이스트 영역 검출 메인 함수

    3D 높이 맵 모드: Blue channel을 높이로 해석하여 검출
    2D 색상 모드: HSV 색상 범위 기반 검출

    Args:
        img (numpy.ndarray): BGR 또는 HSV 형식 이미지
        config: 설정 객체 (Config 클래스)

    Returns:
        tuple: (검출된 윤곽선 리스트, 이진 마스크)
    """
    if config.HEIGHT_MAP_MODE:
        # 3D 높이 맵 모드: Blue channel 기반 검출
        mask = create_height_mask(img, config.HEIGHT_THRESHOLD_MIN, config.HEIGHT_THRESHOLD_MAX)
    else:
        # 2D 색상 모드: HSV 기반 검출
        mask = create_mask_from_hsv(img, config.LOWER_HSV, config.UPPER_HSV)

    # 2. 형태학적 연산 (노이즈 제거)
    # Opening: 작은 노이즈 제거
    mask = apply_morphology(mask, 'open', config.MORPH_KERNEL_SIZE, cv.MORPH_ELLIPSE)

    # Closing: 작은 구멍 메우기
    mask = apply_morphology(mask, 'close', config.MORPH_KERNEL_SIZE, cv.MORPH_ELLIPSE)

    # 3. 윤곽선 검출
    contours = find_contours(mask)

    # 4. 필터링 (면적, 원형도 기준)
    filtered_contours = filter_contours(contours, config)

    return filtered_contours, mask


def create_height_mask(img, min_height, max_height):
    """
    3D 높이 맵에서 높이 기반 마스크 생성

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
    # Blue channel 추출 (BGR 이미지이므로 index 0)
    blue_channel = img[:, :, 0]

    # 높이 임계값 적용
    mask = cv.inRange(blue_channel, min_height, max_height)

    return mask


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
        cv.RETR_EXTERNAL,      # 외부 윤곽선만 검출
        cv.CHAIN_APPROX_SIMPLE  # 윤곽선 압축
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
        # 면적 계산
        area = cv.contourArea(cnt)

        # 면적 필터
        if area < config.MIN_AREA or area > config.MAX_AREA:
            continue

        # 원형도 계산
        circularity = calculate_circularity(cnt)

        # 원형도 필터
        if circularity < config.MIN_CIRCULARITY or circularity > config.MAX_CIRCULARITY:
            continue

        # 필터 통과
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

    # 수치 안정성을 위해 1.0을 초과하지 않도록 제한
    return min(circularity, 1.0)


def get_contour_properties(contour):
    """
    윤곽선의 기하학적 특성 계산

    Args:
        contour (numpy.ndarray): 윤곽선

    Returns:
        dict: 윤곽선 특성
            - area: 면적
            - perimeter: 둘레
            - circularity: 원형도
            - center: 무게중심 (x, y)
            - bounding_rect: 외접 사각형 (x, y, w, h)
    """
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    circularity = calculate_circularity(contour)

    # 무게중심
    M = cv.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0

    # 외접 사각형
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
    # 가우시안 블러 적용
    blurred = cv.GaussianBlur(gray_img, config.BLUR_KERNEL_SIZE, 0)

    # 적응형 임계값
    binary = cv.adaptiveThreshold(
        blurred,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        11,  # 블록 크기
        2    # C 상수
    )

    # 형태학적 연산
    binary = apply_morphology(binary, 'open', config.MORPH_KERNEL_SIZE, cv.MORPH_ELLIPSE)
    binary = apply_morphology(binary, 'close', config.MORPH_KERNEL_SIZE, cv.MORPH_ELLIPSE)

    # 윤곽선 검출
    contours = find_contours(binary)

    # 필터링
    filtered_contours = filter_contours(contours, config)

    return filtered_contours, binary


def sort_contours_by_area(contours, descending=True):
    """
    윤곽선을 면적 순으로 정렬

    Args:
        contours (list): 윤곽선 리스트
        descending (bool): True=내림차순, False=오름차순

    Returns:
        list: 정렬된 윤곽선 리스트
    """
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=descending)
    return sorted_contours


def sort_contours_by_position(contours, axis='x'):
    """
    윤곽선을 위치 순으로 정렬 (좌→우 또는 상→하)

    Args:
        contours (list): 윤곽선 리스트
        axis (str): 'x' (좌→우) 또는 'y' (상→하)

    Returns:
        list: 정렬된 윤곽선 리스트
    """
    def get_position(cnt):
        M = cv.moments(cnt)
        if M['m00'] != 0:
            if axis == 'x':
                return int(M['m10'] / M['m00'])
            else:
                return int(M['m01'] / M['m00'])
        return 0

    sorted_contours = sorted(contours, key=get_position)
    return sorted_contours


if __name__ == '__main__':
    # 테스트
    from config import Config
    from image_processor import load_image, preprocess_image

    # 설정 로드
    config = Config()

    # 테스트 이미지 로드
    # 주의: 실제 솔더 페이스트 이미지 경로로 변경 필요
    test_image_path = '../opencv-course-master/Resources/Photos/cats.jpg'

    img = load_image(test_image_path)
    if img is not None:
        # 전처리
        hsv, resized = preprocess_image(img, config)

        # 검출
        contours, mask = detect_solder_paste(hsv, config)

        print(f"검출된 윤곽선 수: {len(contours)}")

        # 각 윤곽선의 특성 출력
        for i, cnt in enumerate(contours):
            props = get_contour_properties(cnt)
            print(f"윤곽선 {i+1}: 면적={props['area']:.2f}, "
                  f"원형도={props['circularity']:.2f}, "
                  f"중심={props['center']}")

        # 결과 표시
        output = resized.copy()
        cv.drawContours(output, contours, -1, (0, 255, 0), 2)

        cv.imshow('Original', resized)
        cv.imshow('Mask', mask)
        cv.imshow('Detected', output)
        cv.waitKey(0)
        cv.destroyAllWindows()
