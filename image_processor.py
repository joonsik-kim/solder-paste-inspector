"""
이미지 전처리 모듈
이미지 로드, 색상 변환, 블러, 리사이징 등의 전처리 작업 수행
"""

import cv2 as cv
import numpy as np


def load_image(image_path):
    """
    이미지 파일 로드

    Args:
        image_path (str): 이미지 파일 경로

    Returns:
        numpy.ndarray: 로드된 이미지 (BGR 형식)
        None: 로드 실패 시
    """
    try:
        img = cv.imread(image_path)
        if img is None:
            print(f"이미지 로드 실패: {image_path}")
            return None
        return img
    except Exception as e:
        print(f"이미지 로드 오류: {e}")
        return None


def resize_image(img, scale=1.0):
    """
    이미지 크기 조정

    Args:
        img (numpy.ndarray): 입력 이미지
        scale (float): 스케일 비율 (0.5 = 50%, 1.0 = 원본)

    Returns:
        numpy.ndarray: 리사이징된 이미지
    """
    if scale == 1.0:
        return img

    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dimensions = (width, height)

    resized = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    return resized


def convert_to_hsv(img):
    """
    BGR 이미지를 HSV 색상 공간으로 변환

    Args:
        img (numpy.ndarray): BGR 형식 이미지

    Returns:
        numpy.ndarray: HSV 형식 이미지
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return hsv


def convert_to_gray(img):
    """
    BGR 이미지를 그레이스케일로 변환

    Args:
        img (numpy.ndarray): BGR 형식 이미지

    Returns:
        numpy.ndarray: 그레이스케일 이미지
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray


def apply_gaussian_blur(img, kernel_size=(5, 5)):
    """
    가우시안 블러 적용 (노이즈 제거)

    Args:
        img (numpy.ndarray): 입력 이미지
        kernel_size (tuple): 커널 크기 (홀수여야 함)

    Returns:
        numpy.ndarray: 블러 처리된 이미지
    """
    blurred = cv.GaussianBlur(img, kernel_size, 0)
    return blurred


def apply_morphology(img, operation='open', kernel_size=(5, 5), shape=cv.MORPH_ELLIPSE):
    """
    형태학적 연산 적용 (Opening/Closing)

    Args:
        img (numpy.ndarray): 입력 이미지 (이진 이미지)
        operation (str): 'open', 'close', 'erode', 'dilate'
        kernel_size (tuple): 커널 크기
        shape: 커널 형태 (cv.MORPH_RECT, cv.MORPH_ELLIPSE, cv.MORPH_CROSS)

    Returns:
        numpy.ndarray: 형태학적 연산이 적용된 이미지
    """
    kernel = cv.getStructuringElement(shape, kernel_size)

    if operation == 'open':
        result = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    elif operation == 'close':
        result = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    elif operation == 'erode':
        result = cv.erode(img, kernel)
    elif operation == 'dilate':
        result = cv.dilate(img, kernel)
    else:
        print(f"알 수 없는 연산: {operation}")
        result = img

    return result


def preprocess_image(img, config):
    """
    이미지 전처리 파이프라인
    1. 리사이징
    2. HSV 변환
    3. 블러 적용

    Args:
        img (numpy.ndarray): 입력 이미지 (BGR)
        config: 설정 객체 (Config 클래스)

    Returns:
        tuple: (전처리된 HSV 이미지, 리사이징된 원본 이미지)
    """
    # 1. 리사이징 (처리 속도 향상)
    resized = resize_image(img, config.RESIZE_SCALE)

    # 2. HSV 색상 공간 변환
    hsv = convert_to_hsv(resized)

    # 3. 가우시안 블러 적용 (노이즈 제거)
    blurred_hsv = apply_gaussian_blur(hsv, config.BLUR_KERNEL_SIZE)

    return blurred_hsv, resized


def create_mask_from_hsv(hsv_img, lower_bound, upper_bound):
    """
    HSV 이미지에서 색상 범위 마스크 생성

    Args:
        hsv_img (numpy.ndarray): HSV 형식 이미지
        lower_bound (numpy.ndarray): 하한 HSV 값
        upper_bound (numpy.ndarray): 상한 HSV 값

    Returns:
        numpy.ndarray: 이진 마스크 (255 = 범위 내, 0 = 범위 외)
    """
    mask = cv.inRange(hsv_img, lower_bound, upper_bound)
    return mask


def enhance_contrast(img):
    """
    이미지 대비 향상 (CLAHE: Contrast Limited Adaptive Histogram Equalization)

    Args:
        img (numpy.ndarray): 입력 이미지 (그레이스케일 또는 BGR)

    Returns:
        numpy.ndarray: 대비가 향상된 이미지
    """
    # BGR 이미지인 경우 LAB 색상 공간으로 변환
    if len(img.shape) == 3:
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)

        # L 채널에 CLAHE 적용
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # 채널 병합
        lab_enhanced = cv.merge([l_enhanced, a, b])
        enhanced = cv.cvtColor(lab_enhanced, cv.COLOR_LAB2BGR)
    else:
        # 그레이스케일 이미지
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)

    return enhanced


if __name__ == '__main__':
    # 테스트
    from config import Config

    # 설정 로드
    config = Config()

    # 테스트 이미지 로드
    # 주의: 실제 이미지 경로로 변경 필요
    test_image_path = '../opencv-course-master/Resources/Photos/cats.jpg'

    img = load_image(test_image_path)
    if img is not None:
        print(f"원본 이미지 크기: {img.shape}")

        # 전처리
        hsv, resized = preprocess_image(img, config)
        print(f"전처리 후 크기: {hsv.shape}")

        # 마스크 생성 테스트
        mask = create_mask_from_hsv(hsv, config.LOWER_HSV, config.UPPER_HSV)
        print(f"마스크 크기: {mask.shape}")

        # 결과 표시
        cv.imshow('Original', resized)
        cv.imshow('HSV', hsv)
        cv.imshow('Mask', mask)
        cv.waitKey(0)
        cv.destroyAllWindows()
