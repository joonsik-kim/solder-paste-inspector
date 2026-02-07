"""
면적 측정 모듈
윤곽선 면적 계산, 픽셀-mm 변환, 통계 분석
"""

import cv2 as cv
import numpy as np


def measure_area(contour, pixels_per_mm):
    """
    단일 윤곽선의 면적 측정 및 특성 계산

    Args:
        contour (numpy.ndarray): 윤곽선
        pixels_per_mm (float): 픽셀당 mm 비율

    Returns:
        dict: 측정 결과
            - area_pixels: 픽셀 단위 면적
            - area_mm2: mm² 단위 면적
            - center: 무게중심 (x, y)
            - circularity: 원형도
            - perimeter_pixels: 픽셀 단위 둘레
            - perimeter_mm: mm 단위 둘레
            - bounding_rect: 외접 사각형
    """
    # 픽셀 단위 면적
    area_pixels = cv.contourArea(contour)

    # 실제 면적 (mm²)
    area_mm2 = area_pixels / (pixels_per_mm ** 2)

    # 둘레
    perimeter_pixels = cv.arcLength(contour, True)
    perimeter_mm = perimeter_pixels / pixels_per_mm

    # 무게중심
    M = cv.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0

    # 원형도
    if perimeter_pixels > 0:
        circularity = 4 * np.pi * (area_pixels / (perimeter_pixels ** 2))
        circularity = min(circularity, 1.0)
    else:
        circularity = 0.0

    # 외접 사각형
    x, y, w, h = cv.boundingRect(contour)

    return {
        'area_pixels': area_pixels,
        'area_mm2': area_mm2,
        'center': (cx, cy),
        'circularity': circularity,
        'perimeter_pixels': perimeter_pixels,
        'perimeter_mm': perimeter_mm,
        'bounding_rect': (x, y, w, h),
        'contour': contour
    }


def measure_all_contours(contours, pixels_per_mm):
    """
    모든 윤곽선의 면적 측정

    Args:
        contours (list): 윤곽선 리스트
        pixels_per_mm (float): 픽셀당 mm 비율

    Returns:
        list: 측정 결과 리스트 (dict의 리스트)
    """
    measurements = []

    for i, cnt in enumerate(contours):
        measurement = measure_area(cnt, pixels_per_mm)
        measurement['id'] = i + 1  # ID 추가
        measurements.append(measurement)

    return measurements


def calculate_statistics(measurements):
    """
    측정 결과 통계 분석

    Args:
        measurements (list): 측정 결과 리스트

    Returns:
        dict: 통계 결과
            - count: 검출된 개수
            - total_area_mm2: 총 면적
            - mean_area_mm2: 평균 면적
            - std_area_mm2: 면적 표준편차
            - min_area_mm2: 최소 면적
            - max_area_mm2: 최대 면적
            - median_area_mm2: 중앙값 면적
    """
    if not measurements:
        return {
            'count': 0,
            'total_area_mm2': 0.0,
            'mean_area_mm2': 0.0,
            'std_area_mm2': 0.0,
            'min_area_mm2': 0.0,
            'max_area_mm2': 0.0,
            'median_area_mm2': 0.0
        }

    areas = [m['area_mm2'] for m in measurements]

    return {
        'count': len(measurements),
        'total_area_mm2': sum(areas),
        'mean_area_mm2': np.mean(areas),
        'std_area_mm2': np.std(areas),
        'min_area_mm2': min(areas),
        'max_area_mm2': max(areas),
        'median_area_mm2': np.median(areas)
    }


def format_measurement_report(measurements, statistics):
    """
    측정 결과를 텍스트 리포트로 포맷

    Args:
        measurements (list): 측정 결과 리스트
        statistics (dict): 통계 결과

    Returns:
        str: 포맷된 리포트 텍스트
    """
    report = []
    report.append("="*60)
    report.append("솔더 페이스트 면적 측정 리포트")
    report.append("="*60)
    report.append("")

    # 통계 정보
    report.append("[통계 정보]")
    report.append(f"검출 개수: {statistics['count']}개")
    report.append(f"총 면적: {statistics['total_area_mm2']:.3f} mm²")
    report.append(f"평균 면적: {statistics['mean_area_mm2']:.3f} ± {statistics['std_area_mm2']:.3f} mm²")
    report.append(f"면적 범위: {statistics['min_area_mm2']:.3f} ~ {statistics['max_area_mm2']:.3f} mm²")
    report.append(f"중앙값 면적: {statistics['median_area_mm2']:.3f} mm²")
    report.append("")

    # 개별 측정 결과
    report.append("[개별 측정 결과]")
    report.append(f"{'ID':<5} {'면적(mm²)':<12} {'중심(x,y)':<15} {'원형도':<8}")
    report.append("-"*60)

    for m in measurements:
        report.append(
            f"{m['id']:<5} "
            f"{m['area_mm2']:<12.3f} "
            f"{str(m['center']):<15} "
            f"{m['circularity']:<8.2f}"
        )

    report.append("="*60)

    return "\n".join(report)


def save_measurements_to_json(measurements, statistics, output_path):
    """
    측정 결과를 JSON 파일로 저장

    Args:
        measurements (list): 측정 결과 리스트
        statistics (dict): 통계 결과
        output_path (str): 출력 파일 경로
    """
    import json

    # NumPy 타입을 Python 기본 타입으로 변환
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    # 윤곽선 데이터 제거 (JSON 직렬화 불가)
    measurements_copy = []
    for m in measurements:
        m_copy = {k: v for k, v in m.items() if k != 'contour'}
        measurements_copy.append(m_copy)

    data = {
        'measurements': measurements_copy,
        'statistics': statistics
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False, default=convert_to_serializable)
        print(f"측정 결과 저장 완료: {output_path}")
    except Exception as e:
        print(f"측정 결과 저장 실패: {e}")


def save_measurements_to_csv(measurements, output_path):
    """
    측정 결과를 CSV 파일로 저장

    Args:
        measurements (list): 측정 결과 리스트
        output_path (str): 출력 파일 경로
    """
    import csv

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # CSV 헤더
            fieldnames = ['id', 'area_mm2', 'area_pixels', 'center_x', 'center_y',
                         'circularity', 'perimeter_mm', 'perimeter_pixels']

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # 데이터 쓰기
            for m in measurements:
                row = {
                    'id': m['id'],
                    'area_mm2': m['area_mm2'],
                    'area_pixels': m['area_pixels'],
                    'center_x': m['center'][0],
                    'center_y': m['center'][1],
                    'circularity': m['circularity'],
                    'perimeter_mm': m['perimeter_mm'],
                    'perimeter_pixels': m['perimeter_pixels']
                }
                writer.writerow(row)

        print(f"측정 결과 CSV 저장 완료: {output_path}")
    except Exception as e:
        print(f"측정 결과 CSV 저장 실패: {e}")


def calculate_area_histogram(measurements, bins=10):
    """
    면적 히스토그램 데이터 생성

    Args:
        measurements (list): 측정 결과 리스트
        bins (int): 히스토그램 구간 수

    Returns:
        tuple: (히스토그램 값, 구간 경계)
    """
    if not measurements:
        return [], []

    areas = [m['area_mm2'] for m in measurements]
    hist, bin_edges = np.histogram(areas, bins=bins)

    return hist, bin_edges


if __name__ == '__main__':
    # 테스트
    from config import Config
    from image_processor import load_image, preprocess_image
    from detector import detect_solder_paste

    # 설정 로드
    config = Config()

    # 테스트 이미지 로드
    test_image_path = '../opencv-course-master/Resources/Photos/cats.jpg'

    img = load_image(test_image_path)
    if img is not None:
        # 전처리
        hsv, resized = preprocess_image(img, config)

        # 검출
        contours, mask = detect_solder_paste(hsv, config)

        print(f"검출된 윤곽선 수: {len(contours)}")

        if contours:
            # 측정
            measurements = measure_all_contours(contours, config.PIXELS_PER_MM)

            # 통계
            statistics = calculate_statistics(measurements)

            # 리포트 생성
            report = format_measurement_report(measurements, statistics)
            print("\n" + report)

            # JSON 저장 테스트
            save_measurements_to_json(measurements, statistics, 'test_results.json')

            # CSV 저장 테스트
            save_measurements_to_csv(measurements, 'test_results.csv')
