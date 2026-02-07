"""
시각화 모듈
검출 결과 시각화, 윤곽선 및 측정 정보 표시
"""

import cv2 as cv
import numpy as np


def visualize_results(img, measurements, config):
    """
    검출 결과 시각화

    Args:
        img (numpy.ndarray): 원본 이미지 (BGR)
        measurements (list): 측정 결과 리스트
        config: 설정 객체

    Returns:
        numpy.ndarray: 시각화된 이미지
    """
    output = img.copy()

    for m in measurements:
        # 윤곽선 그리기
        cv.drawContours(output, [m['contour']], 0,
                       config.CONTOUR_COLOR, config.CONTOUR_THICKNESS)

        # 중심점 표시
        cv.circle(output, m['center'], config.CENTER_RADIUS,
                 config.CENTER_COLOR, -1)

        # 면적 텍스트 표시
        text = f"{m['area_mm2']:.2f} mm2"
        text_pos = (m['center'][0] + 10, m['center'][1] - 10)

        # 텍스트 배경 (가독성 향상)
        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX,
                                   config.FONT_SCALE, config.FONT_THICKNESS)[0]
        cv.rectangle(output,
                    (text_pos[0] - 2, text_pos[1] - text_size[1] - 2),
                    (text_pos[0] + text_size[0] + 2, text_pos[1] + 2),
                    (0, 0, 0), -1)

        # 텍스트 표시
        cv.putText(output, text, text_pos,
                  cv.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                  config.TEXT_COLOR, config.FONT_THICKNESS)

        # ID 표시
        id_text = f"#{m['id']}"
        id_pos = (m['center'][0] - 20, m['center'][1] + 5)
        cv.putText(output, id_text, id_pos,
                  cv.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                  (255, 255, 0), config.FONT_THICKNESS)

    return output


def draw_bounding_boxes(img, measurements, config):
    """
    외접 사각형 그리기

    Args:
        img (numpy.ndarray): 원본 이미지
        measurements (list): 측정 결과 리스트
        config: 설정 객체

    Returns:
        numpy.ndarray: 시각화된 이미지
    """
    output = img.copy()

    for m in measurements:
        x, y, w, h = m['bounding_rect']
        cv.rectangle(output, (x, y), (x+w, y+h),
                    (255, 0, 0), 2)  # 파란색

    return output


def create_comparison_view(original, mask, detected, measurements=None):
    """
    원본, 마스크, 검출 결과를 한 화면에 표시

    Args:
        original (numpy.ndarray): 원본 이미지
        mask (numpy.ndarray): 이진 마스크
        detected (numpy.ndarray): 검출 결과 이미지
        measurements (list, optional): 측정 결과 (통계 표시용)

    Returns:
        numpy.ndarray: 3개 이미지를 병합한 결과
    """
    # 마스크를 BGR로 변환
    mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    # 이미지 크기 통일
    h, w = original.shape[:2]
    mask_bgr = cv.resize(mask_bgr, (w, h))
    detected = cv.resize(detected, (w, h))

    # 레이블 추가
    label_height = 30
    labels = ['Original', 'Mask', 'Detected']
    images = [original, mask_bgr, detected]

    labeled_images = []
    for img, label in zip(images, labels):
        # 레이블 영역 생성
        label_area = np.zeros((label_height, w, 3), dtype=np.uint8)
        cv.putText(label_area, label, (10, 20),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 이미지와 레이블 합치기
        labeled_img = np.vstack([label_area, img])
        labeled_images.append(labeled_img)

    # 가로로 병합
    comparison = np.hstack(labeled_images)

    # 측정 통계 추가 (하단)
    if measurements:
        from measurement import calculate_statistics
        stats = calculate_statistics(measurements)

        stats_height = 60
        stats_area = np.zeros((stats_height, comparison.shape[1], 3), dtype=np.uint8)

        stats_text = [
            f"Count: {stats['count']}",
            f"Total: {stats['total_area_mm2']:.2f} mm2",
            f"Mean: {stats['mean_area_mm2']:.2f} mm2",
            f"Range: {stats['min_area_mm2']:.2f} ~ {stats['max_area_mm2']:.2f} mm2"
        ]

        y_offset = 20
        for i, text in enumerate(stats_text):
            x_offset = 10 + (i * (comparison.shape[1] // 4))
            cv.putText(stats_area, text, (x_offset, y_offset),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        comparison = np.vstack([comparison, stats_area])

    return comparison


def display_image(window_name, img, wait_key=True):
    """
    이미지 표시

    Args:
        window_name (str): 윈도우 이름
        img (numpy.ndarray): 표시할 이미지
        wait_key (bool): 키 입력 대기 여부
    """
    cv.imshow(window_name, img)
    if wait_key:
        cv.waitKey(0)
        cv.destroyAllWindows()


def save_image(img, output_path):
    """
    이미지 저장

    Args:
        img (numpy.ndarray): 저장할 이미지
        output_path (str): 출력 파일 경로
    """
    try:
        cv.imwrite(output_path, img)
        print(f"이미지 저장 완료: {output_path}")
    except Exception as e:
        print(f"이미지 저장 실패: {e}")


def create_overlay(original, mask, alpha=0.3):
    """
    마스크를 원본 이미지 위에 오버레이

    Args:
        original (numpy.ndarray): 원본 이미지 (BGR)
        mask (numpy.ndarray): 이진 마스크
        alpha (float): 투명도 (0.0 ~ 1.0)

    Returns:
        numpy.ndarray: 오버레이된 이미지
    """
    # 마스크를 컬러로 변환 (녹색)
    mask_colored = np.zeros_like(original)
    mask_colored[:, :, 1] = mask  # 녹색 채널

    # 오버레이
    overlay = cv.addWeighted(original, 1-alpha, mask_colored, alpha, 0)

    return overlay


def draw_measurement_table(img, measurements, position=(10, 30)):
    """
    이미지에 측정 결과 테이블 그리기

    Args:
        img (numpy.ndarray): 원본 이미지
        measurements (list): 측정 결과 리스트
        position (tuple): 테이블 시작 위치 (x, y)

    Returns:
        numpy.ndarray: 테이블이 그려진 이미지
    """
    output = img.copy()
    x, y = position

    # 테이블 헤더
    header = "ID | Area(mm2) | Center"
    cv.putText(output, header, (x, y),
              cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 구분선
    y += 5
    cv.line(output, (x, y), (x + 250, y), (255, 255, 255), 1)

    # 데이터 행
    y += 15
    for m in measurements[:10]:  # 최대 10개만 표시
        row = f"{m['id']:2d} | {m['area_mm2']:9.2f} | {m['center']}"
        cv.putText(output, row, (x, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        y += 15

    if len(measurements) > 10:
        cv.putText(output, f"... and {len(measurements)-10} more", (x, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    return output


if __name__ == '__main__':
    # 테스트
    from config import Config
    from image_processor import load_image, preprocess_image
    from detector import detect_solder_paste
    from measurement import measure_all_contours

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

        if contours:
            # 측정
            measurements = measure_all_contours(contours, config.PIXELS_PER_MM)

            # 시각화
            result = visualize_results(resized, measurements, config)
            comparison = create_comparison_view(resized, mask, result, measurements)

            # 표시
            cv.imshow('Result', result)
            cv.imshow('Comparison', comparison)
            cv.waitKey(0)
            cv.destroyAllWindows()

            # 저장
            save_image(result, 'test_result.jpg')
            save_image(comparison, 'test_comparison.jpg')
