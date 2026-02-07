"""
카메라 캘리브레이션 모듈
픽셀-mm 변환 계수 계산
"""

import cv2 as cv
import numpy as np
import json
import os


class CalibrationTool:
    """캘리브레이션 도구 클래스"""

    def __init__(self):
        self.points = []
        self.image = None
        self.window_name = "Calibration"

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 콜백"""
        if event == cv.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                print(f"점 {len(self.points)} 선택: ({x}, {y})")

                # 점 표시
                cv.circle(self.image, (x, y), 5, (0, 0, 255), -1)
                cv.imshow(self.window_name, self.image)

                if len(self.points) == 2:
                    # 두 점 사이 선 그리기
                    cv.line(self.image, self.points[0], self.points[1],
                           (0, 255, 0), 2)
                    cv.imshow(self.window_name, self.image)

                    # 거리 계산
                    pixel_distance = self.calculate_pixel_distance()
                    print(f"픽셀 거리: {pixel_distance:.2f} pixels")

    def calculate_pixel_distance(self):
        """두 점 사이의 픽셀 거리 계산"""
        if len(self.points) != 2:
            return 0.0

        p1, p2 = self.points
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return distance

    def calibrate_interactive(self, image_path):
        """
        대화형 캘리브레이션

        Args:
            image_path (str): 캘리브레이션 이미지 경로

        Returns:
            float: 픽셀당 mm 비율 (pixels/mm)
        """
        # 이미지 로드
        self.image = cv.imread(image_path)
        if self.image is None:
            print(f"이미지 로드 실패: {image_path}")
            return None

        # 윈도우 생성
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n캘리브레이션 시작")
        print("알려진 거리를 가진 두 점을 클릭하세요.")
        print("ESC 키를 눌러 종료")

        cv.imshow(self.window_name, self.image)

        while True:
            key = cv.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            if len(self.points) == 2:
                break

        cv.destroyAllWindows()

        # 두 점이 선택되었는지 확인
        if len(self.points) != 2:
            print("캘리브레이션 취소됨")
            return None

        # 실제 거리 입력
        pixel_distance = self.calculate_pixel_distance()
        print(f"\n픽셀 거리: {pixel_distance:.2f} pixels")

        try:
            real_distance_mm = float(input("실제 거리 (mm): "))

            if real_distance_mm <= 0:
                print("유효하지 않은 거리")
                return None

            # 픽셀당 mm 비율 계산
            pixels_per_mm = pixel_distance / real_distance_mm

            print(f"\n캘리브레이션 완료")
            print(f"픽셀당 mm 비율: {pixels_per_mm:.4f} pixels/mm")
            print(f"mm당 픽셀 비율: {1/pixels_per_mm:.4f} mm/pixel")

            return pixels_per_mm

        except ValueError:
            print("유효하지 않은 입력")
            return None


def calibrate_with_known_object(image, known_width_mm, known_height_mm=None):
    """
    알려진 크기의 객체를 사용한 자동 캘리브레이션

    Args:
        image (numpy.ndarray): 입력 이미지
        known_width_mm (float): 알려진 너비 (mm)
        known_height_mm (float, optional): 알려진 높이 (mm)

    Returns:
        tuple: (pixels_per_mm_x, pixels_per_mm_y) 또는 None
    """
    # 그레이스케일 변환
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 임계값 처리
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # 윤곽선 검출
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("객체를 찾을 수 없습니다")
        return None

    # 가장 큰 윤곽선 선택
    largest_contour = max(contours, key=cv.contourArea)

    # 외접 사각형
    x, y, w, h = cv.boundingRect(largest_contour)

    # 픽셀당 mm 비율 계산
    pixels_per_mm_x = w / known_width_mm

    if known_height_mm:
        pixels_per_mm_y = h / known_height_mm
        return (pixels_per_mm_x, pixels_per_mm_y)
    else:
        return (pixels_per_mm_x, pixels_per_mm_x)


def save_calibration(pixels_per_mm, output_path='calibration.json'):
    """
    캘리브레이션 결과 저장

    Args:
        pixels_per_mm (float): 픽셀당 mm 비율
        output_path (str): 출력 파일 경로
    """
    calibration_data = {
        'pixels_per_mm': pixels_per_mm,
        'mm_per_pixel': 1.0 / pixels_per_mm
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=4)
        print(f"캘리브레이션 저장 완료: {output_path}")
    except Exception as e:
        print(f"캘리브레이션 저장 실패: {e}")


def load_calibration(calibration_path='calibration.json'):
    """
    캘리브레이션 결과 로드

    Args:
        calibration_path (str): 캘리브레이션 파일 경로

    Returns:
        float: 픽셀당 mm 비율 또는 None
    """
    if not os.path.exists(calibration_path):
        print(f"캘리브레이션 파일이 없습니다: {calibration_path}")
        return None

    try:
        with open(calibration_path, 'r', encoding='utf-8') as f:
            calibration_data = json.load(f)
        pixels_per_mm = calibration_data['pixels_per_mm']
        print(f"캘리브레이션 로드 완료: {pixels_per_mm:.4f} pixels/mm")
        return pixels_per_mm
    except Exception as e:
        print(f"캘리브레이션 로드 실패: {e}")
        return None


def calculate_scale_factor(source_resolution, target_resolution):
    """
    해상도 변경에 따른 스케일 팩터 계산

    Args:
        source_resolution (tuple): 원본 해상도 (width, height)
        target_resolution (tuple): 대상 해상도 (width, height)

    Returns:
        float: 스케일 팩터
    """
    scale_x = target_resolution[0] / source_resolution[0]
    scale_y = target_resolution[1] / source_resolution[1]

    # 평균 스케일 사용
    scale_factor = (scale_x + scale_y) / 2.0

    return scale_factor


def adjust_calibration_for_resize(pixels_per_mm, resize_scale):
    """
    이미지 리사이징에 따른 캘리브레이션 조정

    Args:
        pixels_per_mm (float): 원본 픽셀당 mm 비율
        resize_scale (float): 리사이징 스케일 (0.5 = 50%)

    Returns:
        float: 조정된 픽셀당 mm 비율
    """
    adjusted_pixels_per_mm = pixels_per_mm * resize_scale
    return adjusted_pixels_per_mm


if __name__ == '__main__':
    # 테스트: 대화형 캘리브레이션
    print("캘리브레이션 도구 테스트")
    print("1. 대화형 캘리브레이션")
    print("2. 종료")

    choice = input("선택: ")

    if choice == '1':
        # 테스트 이미지 경로 입력
        image_path = input("캘리브레이션 이미지 경로: ")

        if os.path.exists(image_path):
            tool = CalibrationTool()
            pixels_per_mm = tool.calibrate_interactive(image_path)

            if pixels_per_mm:
                # 저장 여부 확인
                save_choice = input("\n캘리브레이션을 저장하시겠습니까? (y/n): ")
                if save_choice.lower() == 'y':
                    save_calibration(pixels_per_mm)
        else:
            print(f"파일이 존재하지 않습니다: {image_path}")
