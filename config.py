"""
설정 관리 모듈
솔더 페이스트 검출 및 측정을 위한 파라미터 정의
"""

import numpy as np
import json
import os


class Config:
    """솔더 페이스트 검출 설정 클래스"""

    def __init__(self, config_path=None):
        """
        설정 초기화

        Args:
            config_path (str, optional): 설정 파일 경로. 없으면 기본값 사용
        """
        # ===== 3D 높이 맵 모드 설정 =====
        # True: 3D 높이 맵 이미지 (RGB = 높이 정보)
        # False: 일반 2D 색상 이미지
        self.HEIGHT_MAP_MODE = True

        # 높이 임계값 (Blue channel 값, 0-255)
        # Blue 값이 이 값 이상이면 솔더 페이스트로 인식
        # 파란색 = 높음, 초록색 = 중간, 빨강 = 낮음
        self.HEIGHT_THRESHOLD_MIN = 100  # 최소 높이
        self.HEIGHT_THRESHOLD_MAX = 255  # 최대 높이

        # 검출 방법 선택 (HEIGHT_MAP_MODE=True일 때)
        # 'ensemble_v4': v4 앙상블 (Normalized RGB + 노이즈 제거, 권장)
        # 'ensemble': v3 앙상블 투표 (하위 호환)
        # 'norm_rgb': Normalized RGB 단독 (산업용 AOI 표준)
        # 'otsu_blue': Blue 채널 Otsu 반전
        # 'clahe_blue': CLAHE + Blue Otsu 반전
        # 'adaptive': 적응형 임계값
        # 'legacy': 기존 방식 (단순 Blue 채널 범위)
        self.DETECTION_METHOD = 'ensemble_v4'

        # v4 노이즈 제거 설정
        # 'bilateral': 엣지 보존 노이즈 제거 (빠름, 권장)
        # 'nlm': Non-local Means (강력, 느림)
        # 'none': 노이즈 제거 안 함
        self.DENOISE_METHOD = 'bilateral'

        # Normalized RGB 임계값 (norm_rgb 방법 사용 시)
        self.NORM_RGB_THRESHOLD = 0.40

        # HSV 색상 범위 (2D 색상 모드용, HEIGHT_MAP_MODE=False일 때 사용)
        # 주의: 실제 이미지에 맞춰 조정 필요
        self.LOWER_HSV = np.array([0, 0, 180])
        self.UPPER_HSV = np.array([20, 50, 255])

        # 면적 필터 (픽셀²)
        self.MIN_AREA = 50   # 작은 노이즈 제거용
        self.MAX_AREA = 100000  # 비정상적으로 큰 영역 제외

        # 원형도 필터 (0.0 ~ 1.0, 1.0은 완전한 원)
        self.MIN_CIRCULARITY = 0.3  # 3D 높이 맵에서는 더 낮게 설정
        self.MAX_CIRCULARITY = 1.0

        # 형태학적 연산 커널 크기
        self.MORPH_KERNEL_SIZE = (5, 5)

        # 가우시안 블러 커널 크기
        self.BLUR_KERNEL_SIZE = (5, 5)

        # 이미지 리사이징 스케일 (처리 속도 향상)
        self.RESIZE_SCALE = 1.0  # 1.0 = 원본 크기

        # 캘리브레이션: 픽셀당 mm 비율
        # 주의: 실제 측정을 통해 설정 필요
        self.PIXELS_PER_MM = 10.0  # 예: 10 pixels = 1 mm

        # 시각화 설정
        self.CONTOUR_COLOR = (0, 255, 0)  # 녹색 (BGR)
        self.CENTER_COLOR = (0, 0, 255)   # 빨간색 (BGR)
        self.TEXT_COLOR = (255, 255, 255) # 흰색 (BGR)
        self.CONTOUR_THICKNESS = 2
        self.CENTER_RADIUS = 5
        self.FONT_SCALE = 0.5
        self.FONT_THICKNESS = 1

        # 외부 설정 파일이 있으면 로드
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)

    def load_from_file(self, config_path):
        """
        JSON 파일에서 설정 로드

        Args:
            config_path (str): 설정 파일 경로
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            # 높이 맵 모드
            if 'height_map_mode' in config_dict:
                self.HEIGHT_MAP_MODE = config_dict['height_map_mode']

            # 검출 방법
            if 'detection_method' in config_dict:
                self.DETECTION_METHOD = config_dict['detection_method']

            # 높이 임계값
            if 'height_threshold_min' in config_dict:
                self.HEIGHT_THRESHOLD_MIN = config_dict['height_threshold_min']
            if 'height_threshold_max' in config_dict:
                self.HEIGHT_THRESHOLD_MAX = config_dict['height_threshold_max']

            # HSV 범위
            if 'lower_hsv' in config_dict:
                self.LOWER_HSV = np.array(config_dict['lower_hsv'])
            if 'upper_hsv' in config_dict:
                self.UPPER_HSV = np.array(config_dict['upper_hsv'])

            # 면적 필터
            if 'min_area' in config_dict:
                self.MIN_AREA = config_dict['min_area']
            if 'max_area' in config_dict:
                self.MAX_AREA = config_dict['max_area']

            # 원형도 필터
            if 'min_circularity' in config_dict:
                self.MIN_CIRCULARITY = config_dict['min_circularity']
            if 'max_circularity' in config_dict:
                self.MAX_CIRCULARITY = config_dict['max_circularity']

            # 캘리브레이션
            if 'pixels_per_mm' in config_dict:
                self.PIXELS_PER_MM = config_dict['pixels_per_mm']

            # 기타 설정
            if 'resize_scale' in config_dict:
                self.RESIZE_SCALE = config_dict['resize_scale']

            print(f"설정 파일 로드 완료: {config_path}")

        except Exception as e:
            print(f"설정 파일 로드 실패: {e}")
            print("기본 설정을 사용합니다.")

    def save_to_file(self, config_path):
        """
        현재 설정을 JSON 파일로 저장

        Args:
            config_path (str): 저장할 파일 경로
        """
        config_dict = {
            'height_map_mode': self.HEIGHT_MAP_MODE,
            'detection_method': self.DETECTION_METHOD,
            'height_threshold_min': self.HEIGHT_THRESHOLD_MIN,
            'height_threshold_max': self.HEIGHT_THRESHOLD_MAX,
            'lower_hsv': self.LOWER_HSV.tolist(),
            'upper_hsv': self.UPPER_HSV.tolist(),
            'min_area': self.MIN_AREA,
            'max_area': self.MAX_AREA,
            'min_circularity': self.MIN_CIRCULARITY,
            'max_circularity': self.MAX_CIRCULARITY,
            'morph_kernel_size': self.MORPH_KERNEL_SIZE,
            'blur_kernel_size': self.BLUR_KERNEL_SIZE,
            'resize_scale': self.RESIZE_SCALE,
            'pixels_per_mm': self.PIXELS_PER_MM
        }

        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
            print(f"설정 파일 저장 완료: {config_path}")

        except Exception as e:
            print(f"설정 파일 저장 실패: {e}")

    def print_config(self):
        """현재 설정 출력"""
        print("\n" + "="*50)
        print("현재 설정")
        print("="*50)
        print(f"3D 높이 맵 모드: {'ON' if self.HEIGHT_MAP_MODE else 'OFF'}")
        if self.HEIGHT_MAP_MODE:
            print(f"높이 임계값: {self.HEIGHT_THRESHOLD_MIN} ~ {self.HEIGHT_THRESHOLD_MAX}")
        else:
            print(f"HSV 범위: {self.LOWER_HSV} ~ {self.UPPER_HSV}")
        print(f"면적 필터: {self.MIN_AREA} ~ {self.MAX_AREA} pixels²")
        print(f"원형도 필터: {self.MIN_CIRCULARITY} ~ {self.MAX_CIRCULARITY}")
        print(f"캘리브레이션: {self.PIXELS_PER_MM} pixels/mm")
        print(f"리사이징 스케일: {self.RESIZE_SCALE}")
        print("="*50 + "\n")


# 기본 설정 인스턴스
default_config = Config()


if __name__ == '__main__':
    # 테스트: 기본 설정 출력
    config = Config()
    config.print_config()

    # 테스트: 설정 저장
    config.save_to_file('config_default.json')

    # 테스트: 설정 로드
    config2 = Config('config_default.json')
    config2.print_config()
