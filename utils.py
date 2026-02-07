"""
유틸리티 함수 모듈
파일 처리, 배치 작업, 일반 헬퍼 함수
"""

import os
import glob
import cv2 as cv
from datetime import datetime


def get_image_files(directory, extensions=None):
    """
    디렉토리에서 이미지 파일 목록 가져오기

    Args:
        directory (str): 검색할 디렉토리 경로
        extensions (list, optional): 파일 확장자 리스트 (기본: jpg, jpeg, png, bmp)

    Returns:
        list: 이미지 파일 경로 리스트
    """
    if extensions is None:
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']

    image_files = []
    for ext in extensions:
        pattern = os.path.join(directory, f'*.{ext}')
        image_files.extend(glob.glob(pattern))
        # 대문자 확장자도 포함
        pattern = os.path.join(directory, f'*.{ext.upper()}')
        image_files.extend(glob.glob(pattern))

    return sorted(image_files)


def create_output_directory(base_dir='output'):
    """
    타임스탬프가 포함된 출력 디렉토리 생성

    Args:
        base_dir (str): 기본 디렉토리 이름

    Returns:
        str: 생성된 디렉토리 경로
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_dir}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리 생성: {output_dir}")

    return output_dir


def get_output_path(input_path, output_dir, suffix='_result', extension=None):
    """
    입력 파일 경로로부터 출력 파일 경로 생성

    Args:
        input_path (str): 입력 파일 경로
        output_dir (str): 출력 디렉토리
        suffix (str): 파일명 접미사
        extension (str, optional): 출력 파일 확장자 (없으면 입력과 동일)

    Returns:
        str: 출력 파일 경로
    """
    # 파일명과 확장자 분리
    basename = os.path.basename(input_path)
    filename, ext = os.path.splitext(basename)

    # 출력 확장자 결정
    if extension:
        output_ext = extension if extension.startswith('.') else f'.{extension}'
    else:
        output_ext = ext

    # 출력 파일명 생성
    output_filename = f"{filename}{suffix}{output_ext}"
    output_path = os.path.join(output_dir, output_filename)

    return output_path


def ensure_directory_exists(directory):
    """
    디렉토리가 존재하지 않으면 생성

    Args:
        directory (str): 디렉토리 경로
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"디렉토리 생성: {directory}")


def format_timestamp():
    """
    현재 시간을 포맷된 문자열로 반환

    Returns:
        str: 타임스탬프 문자열 (YYYY-MM-DD HH:MM:SS)
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    터미널에 진행률 표시줄 출력

    Args:
        iteration (int): 현재 반복 횟수
        total (int): 총 반복 횟수
        prefix (str): 접두사
        suffix (str): 접미사
        length (int): 진행률 바 길이
        fill (str): 채움 문자
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)

    # 완료 시 줄바꿈
    if iteration == total:
        print()


def resize_to_fit_screen(image, max_width=1920, max_height=1080):
    """
    화면에 맞게 이미지 리사이징

    Args:
        image (numpy.ndarray): 입력 이미지
        max_width (int): 최대 너비
        max_height (int): 최대 높이

    Returns:
        numpy.ndarray: 리사이징된 이미지
    """
    h, w = image.shape[:2]

    # 스케일 계산
    scale_w = max_width / w if w > max_width else 1.0
    scale_h = max_height / h if h > max_height else 1.0
    scale = min(scale_w, scale_h)

    # 리사이징 필요 여부 확인
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
        return resized
    else:
        return image


def validate_image_file(image_path):
    """
    이미지 파일 유효성 검사

    Args:
        image_path (str): 이미지 파일 경로

    Returns:
        bool: 유효하면 True, 아니면 False
    """
    # 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"파일이 존재하지 않습니다: {image_path}")
        return False

    # 이미지 로드 시도
    img = cv.imread(image_path)
    if img is None:
        print(f"유효하지 않은 이미지 파일: {image_path}")
        return False

    return True


def create_summary_report(results, output_path):
    """
    배치 처리 결과 요약 리포트 생성

    Args:
        results (list): 각 이미지의 처리 결과 리스트
        output_path (str): 리포트 파일 경로
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("솔더 페이스트 검사 배치 처리 리포트\n")
            f.write("="*80 + "\n")
            f.write(f"생성 시간: {format_timestamp()}\n")
            f.write(f"처리 이미지 수: {len(results)}\n\n")

            # 개별 결과
            for result in results:
                f.write(f"\n파일: {result['filename']}\n")
                f.write(f"  검출 개수: {result['count']}\n")
                f.write(f"  총 면적: {result['total_area']:.3f} mm²\n")
                f.write(f"  평균 면적: {result['mean_area']:.3f} mm²\n")
                f.write(f"  처리 시간: {result['processing_time']:.2f}초\n")

            # 전체 통계
            total_detections = sum(r['count'] for r in results)
            total_time = sum(r['processing_time'] for r in results)

            f.write("\n" + "="*80 + "\n")
            f.write("전체 통계\n")
            f.write("-"*80 + "\n")
            f.write(f"총 검출 개수: {total_detections}\n")
            f.write(f"평균 검출 개수/이미지: {total_detections/len(results):.1f}\n")
            f.write(f"총 처리 시간: {total_time:.2f}초\n")
            f.write(f"평균 처리 시간/이미지: {total_time/len(results):.2f}초\n")
            f.write("="*80 + "\n")

        print(f"요약 리포트 저장 완료: {output_path}")

    except Exception as e:
        print(f"요약 리포트 저장 실패: {e}")


def log_message(message, log_file='process.log'):
    """
    로그 메시지 파일에 기록

    Args:
        message (str): 로그 메시지
        log_file (str): 로그 파일 경로
    """
    timestamp = format_timestamp()
    log_entry = f"[{timestamp}] {message}\n"

    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"로그 기록 실패: {e}")


if __name__ == '__main__':
    # 테스트
    print("유틸리티 함수 테스트")

    # 타임스탬프
    print(f"현재 시간: {format_timestamp()}")

    # 진행률 바
    print("\n진행률 바 테스트:")
    import time
    for i in range(101):
        print_progress_bar(i, 100, prefix='처리 중:', suffix='완료')
        time.sleep(0.02)

    # 출력 디렉토리 생성 테스트
    output_dir = create_output_directory('test_output')
    print(f"생성된 디렉토리: {output_dir}")
