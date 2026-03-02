"""
솔더 페이스트 면적 측정 프로그램 - 메인 실행 파일
AOI 이미지에서 솔더 페이스트를 검출하고 면적을 측정합니다.
"""

import cv2 as cv
import argparse
import os
import time

from config import Config
from image_processor import load_image, preprocess_image
from detector import detect_solder_paste
from measurement import (measure_all_contours, calculate_statistics,
                        format_measurement_report, save_measurements_to_json,
                        save_measurements_to_csv)
from visualizer import visualize_results, create_comparison_view, save_image
from calibration import load_calibration, save_calibration, CalibrationTool
from utils import (get_image_files, create_output_directory, get_output_path,
                  print_progress_bar, create_summary_report, log_message)


def process_single_image(image_path, config, output_dir=None, show_results=True):
    """
    단일 이미지 처리

    Args:
        image_path (str): 이미지 파일 경로
        config: 설정 객체
        output_dir (str, optional): 출력 디렉토리
        show_results (bool): 결과 표시 여부

    Returns:
        dict: 처리 결과
    """
    start_time = time.time()

    # 1. 이미지 로드
    print(f"\n처리 중: {os.path.basename(image_path)}")
    img = load_image(image_path)
    if img is None:
        return None

    # 2. 전처리
    hsv, resized = preprocess_image(img, config)

    # 3. 검출
    # 3D 높이 맵 모드: 원본 이미지 사용 (Blue channel = 높이)
    # 2D 색상 모드: HSV 이미지 사용
    detect_input = resized if config.HEIGHT_MAP_MODE else hsv
    contours, mask = detect_solder_paste(detect_input, config)
    print(f"  검출된 윤곽선 수: {len(contours)}")

    if not contours:
        print("  검출된 솔더 페이스트가 없습니다")
        processing_time = time.time() - start_time
        return {
            'filename': os.path.basename(image_path),
            'count': 0,
            'total_area': 0.0,
            'mean_area': 0.0,
            'processing_time': processing_time
        }

    # 4. 측정
    measurements = measure_all_contours(contours, config.PIXELS_PER_MM)

    # 5. 통계
    statistics = calculate_statistics(measurements)

    print(f"  총 면적: {statistics['total_area_mm2']:.3f} mm²")
    print(f"  평균 면적: {statistics['mean_area_mm2']:.3f} mm²")

    # 6. 시각화
    result_img = visualize_results(resized, measurements, config)
    comparison_img = create_comparison_view(resized, mask, result_img, measurements)

    # 7. 결과 저장
    if output_dir:
        # 결과 이미지 저장
        result_path = get_output_path(image_path, output_dir, '_result')
        save_image(result_img, result_path)

        comparison_path = get_output_path(image_path, output_dir, '_comparison')
        save_image(comparison_img, comparison_path)

        # 측정 데이터 저장 (JSON)
        json_path = get_output_path(image_path, output_dir, '_measurements', '.json')
        save_measurements_to_json(measurements, statistics, json_path)

        # 측정 데이터 저장 (CSV)
        csv_path = get_output_path(image_path, output_dir, '_measurements', '.csv')
        save_measurements_to_csv(measurements, csv_path)

        # 텍스트 리포트 저장
        report_path = get_output_path(image_path, output_dir, '_report', '.txt')
        report = format_measurement_report(measurements, statistics)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

    # 8. 결과 표시
    if show_results:
        cv.imshow('Result', result_img)
        cv.imshow('Comparison', comparison_img)
        print("\n결과 확인 후 아무 키나 누르세요...")
        cv.waitKey(0)
        cv.destroyAllWindows()

    processing_time = time.time() - start_time
    print(f"  처리 시간: {processing_time:.2f}초")

    return {
        'filename': os.path.basename(image_path),
        'count': statistics['count'],
        'total_area': statistics['total_area_mm2'],
        'mean_area': statistics['mean_area_mm2'],
        'processing_time': processing_time
    }


def process_batch(image_dir, config, output_dir=None):
    """
    배치 처리 (여러 이미지)

    Args:
        image_dir (str): 이미지 디렉토리
        config: 설정 객체
        output_dir (str, optional): 출력 디렉토리

    Returns:
        list: 처리 결과 리스트
    """
    # 이미지 파일 목록 가져오기
    image_files = get_image_files(image_dir)

    if not image_files:
        print(f"이미지 파일을 찾을 수 없습니다: {image_dir}")
        return []

    print(f"\n총 {len(image_files)}개 이미지 처리 시작")

    # 출력 디렉토리 생성
    if output_dir is None:
        output_dir = create_output_directory('batch_output')

    results = []

    for i, image_path in enumerate(image_files, 1):
        print_progress_bar(i-1, len(image_files), prefix='진행:', suffix=f'{i}/{len(image_files)}')

        # 이미지 처리
        result = process_single_image(image_path, config, output_dir, show_results=False)

        if result:
            results.append(result)
            log_message(f"처리 완료: {result['filename']} - {result['count']}개 검출",
                       os.path.join(output_dir, 'process.log'))

    print_progress_bar(len(image_files), len(image_files), prefix='진행:', suffix='완료')

    # 요약 리포트 생성
    if results:
        summary_path = os.path.join(output_dir, 'summary_report.txt')
        create_summary_report(results, summary_path)

    print(f"\n배치 처리 완료: {len(results)}개 이미지 처리됨")
    print(f"결과 저장 위치: {output_dir}")

    return results


def run_calibration_mode(config):
    """
    캘리브레이션 모드 실행

    Args:
        config: 설정 객체

    Returns:
        bool: 캘리브레이션 성공 여부
    """
    print("\n" + "="*60)
    print("카메라 캘리브레이션 모드")
    print("="*60)

    image_path = input("\n캘리브레이션 이미지 경로: ").strip()

    if not os.path.exists(image_path):
        print(f"파일이 존재하지 않습니다: {image_path}")
        return False

    tool = CalibrationTool()
    pixels_per_mm = tool.calibrate_interactive(image_path)

    if pixels_per_mm:
        # 설정 업데이트
        config.PIXELS_PER_MM = pixels_per_mm

        # 저장 여부 확인
        save_choice = input("\n캘리브레이션을 저장하시겠습니까? (y/n): ").strip().lower()

        if save_choice == 'y':
            # 캘리브레이션 저장
            save_calibration(pixels_per_mm, 'calibration.json')

            # 설정 저장
            config.save_to_file('config.json')

        return True
    else:
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='솔더 페이스트 면적 측정 프로그램',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  단일 이미지 처리:
    python main.py -i image.jpg

  배치 처리:
    python main.py -b image_folder/

  캘리브레이션:
    python main.py --calibrate

  설정 파일 사용:
    python main.py -i image.jpg --config config.json
        """
    )

    parser.add_argument('-i', '--image', type=str,
                       help='단일 이미지 파일 경로')
    parser.add_argument('-b', '--batch', type=str,
                       help='배치 처리할 이미지 디렉토리')
    parser.add_argument('-o', '--output', type=str,
                       help='출력 디렉토리 경로')
    parser.add_argument('--config', type=str, default='config.json',
                       help='설정 파일 경로 (기본: config.json)')
    parser.add_argument('--calibrate', action='store_true',
                       help='캘리브레이션 모드 실행')
    parser.add_argument('--no-display', action='store_true',
                       help='결과 화면에 표시하지 않음')

    args = parser.parse_args()

    # 설정 로드
    if os.path.exists(args.config):
        config = Config(args.config)
        print(f"설정 파일 로드: {args.config}")
    else:
        config = Config()
        print("기본 설정 사용")

    # 캘리브레이션 데이터 로드
    if os.path.exists('calibration.json'):
        pixels_per_mm = load_calibration('calibration.json')
        if pixels_per_mm:
            config.PIXELS_PER_MM = pixels_per_mm

    config.print_config()

    # 모드 선택
    if args.calibrate:
        # 캘리브레이션 모드
        run_calibration_mode(config)

    elif args.image:
        # 단일 이미지 처리
        show_results = not args.no_display
        result = process_single_image(args.image, config, args.output, show_results)

        if result:
            print("\n처리 완료!")

    elif args.batch:
        # 배치 처리
        results = process_batch(args.batch, config, args.output)

        if results:
            print("\n배치 처리 완료!")

    else:
        # 인수가 없으면 도움말 표시
        parser.print_help()


if __name__ == '__main__':
    main()
