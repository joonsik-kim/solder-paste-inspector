# 솔더 페이스트 면적 측정 프로그램

OpenCV를 사용하여 AOI/SPI 3D 높이 맵 이미지에서 솔더 페이스트를 검출하고 면적을 측정하는 프로그램입니다.

## 🎯 이미지 타입

### **3D 높이 맵 (Height Map)** ⭐ 현재 지원
AOI/SPI 광학 검사에서 생성된 RGB 높이 정보 이미지:
- **파란색 (Blue)**: 높은 경사/높이
- **초록색 (Green)**: 중간 경사/높이
- **빨간색 (Red)**: 낮은 경사/높이
- ROI(Region of Interest)로 잘린 솔더 페이스트 영역

### 일반 2D 색상 이미지
기존 RGB 카메라로 촬영한 일반 이미지 (설정으로 전환 가능)

## 📋 기능

### Phase 1 (현재)
- ✅ **3D 높이 맵에서 솔더 페이스트 검출** (Blue channel 기반)
- ✅ 검출된 영역의 **면적 측정** (픽셀 → mm² 변환)
- ✅ 측정 결과 시각화 및 저장
- ✅ 배치 처리 지원 (대량 이미지 처리)
- ✅ 카메라 캘리브레이션 도구
- ✅ 2D 색상 모드 / 3D 높이 맵 모드 전환

### Phase 2 (추후 예정)
- 🔲 **3D 부피 측정** (면적 × 평균 높이)
- 🔲 실장 필렛 검사
- 🔲 품질 판정 기준 적용

## 🚀 설치

### 1. Python 환경 준비
Python 3.7 이상 필요

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

또는 개별 설치:
```bash
pip install opencv-python numpy
```

## 📖 사용법

### 기본 사용

#### 1. 캘리브레이션 (처음 1회)
```bash
python main.py --calibrate
```
- 알려진 크기의 기준 이미지 로드
- 두 점을 클릭하여 거리 측정
- 실제 거리(mm) 입력
- 캘리브레이션 데이터 자동 저장

#### 2. 단일 이미지 처리
```bash
python main.py -i test_image.jpg
```

결과:
- 검출 결과 화면 표시
- 측정 데이터 출력

#### 3. 결과 저장
```bash
python main.py -i test_image.jpg -o output/
```

출력 파일:
- `*_result.jpg` - 검출 결과 이미지
- `*_comparison.jpg` - 원본/마스크/결과 비교
- `*_measurements.json` - 측정 데이터 (JSON)
- `*_measurements.csv` - 측정 데이터 (CSV)
- `*_report.txt` - 텍스트 리포트

#### 4. 배치 처리 (여러 이미지)
```bash
python main.py -b image_folder/ -o batch_output/
```

추가 생성:
- `summary_report.txt` - 전체 요약 리포트
- `process.log` - 처리 로그

### 고급 사용

#### 사용자 정의 설정 파일
```bash
python main.py -i test.jpg --config my_config.json
```

#### 화면 표시 없이 처리
```bash
python main.py -i test.jpg -o output/ --no-display
```

## ⚙️ 설정

### config.json 편집

```json
{
  "height_map_mode": true,
  "height_threshold_min": 100,
  "height_threshold_max": 255,
  "min_area": 50,
  "max_area": 100000,
  "min_circularity": 0.3,
  "max_circularity": 1.0,
  "pixels_per_mm": 10.0,
  "resize_scale": 1.0
}
```

**주요 파라미터**:

**3D 높이 맵 모드** (기본):
- `height_map_mode`: true (3D 높이 맵) / false (2D 색상)
- `height_threshold_min`, `height_threshold_max`: Blue channel 높이 임계값 (0-255)
  - 100-255: 중간~높은 높이만 검출
  - 50-255: 낮은 높이도 포함

**2D 색상 모드** (height_map_mode: false):
- `lower_hsv`, `upper_hsv`: HSV 색상 범위

**공통**:
- `min_area`, `max_area`: 면적 필터 (픽셀²)
- `min_circularity`, `max_circularity`: 원형도 필터 (0.0 ~ 1.0)
- `pixels_per_mm`: 캘리브레이션 값
- `resize_scale`: 이미지 리사이징 비율

### 높이 임계값 조정 (3D 높이 맵)

실제 이미지로 높이 임계값 조정 필요:

1. 테스트 이미지로 실행
2. 검출 결과 확인
3. `config.json`의 `height_threshold_min` 조정
4. 재실행하여 검증

**조정 가이드**:
```
검출 안됨 (0개)     → height_threshold_min 낮추기 (예: 100 → 50)
너무 많이 검출됨    → height_threshold_min 높이기 (예: 100 → 150)
배경까지 검출됨     → height_threshold_min 높이기 + min_area 증가
```

**Blue channel 값 (0-255)**:
- 200-255: 매우 높은 높이 (밝은 파란색)
- 100-200: 중간 높이 (파란색~초록색)
- 0-100: 낮은 높이 (초록색~빨강색)

### HSV 색상 범위 조정 (2D 색상 모드)

`height_map_mode: false`로 설정 시:

1. 테스트 이미지로 실행
2. 검출 결과 확인
3. `config.json`의 HSV 범위 조정
4. 재실행하여 검증

**팁**:
- H (Hue): 색조 (0-180)
- S (Saturation): 채도 (0-255)
- V (Value): 명도 (0-255)
- 회색-은색 페이스트: H=0-20, S=0-50, V=180-255

## 📊 출력 예시

### 콘솔 출력
```
처리 중: test_001.jpg
  검출된 윤곽선 수: 12
  총 면적: 28.654 mm²
  평균 면적: 2.388 mm²
  처리 시간: 0.42초
```

### 텍스트 리포트
```
============================================================
솔더 페이스트 면적 측정 리포트
============================================================

[통계 정보]
검출 개수: 12개
총 면적: 28.654 mm²
평균 면적: 2.388 ± 0.234 mm²
면적 범위: 1.850 ~ 2.950 mm²
중앙값 면적: 2.405 mm²

[개별 측정 결과]
ID    면적(mm²)     중심(x,y)       원형도
------------------------------------------------------------
1     2.450        (320, 240)     0.87
2     2.320        (450, 260)     0.92
...
============================================================
```

## 🏗️ 프로젝트 구조

```
solder_paste_inspector/
├── main.py                    # 메인 실행 파일
├── config.py                  # 설정 관리
├── image_processor.py         # 이미지 전처리
├── detector.py                # 페이스트 검출 엔진
├── measurement.py             # 면적 측정
├── visualizer.py              # 결과 시각화
├── calibration.py             # 카메라 캘리브레이션
├── utils.py                   # 유틸리티 함수
├── requirements.txt           # 필수 패키지
└── README.md                  # 본 문서
```

## 🔧 알고리즘

### 처리 파이프라인

1. **이미지 로드 및 전처리**
   - BGR → HSV 색상 공간 변환
   - 가우시안 블러 (노이즈 제거)

2. **색상 기반 세그멘테이션**
   - HSV 범위 기반 마스크 생성
   - 형태학적 연산 (Opening/Closing)

3. **윤곽선 검출**
   - 외부 윤곽선 검출
   - 면적 및 원형도 필터링

4. **면적 측정**
   - 픽셀 단위 면적 계산
   - 캘리브레이션 적용 (픽셀 → mm²)
   - 무게중심, 원형도 계산

5. **결과 시각화**
   - 윤곽선 표시 (녹색)
   - 중심점 마커 (빨간 점)
   - 면적 정보 텍스트

## ⚠️ 주의사항

### 캘리브레이션
- **필수**: 정확한 측정을 위해 카메라 캘리브레이션 필요
- **방법**: 알려진 크기의 기준 객체 (예: 1mm 그리드) 촬영

### HSV 범위
- **솔더 페이스트 종류별** 색상 특성이 다름
- **조명 조건**에 따라 HSV 값 변동
- **권장**: 실제 이미지로 범위 최적화

### 이미지 품질
- **조명**: 균일한 조명 필수
- **초점**: 선명한 이미지 필요
- **해상도**: 충분한 해상도 (최소 1024x768 권장)

## 🐛 문제 해결

### 검출되지 않는 경우
1. HSV 범위 조정 (`config.json`)
2. 면적 필터 조정 (MIN_AREA, MAX_AREA)
3. 이미지 품질 확인 (조명, 초점)

### 과검출 (노이즈 많음)
1. 원형도 필터 강화 (MIN_CIRCULARITY 증가)
2. 형태학적 연산 커널 크기 증가
3. 블러 강도 증가

### 측정값 부정확
1. 캘리브레이션 재실행
2. 리사이징 스케일 확인
3. 이미지 왜곡 보정 필요 여부 확인

## 📝 개발 로드맵

### Phase 1 (완료) ✅
- [x] 기본 검출 및 측정 시스템
- [x] 배치 처리
- [x] 캘리브레이션 도구
- [x] 결과 저장 (이미지, JSON, CSV)

### Phase 2 (예정)
- [ ] 실장 필렛 검사 모듈
- [ ] 품질 판정 로직
- [ ] GUI 인터페이스
- [ ] 실시간 카메라 처리
- [ ] 머신러닝 기반 검출

## 📄 라이선스

MIT License

## 👨‍💻 개발자

AOI 검사 시스템 개발팀

## 🙏 감사의 말

OpenCV 커뮤니티에 감사드립니다.
