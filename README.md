# 드론 데이터셋 처리 파이프라인

드론 열화상 데이터와 위성 이미지를 처리하여 지리공간 데이터셋을 생성, 검증, 시각화하는 완전한 파이프라인입니다.

---

## Part 1: 데이터셋 생성 (Dataset_making)

원본 이미지 → 좌표 추출 → 크롭 → HDF5 변환

| 단계 | 파일 | 입력 | 출력 |
|------|------|------|------|
| 1 | `1_make_csv.py` | PNG 이미지 | coordinates.csv |
| 2 | `2_image_crop.py` | 2048×1536 이미지 | cropped/ (1536×1536) |
| 3 | `3_change_image_name.py` | cropped/ | 순차 파일명 변경 |
| 4 | `4_refinement_china.py` | CSV | 픽셀 좌표 변환 |
| 5 | `5_preprocess.py` | 이미지 + 좌표 | HDF5 파일 |
| 6 | `6_make_x_y_swap.py` | HDF5 | 좌표 순서 수정 |

**실행:**
```bash
python 1_make_csv.py && python 2_image_crop.py && python 3_change_image_name.py
python 4_refinement_china.py && python 5_preprocess.py && python 6_make_x_y_swap.py
```

**입력 파일명 형식:** `@timestamp@longitude@latitude@.png`

**출력:** `test_queries.h5`, `test_database.h5`

---

## Part 2: 데이터셋 검증 (Datasets_checking)

생성된 HDF5 파일의 구조, 좌표, 정렬 상태를 검증합니다.

| 파일 | 설명 | 명령어 |
|------|------|--------|
| `check_1_q_db_info.py` | H5 파일 구조 확인 | `python check_1_q_db_info.py` |
| `check_1_custom_data.py` | 좌표 정보 추출 | `python check_1_custom_data.py` |
| `check_2_sequence_plot.py` | 비행 경로 플롯 | `python check_2_sequence_plot.py` |
| `check_2_sequence_satellite.py` | 위성 이미지 위 경로 표시 | `python check_2_sequence_satellite.py` |
| `check_3_database_crop_random_test.py` | 랜덤 5개 샘플 비교 | `python check_3_database_crop_random_test.py` |
| `check_4_satellite_info.py` | 위성 이미지 정보 | `python check_4_satellite_info.py` |
| `check_5_target_grid_crop.py` | 특정 좌표 크롭 | 스크립트 수정 후 실행 |
| `query_satellite_check_center.py` | 전체 쌍 시각화 | `python query_satellite_check_center.py` |

**빠른 검증:**
```bash
python check_1_q_db_info.py
python check_1_custom_data.py
python check_2_sequence_satellite.py
python query_satellite_check_center.py
```

**좌표 형식:** `@row@col` (row: 세로, col: 가로)

---

## Part 3: 이미지 및 영상 생성 (Dataset_image_video_making)

HDF5 파일을 이미지로 변환하고 영상을 생성합니다.

| 단계 | 파일 | 처리 | 출력 |
|------|------|------|------|
| 1 | `make_1_query_image.py` | 768×768 → 512×512 크롭, PNG 저장 | query/*.png |
| 2 | `make_2_query_vidio.py` | PNG → MP4 비디오 (5fps) | query_video/uav_flight.mp4 |
| 3 | `make_3_total_jeju_0_125_copy.py` | 위성-쿼리 쌍 시각화 | total_images/pair_*.png |
| 3 | `make_3_total_NewYork.py` | 위성-쿼리 쌍 시각화 | total_images/pair_*.png |
| 4 | `make_4_toal_video.py` | 시각화 → MP4 비디오 | match_video/match.mp4 |

**실행:**
```bash
python make_1_query_image.py
python make_2_query_vidio.py
python make_3_total_jeju_0_125_copy.py  # 또는 make_3_total_NewYork.py
python make_4_toal_video.py
```

**검증:** 중앙점(빨간 점) 일치 여부 확인

---

## Part 4: 특정 행 데이터셋 생성 (Datasets_make_for_target_row)

### 개요

**독립적으로 작동합니다.** Part 1-3과 함께 사용할 수도, 별도로 사용할 수도 있습니다.

이 파이프라인은 생성된 HDF5 파일에서 특정 비행 경로(row)만 선택하여 개별 데이터셋으로 처리합니다.

**사용 시나리오:**
- 특정 지역의 비행 데이터만 추출
- 모델 학습 시 특정 경로의 데이터 집중 분석
- 작은 규모의 테스트 데이터셋 생성

### 처리 단계

| 단계 | 파일 | 설명 | 명령어 |
|------|------|------|--------|
| 0 | `make_check_row.py` | 가능한 row 목록 확인 | `python make_check_row.py` |
| 1 | `make_row_1_query.py` | 쿼리 추출 및 중복 제거 | `python make_row_1_query.py 3131` |
| 2 | `make_row_2_database.py` | 데이터베이스 추출 | `python make_row_2_database.py 3131` |
| 2.5 | `make_row_2.5_refine.py` | 좌표 정렬 정제 (선택) | `python make_row_2.5_refine.py 3131` |
| 3 | `make_row_3_query_image.py` | PNG 내보내기 | `python make_row_3_query_image.py 3131` |
| 4 | `make_row_4_query_vidio.py` | 비디오 생성 | `python make_row_4_query_vidio.py 3131` |
| 6 | `make_row_6_satellite_trajectory.py` | 경로 시각화 | `python make_row_6_satellite_trajectory.py 3131` |

### 실행 방법

**Step 0: 가능한 Row 확인**
```bash
python make_check_row.py
```

**순차 실행 (특정 row 선택, 예: 3131)**
```bash
python make_row_1_query.py 3131
python make_row_2_database.py 3131
python make_row_2.5_refine.py 3131      # 선택사항
python make_row_3_query_image.py 3131
python make_row_4_query_vidio.py 3131
python make_row_6_satellite_trajectory.py 3131
```

**자동 실행 (권장)**
```bash
# make_row_7_run_all.py 상단에서 target_number 수정
target_number = 3131

python make_row_7_run_all.py
```

### 출력 구조

```
t_datasets/
├── 3131_datasets/
│   ├── test_queries.h5       # 추출된 쿼리
│   └── test_database.h5      # 추출된 데이터베이스
├── 3131_query_images/
│   ├── q1_@3131@1000.png
│   ├── q2_@3131@1020.png
│   └── ...
├── video/
│   └── uav_flight_row3131.mp4
└── satellite_trajectory/
    └── 3131_trajectory.png
```

### 사용 예시

**제주 데이터셋에서 row 2276 추출**
```bash
python make_check_row.py
python make_row_1_query.py 2276
python make_row_2_database.py 2276
python make_row_3_query_image.py 2276
python make_row_4_query_vidio.py 2276
python make_row_6_satellite_trajectory.py 2276
```

**자동 모드로 빠르게 처리**
```bash
# make_row_7_run_all.py 수정 후
python make_row_7_run_all.py
```

---

## 전체 파이프라인 흐름

```
원본 이미지
    ↓
[Part 1] 데이터셋 생성
    ├─ 좌표 추출
    ├─ 이미지 크롭
    └─ HDF5 변환
    ↓
[Part 2] 데이터셋 검증
    ├─ 구조 확인
    ├─ 좌표 검증
    └─ 시각화 확인
    ↓
[Part 3] 이미지 및 영상 생성
    ├─ PNG 내보내기
    ├─ 쌍 시각화
    └─ 비디오 생성
    ↓
최종 데이터셋 완성

[Part 4] 특정 행 추출 (선택사항 & 독립적)
    ├─ Row 선택
    ├─ 직선 경로 추출
    └─ 개별 처리
```

---

## 설치 및 설정

### 필수 라이브러리
```bash
pip install h5py opencv-python matplotlib numpy pandas natsort pillow
```

### 경로 설정
각 스크립트 상단에서 수정:
- `input_dir`: 원본 이미지 폴더
- `output_dir`: 출력 폴더
- `query_h5_path`: 쿼리 H5 경로
- `db_h5_path`: 데이터베이스 H5 경로
- `sat_img_path`: 위성 이미지 경로

---

## 주요 기능

- 자동 좌표 파싱 (@row@col, @timestamp@lon@lat)
- 경계 조건 처리 및 유효성 검사
- 자동 이미지 정렬 (자연 정렬)
- 멀티프로세싱으로 빠른 배치 처리
- 실시간 진행 상황 표시
- 시각적 검증 (중앙점 표시, 경로 오버레이)

---

## 문제 해결

| 문제 | 해결 |
|------|------|
| 좌표 파싱 실패 | 파일명 형식 확인: `@timestamp@lon@lat@.png` |
| PNG 이미지 없음 | 이전 단계 완료 및 경로 확인 |
| 좌표 범위 초과 | 정상 (경계 근처 샘플 제외) |
| HDF5 생성 실패 | 디스크 공간 및 메모리 확인 |
| 비디오 생성 실패 | `pip install opencv-contrib-python` |
| Row 찾기 실패 | `make_check_row.py` 실행하여 가능한 row 확인 |
| H5 파일 오류 | 입력 H5 파일 경로 및 형식 확인 |

---

## 주의사항

- 모든 좌표는 WGS84 기준
- 중복 제거 후 데이터 개수 감소는 정상
- 대용량 데이터셋은 배치 크기 줄이기
- 비행 경로는 같은 row 내에서만 추출
- Step 2와 Step 3의 순서는 중요 (파일명 일관성)
- Part 4는 선택사항이며 Part 1-3의 결과물을 기반으로 작동