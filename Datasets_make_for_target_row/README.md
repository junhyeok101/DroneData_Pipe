# Target Row Dataset Generation Pipeline

전체 데이터셋에서 특정 행(row)을 선택하여 열화상-위성 쌍 데이터셋을 생성합니다.

## Overview

직선 경로를 따라 비행하는 드론 데이터만 추출하여 처리합니다. 각 단계는 이전 단계의 출력을 입력으로 사용합니다.

## Scripts and Workflow

### Step 0: Check Available Rows
```bash
python make_check_row.py
```
전체 데이터셋에서 사용 가능한 row 목록과 각 row별 query 개수 확인.

출력:
```
총 row 종류 수: 20
row 값 목록: [3125, 3126, ...]
각 row별 query 개수:
row=3125: 500개
row=3126: 450개
...
```

### Step 1: Extract Query Images for Target Row
```bash
python make_row_1_query.py 3131
```
특정 row에 해당하는 thermal query 이미지만 추출 및 중복 제거.
- 입력: 전체 dataset의 test_queries.h5
- 출력: t_datasets/3131_datasets/test_queries.h5
- 처리: 500개 중 450개 유지 (중복 제거)

### Step 2: Extract Database for Target Row
```bash
python make_row_2_database.py 3131
```
같은 row에 속하는 위성 이미지(database) 추출 및 중복 제거.
- 입력: 전체 dataset의 test_database.h5
- 출력: t_datasets/3131_datasets/test_database.h5

### Step 2.5: Refine Data (Optional)
```bash
python make_row_2.5_refine.py 3131
```
좌표순 정렬 및 순서가 맞지 않는 항목 제거.
- col 값이 증가하는 구간만 유지
- 쿼리 및 데이터베이스 모두 정제

### Step 3: Export Query Images as PNG
```bash
python make_row_3_query_image.py 3131
```
H5 파일의 열화상 이미지를 PNG로 개별 저장.
- 입력: t_datasets/3131_datasets/test_queries.h5
- 출력: t_datasets/3131_query_images/*.png
- 형식: RGB → BGR 변환 후 저장

### Step 4: Generate Query Video
```bash
python make_row_4_query_vidio.py 3131
```
PNG 이미지들을 MP4 비디오로 변환 (UAV 비행 영상 재현).
- 입력: t_datasets/3131_query_images/*.png
- 출력: t_datasets/video/uav_flight_row3131.mp4
- 설정: 5fps, MP4 포맷

### Step 6: Plot Trajectory on Satellite
```bash
python make_row_6_satellite_trajectory.py 3131
```
추출된 쿼리의 비행 경로를 위성 이미지 위에 시각화.
- 입력: t_datasets/3131_datasets/test_queries.h5, 위성 이미지
- 출력: t_datasets/satellite_trajectory/3131_trajectory.png

### Info: Dataset Information
```bash
python make_row_info.py
```
스크립트 내 target_row 수정 후 실행.
추출된 데이터셋의 크기, 형식, 해상도 정보를 텍스트로 저장.
- 출력: t_outputs/dataset_info_row3131.txt

## Automated Execution

모든 단계를 한 번에 실행:
```bash
python make_row_7_run_all.py
```

스크립트 내 target_number를 수정한 후 실행하면 Step 1~6을 자동으로 처리합니다.

## Directory Structure

```
t_datasets/
├── 3131_datasets/           # 추출된 H5 파일
│   ├── test_queries.h5      # 열화상 쿼리
│   └── test_database.h5     # 위성 데이터베이스
├── 3131_query_images/       # 개별 PNG 이미지
│   ├── q1_@row@col.png
│   ├── q2_@row@col.png
│   └── ...
├── video/                   # 생성된 비디오
│   └── uav_flight_row3131.mp4
└── satellite_trajectory/    # 경로 시각화
    └── 3131_trajectory.png
```

## Configuration

각 스크립트의 경로 설정:
- `query_h5_path`: 원본 쿼리 H5 경로
- `db_h5_path`: 원본 데이터베이스 H5 경로
- `sat_img_path`: 위성 이미지 경로

## Usage Example

### 제주 데이터셋에서 row 2276 추출
```bash
python make_check_row.py                  # 가능한 row 확인
python make_row_7_run_all.py             # make_row_7_run_all.py 내 target_number = 2276으로 수정 후 실행
```

### 뉴욕 데이터셋에서 row 3131 추출
```bash
# 각 스크립트 상단의 경로를 NewYork 데이터셋으로 수정 후:
python make_row_1_query.py 3131
python make_row_2_database.py 3131
python make_row_2.5_refine.py 3131
python make_row_3_query_image.py 3131
python make_row_4_query_vidio.py 3131
python make_row_6_satellite_trajectory.py 3131
```

## Output Summary

| Step | Output | Type | Size |
|------|--------|------|------|
| 1 | test_queries.h5 | H5 | ~500MB |
| 2 | test_database.h5 | H5 | ~1GB |
| 3 | PNG images | Images | ~1GB |
| 4 | MP4 video | Video | ~200MB |
| 6 | Trajectory plot | PNG | ~50MB |

## Requirements

```bash
pip install h5py opencv-python matplotlib numpy natsort
```

## Notes

- Row는 UAV의 비행 경로상 세로 좌표를 의미합니다
- 같은 row에서만 쿼리를 선택하여 직선 경로를 보장합니다
- 중복 제거 후 데이터 개수는 감소할 수 있습니다 (정상)
- Step 5 (make_row_5_total_image.py)는 모델 평가 단계로 별도 설정 필요합니다