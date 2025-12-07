# Chinese Aerial Image Processing Pipeline

중국 항공 위성 이미지를 처리하여 지리공간 데이터셋을 생성하는 파이프라인입니다.

## Overview

원본 이미지에서 지리공간 좌표를 추출하고, 모델 학습용 HDF5 형식의 데이터셋으로 변환합니다.

## Pipeline Steps

### Step 1: Extract Coordinates
**File: 1_make_csv.py**

이미지 파일명에서 지리공간 좌표를 추출합니다.

입력: `@timestamp@longitude@latitude@.png` 형식의 이미지
출력: CSV 파일 (image_name, latitude, longitude)

```bash
python 1_make_csv.py
```

### Step 2: Crop Images
**File: 2_image_crop.py**

2048×1536 이미지를 1536×1536 정사각형으로 중앙 기준 자르기.

특징:
- 멀티프로세싱으로 빠른 배치 처리
- 중앙 기준으로 정렬된 자르기
- 출력: cropped/ 폴더

```bash
python 2_image_crop.py
```

### Step 3: Standardize Filenames
**File: 3_change_image_name.py**

이미지 파일명을 순차 형식으로 변경.

입력: cropped/ 폴더의 이미지들
출력: 000000001.png, 000000002.png, ... 형식으로 이름 변경

특징:
- 자연 정렬로 일관된 순서 유지
- Step 1의 CSV와 동일한 순서 유지

```bash
python 3_change_image_name.py
```

### Step 4: Convert to Pixel Coordinates
**File: 4_refinement_china.py**

위도/경도 좌표를 이미지 픽셀 좌표로 변환합니다.

필요한 정보:
- 위성 이미지의 4개 모서리 좌표
- 이미지 해상도 정보

출력: x_px, y_px 컬럼 추가된 CSV

```bash
python 4_refinement_china.py
```

### Step 5: Generate HDF5 Dataset
**File: 5_preprocess.py**

전처리된 데이터를 HDF5 형식으로 내보냅니다.

특징:
- 이미지를 지정 크기로 리사이즈 (기본: 512×512)
- 중복 좌표 자동 제거
- Query/Database 쌍 생성

출력:
- test_queries.h5 (thermal/query 이미지)
- test_database.h5 (satellite/database 이미지)

```bash
python 5_preprocess.py
```

### Step 6: Fix Coordinate Order
**File: 6_make_x_y_swap.py**

이전 단계의 좌표 순서 오류를 수정합니다.

변환: @row@col → @col@row

```bash
python 6_make_x_y_swap.py
```

## Quick Start

### 전체 파이프라인 실행
```bash
python 1_make_csv.py
python 2_image_crop.py
python 3_change_image_name.py
python 4_refinement_china.py
python 5_preprocess.py
python 6_make_x_y_swap.py
```

### 개별 단계 실행
각 스크립트는 독립적으로 실행 가능하며, 입력 경로가 올바른지 확인하세요.

## Input/Output Structure

```
Raw Images
    ↓
1_make_csv.py → coordinates.csv
    ↓
2_image_crop.py → cropped/
    ↓
3_change_image_name.py → renamed/
    ↓
4_refinement_china.py → refined_coordinates.csv
    ↓
5_preprocess.py → {test_queries.h5, test_database.h5}
    ↓
6_make_x_y_swap.py → fixed_{test_queries.h5, test_database.h5}
```

## Configuration

각 스크립트 상단의 경로 설정:
- `input_dir`: 원본 이미지 폴더
- `output_dir`: 처리된 이미지 출력 폴더
- `csv_path`: CSV 파일 경로
- `image_size`: 리사이즈 크기 (기본: 512)

## Requirements

```bash
pip install h5py opencv-python pillow numpy pandas
```

## Data Format

### Input Image Filename
```
@timestamp@longitude@latitude@.png
예: @1234567890@120.5@35.2@.png
```

### CSV Format (Step 1 Output)
```
image_name,latitude,longitude
image_001.png,35.2,120.5
image_002.png,35.21,120.51
```

### HDF5 Format (Step 5 Output)
```
test_queries.h5:
  - image_name: 이미지 파일명 (N,)
  - image_data: 이미지 데이터 (N, H, W, 3)
  - image_size: 원본 크기 (N, 2)

test_database.h5:
  - image_name: 이미지 파일명 with 좌표 (N,)
  - image_size: 이미지 크기 (N, 2)
```

## Notes

- 모든 좌표는 WGS84 기준입니다
- 픽셀 좌표 변환을 위해 위성 이미지의 정확한 경계 좌표가 필요합니다
- Step 2와 Step 3의 순서는 중요합니다 (파일명 일관성)
- HDF5 생성 시 메모리 부족 발생 시 배치 크기를 줄이세요

## Troubleshooting

### "좌표 파싱 실패" 오류
파일명이 `@timestamp@longitude@latitude@.png` 형식인지 확인하세요.

### HDF5 파일 생성 실패
디스크 공간과 메모리 용량을 확인하세요.
대용량 데이터셋은 샘플링으로 축소 후 테스트하세요.

### 좌표 변환 오류
위성 이미지의 4개 모서리 좌표가 정확한지 확인하세요.