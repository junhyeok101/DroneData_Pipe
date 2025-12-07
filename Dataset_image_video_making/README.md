# Drone Dataset Pipeline

드론(UAV) 열화상 데이터와 위성 이미지를 쌍으로 이루는 데이터셋을 생성하고 검증하는 파이프라인입니다.

## Project Structure

이 프로젝트는 4단계의 데이터 처리 파이프라인으로 구성되어 있습니다.

### Stage 1: Query Image Preprocessing
**`make_1_query_image.py`**
- H5 파일에서 열화상(thermal) 이미지 읽기
- 768×768 크기에서 중앙 기준으로 512×512로 크롭
- RGB → BGR 포맷 변환 및 PNG로 저장
- 입력: `test_queries.h5`
- 출력: `query/` 폴더의 크롭된 이미지

### Stage 2: Query Video Generation
**`make_2_query_vidio.py`**
- Stage 1에서 생성한 쿼리 이미지들을 비디오로 변환
- 이미지를 숫자순으로 정렬하여 UAV 비행 궤적 재현
- 프레임레이트: 5fps, 포맷: MP4
- 입력: `query_images/` 폴더
- 출력: `query_video/uav_flight_row.mp4`

### Stage 3: Query-Database Pair Visualization
**`make_3_total_jeju_0_125_copy.py`** / **`make_3_total_NewYork.py`**
- 위성 이미지에서 좌표 기반으로 데이터베이스 이미지 추출
- 쿼리 이미지와 대응하는 위성 이미지를 나란히 시각화
- 중앙 위치를 빨간 점으로 표시하여 정렬 정확성 검증
- 입력: `test_queries.h5`, `test_database.h5`, `satellite_image.jpg`
- 출력: `total_images/pair_XXXX.png` (시각화 이미지 쌍)

### Stage 4: Matching Result Visualization
**`make_4_toal_video.py`**
- 매칭 결과 이미지들을 비디오로 변환
- Stage 3의 시각화 이미지들을 순차적으로 재생
- 입력: `match_images/` 폴더
- 출력: `match_video/match.mp4`

## Setup

### Installation
```bash
pip install h5py opencv-python matplotlib natsort
```

### Execution Order
```bash
# Step 1: Crop query images
python make_1_query_image.py

# Step 2: Generate query video
python make_2_query_vidio.py

# Step 3: Visualize query-database pairs
python make_3_total_jeju_0_125_copy.py  # Jeju dataset
# or
python make_3_total_NewYork.py          # New York dataset

# Step 4: Generate matching video
python make_4_toal_video.py
```

## Data Format

### Input Data
- **H5 Files**: `test_queries.h5`, `test_database.h5`
  - `image_name`: Image names with coordinates (@row@col format)
  - `image_data`: Image pixel data (N, H, W, 3)

- **Satellite Image**: JPG File
  - High-resolution satellite map (Jeju: 0.125m/px)

### Output Data
- **Images**: PNG (512×512 or 768×768)
- **Video**: MP4 (5fps)
- **Visualization**: Paired PNG images

## Validation

This pipeline verifies the following:

1. **Coordinate Validity**: Checks if query image coordinates are within satellite image bounds
2. **Image Alignment**: Visual confirmation that query-DB pairs correspond to the same location
3. **Data Completeness**: Ensures all queries have corresponding DB images
4. **Processing Errors**: Detects empty images or out-of-bounds cases

## Logging

Each script outputs processing progress:
```
Satellite image loaded: 12000×10000
Loaded 5000 query images
Loaded 50000 database names
Saved 100/5000 images
25 skipped (out of coordinate bounds)
Image pair visualization completed
```

## Troubleshooting

### "PNG images not found" error
Check that the previous step completed successfully and paths are correct.

### Coordinate out of bounds warning
This is normal. Samples near dataset boundaries are excluded as they lack sufficient context.

### Video generation failure
Verify OpenCV codec support: `pip install opencv-contrib-python`

## Features

- Automatic coordinate parsing (@row@col format)
- Boundary condition handling and validation
- Automatic image sorting (using natsort)
- Real-time progress output
- Visual verification (center point marking)

## Notes

- Update satellite image path in scripts as needed
- Verify H5 file keys (`image_name`, `image_data`) are correct
- Ensure sufficient disk space for large datasets
- Batch processing recommended if memory is limited

## License

[License information]

## Author

[Author information]