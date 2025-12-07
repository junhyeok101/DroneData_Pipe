# Dataset Checking Tools

H5 파일 기반 드론 데이터셋의 구조, 좌표, 정렬을 검증하는 도구 모음입니다.

## Scripts Overview

### check_1_custom_data.py
H5 파일에서 쿼리와 데이터베이스 이미지 이름 및 좌표 추출.
```bash
python check_1_custom_data.py
```
출력: 각 이미지별 @row@col 형식의 좌표 정보

### check_1_q_db_info.py
H5 파일 구조 검사 (키, shape, dtype 등).
```bash
python check_1_q_db_info.py
```
출력: 데이터셋 메타데이터, 이름/크기 샘플

### check_2_sequence_plot.py
쿼리 좌표를 기반으로 비행 경로 플롯 생성.
```bash
python check_2_sequence_plot.py
```
출력: query_trajectory.png (좌표 범위 및 경로 시각화)

### check_2_sequence_satellite.py
비행 경로를 위성 이미지 위에 오버레이.
```bash
python check_2_sequence_satellite.py
```
출력: satellite_trajectory.png (위성 맵 위의 쿼리 위치)

### check_3_database_crop_random_test.py
랜덤하게 선택된 5개 쿼리의 위성 크롭 및 열화상 비교.
```bash
python check_3_database_crop_random_test.py
```
출력: sample_pairs_random/ (위성 패치와 쿼리 쌍 이미지)

### check_4_satellite_info.py
위성 이미지 및 H5 파일의 크기, 해상도 정보 출력.
```bash
python check_4_satellite_info.py
```
출력: 터미널에 이미지 크기 및 데이터셋 통계

### check_5_target_grid_crop.py
지정된 좌표에서 특정 크기로 위성 이미지 크롭.
```bash
python check_5_target_grid_crop.py
```
사용: 스크립트 내 coord_str과 crop_size 수정 후 실행
출력: korea_datasets/sample/ (크롭된 이미지)

### query_satellite_center_grid.py
수동으로 지정한 좌표에서 위성-쿼리 쌍 시각화.
```bash
python query_satellite_center_grid.py
```
사용: 스크립트 내 manual_coords 리스트 수정
출력: center_query_datbase_manual/ (쌍 비교 이미지)

### query_satellite_check_center.py
모든 쿼리에 대해 위성 크롭과 열화상 쌍을 자동 생성.
```bash
python query_satellite_check_center.py
```
출력: center_query_datbase/ (전체 쌍 이미지, 중앙점 표시)

## Coordinate Format

이미지 이름은 `@row@col` 형식으로 좌표를 포함합니다.
- 예: `q1_@12345@67890` → row=12345, col=67890
- row: 위성 이미지의 세로 좌표
- col: 위성 이미지의 가로 좌표

## Quick Workflow

1. 데이터셋 구조 확인
   ```bash
   python check_1_q_db_info.py
   ```

2. 좌표 확인
   ```bash
   python check_1_custom_data.py
   ```

3. 비행 경로 시각화
   ```bash
   python check_2_sequence_satellite.py
   ```

4. 정렬 검증
   ```bash
   python query_satellite_check_center.py
   ```

## Configuration

각 스크립트 상단의 경로를 데이터셋에 맞게 수정하세요:
- `db_h5_path`: 데이터베이스 H5 파일 경로
- `query_h5_path`: 쿼리 H5 파일 경로
- `sat_img_path`: 위성 이미지 경로

## Requirements

```bash
pip install h5py opencv-python matplotlib numpy
```

## Notes

- 좌표 범위 경고는 정상입니다 (경계 근처 샘플 제외)
- 큰 데이터셋은 샘플 수를 줄여 테스트하세요
- 모든 이미지는 RGB 포맷으로 저장됩니다