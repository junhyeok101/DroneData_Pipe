# 위도/경도를 이미지 픽셀 좌표로 변환하고 CSV에 추가하는 스크립트
# make/1_make_csv.py에서 생성된 CSV 파일을 입력으로 사용
# 변환된 픽셀 좌표는 'x_px', 'y_px' 컬럼으로 추가됨
# 위성 사진의 4개 꼭지점 좌표와 이미지 해상도를 설정해야 함


import pandas as pd
import numpy as np
from pathlib import Path

def lat_lon_to_pixel(lat, lon, bounds, image_width, image_height):
    """
    위도/경도를 이미지 픽셀 좌표로 변환
    
    Args:
        lat, lon: 변환할 위도/경도
        bounds: 위성 사진의 4개 꼭지점
        image_width: 이미지 가로 픽셀
        image_height: 이미지 세로 픽셀
    
    Returns:
        (x_px, y_px): 픽셀 좌표 (범위 밖이면 -1, -1)
    """
    
    # 경계 좌표 추출
    top_left_lat, top_left_lon = bounds['top_left']
    bottom_right_lat, bottom_right_lon = bounds['bottom_right']
    
    # 위도/경도 범위
    lat_min = min(top_left_lat, bottom_right_lat)
    lat_max = max(top_left_lat, bottom_right_lat)
    lon_min = min(top_left_lon, bottom_right_lon)
    lon_max = max(top_left_lon, bottom_right_lon)
    
    # 정규화 (0 ~ 1)
    norm_lon = (lon - lon_min) / (lon_max - lon_min) if (lon_max - lon_min) != 0 else 0
    norm_lat = (lat_max - lat) / (lat_max - lat_min) if (lat_max - lat_min) != 0 else 0
    
    # 픽셀 좌표로 변환
    x_px = int(norm_lon * image_width)
    y_px = int(norm_lat * image_height)
    
    # 경계 확인
    if x_px < 0 or x_px >= image_width or y_px < 0 or y_px >= image_height:
        return -1, -1
    
    return x_px, y_px


def process_csv_with_bounds(csv_file, bounds, image_width, image_height, output_file=None):
    """
    CSV 파일에 x_px, y_px 컬럼 추가
    
    Args:
        csv_file: 입력 CSV 파일 경로
        bounds: 위성 사진 4개 꼭지점 좌표
        image_width: 이미지 가로 픽셀
        image_height: 이미지 세로 픽셀
        output_file: 출력 CSV 파일 경로 (기본값: input_with_pixels.csv)
    
    Returns:
        처리된 DataFrame
    """
    
    # CSV 읽기
    df = pd.read_csv(csv_file)
    
    # 위도/경도 컬럼명 찾기
    lat_col = None
    lon_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['lat', 'latitude']:
            lat_col = col
        elif col_lower in ['lon', 'long', 'longitude']:
            lon_col = col
    
    if lat_col is None or lon_col is None:
        raise ValueError("CSV에 'lat'과 'lon' (또는 'long') 컬럼이 필요합니다")
    
    print(f"✓ 감지된 컬럼: lat='{lat_col}', lon='{lon_col}'")
    
    # 픽셀 좌표 계산
    x_pixels = []
    y_pixels = []
    
    total = len(df)
    for idx, row in df.iterrows():
        x, y = lat_lon_to_pixel(row[lat_col], row[lon_col], bounds, image_width, image_height)
        x_pixels.append(x)
        y_pixels.append(y)
        
        if (idx + 1) % 500 == 0:
            print(f"  처리 중: {idx + 1}/{total}")
    
    df['x_px'] = x_pixels
    df['y_px'] = y_pixels
    
    # 결과 저장
    if output_file is None:
        output_file = Path(csv_file).stem + '_train_with_pixels.csv'
    
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✓ 완료! 결과 저장: {output_file}")
    print(f"  총 {total}개 포인트 처리됨")
    print(f"\n샘플 (처음 5개):")
    print(df.head())
    
    return df


# ============================================================================
# 설정 및 실행
# ============================================================================

if __name__ == "__main__":
    # ★★★ 여기서 값을 수정하세요 ★★★
    
    # 위성 사진의 4개 꼭지점 좌표
    bounds = {
        # china_v2
        # 'top_left': (36.605606, 120.423889),      # 왼쪽 위
        # 'bottom_right':   (36.5736403827789, 120.466329753399)  # 오른쪽 아래

        #china_v3
        'top_left': (36.605606, 120.423889),      # 왼쪽 위
        'bottom_right':   (36.573338813822, 120.465903282166)  # 오른쪽 아래
    }
    
    # 이미지 해상도
    image_width = 7936   # 가로 픽셀 (수정 필요)
    image_height = 7424 # 픽셀 (수정 필요)
    
    # CSV 파일 경로
    csv_file = 'output_china.csv'  # 수정 필요
    
    # 실행
    try:
        result = process_csv_with_bounds(csv_file, bounds, image_width, image_height)
    except FileNotFoundError:
        print(f"❌ 오류: '{csv_file}' 파일을 찾을 수 없습니다")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")