# 중국 데이터셋의 경우, 사진의 이름이 좌표 정보를 닮고 있음. 
# 이를 활용하여 사진의 이름에서 좌표를 추출하고, CSV 파일로 저장하는 스크립트 작성
# 예: @1679121754506@120.44571166666668@36.60151833333333@.png

import os
import csv
import re
from pathlib import Path
from datetime import datetime

def extract_coordinates_from_filename(filename):
    """
    파일명에서 좌표 추출
    예: @1679121754506@120.44571166666668@36.60151833333333@.png
    반환: (longitude, latitude) 또는 (None, None)
    """
    # @ 기준으로 분할
    parts = filename.split('@')
    
    if len(parts) >= 4:
        try:
            longitude = float(parts[2])  # 두 번째 숫자
            latitude = float(parts[3])   # 세 번째 숫자
            return longitude, latitude
        except (ValueError, IndexError):
            return None, None
    
    return None, None

def process_image_folder(image_folder, output_csv):
    """
    이미지 폴더를 입력받아 CSV 파일 생성
    
    Args:
        image_folder: 이미지가 들어있는 폴더 경로
        output_csv: 생성할 CSV 파일 경로
    """
    
    # 이미지 폴더 존재 확인
    if not os.path.isdir(image_folder):
        print(f"❌ 폴더를 찾을 수 없습니다: {image_folder}")
        return
    
    # 이미지 파일 수집 (.png 파일만)
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])
    
    if not image_files:
        print(f"❌ {image_folder}에서 PNG 파일을 찾을 수 없습니다")
        return
    
    print(f"✓ 발견된 이미지: {len(image_files)}개")
    
    # CSV 파일 생성
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 헤더 작성
        writer.writerow(['image_name', 'latitude', 'longitude'])
        
        # 각 이미지 처리
        success_count = 0
        for idx, filename in enumerate(image_files, 1):
            # 새로운 이미지명: 000000001.png, 000000002.png, ...
            new_image_name = f"{idx:09d}.png"
            
            # 파일명에서 좌표 추출
            longitude, latitude = extract_coordinates_from_filename(filename)
            
            if longitude is not None and latitude is not None:
                writer.writerow([new_image_name, latitude, longitude])
                success_count += 1
                print(f"  [{idx}] {filename} → {new_image_name} (lat: {latitude}, lon: {longitude})")
            else:
                print(f"  [{idx}] {filename} → ⚠ 좌표 추출 실패")
        
        print(f"\n✓ CSV 파일 생성 완료: {output_csv}")
        print(f"✓ 성공한 이미지: {success_count}/{len(image_files)}")

if __name__ == "__main__":
    # 사용 예시
    # 여기서 이미지 폴더 경로와 출력 CSV 경로를 설정하세요
    
    # 입력 폴더 경로 (수정 필요)
    image_folder = "AerialVL_sequence/long_trajtr/2023-03-18-14-38-32_영상"
    
    # 출력 CSV 파일 경로
    output_csv = "output.csv"
    
    process_image_folder(image_folder, output_csv)