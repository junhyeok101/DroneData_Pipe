# 이미지 크롭 스크립트
# 2048x1536 이미지를 중앙에서 1536x1536으로 자르고 저장
# 멀티프로세싱을 사용하여 속도 향상
# 출력 폴더는 원본 폴더 내 'cropped'로 생성
# 이는 STHN 모델에 넣을 query 이미지를 정사각형으로 만들기 위함. 

import os
from PIL import Image
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

def process_image(args):
    """단일 이미지 처리 함수"""
    idx, total, filename, folder_path, output_folder = args
    file_path = os.path.join(folder_path, filename)
    
    try:
        # 이미지 열기
        img = Image.open(file_path)
        width, height = img.size
        
        # 2048x1536인 경우만 자르기
        if width == 2048 and height == 1536:
            # 중앙에서 1536x1536으로 자르기
            left = (width - 1536) // 2  # 중앙 자르기: 256
            top = 0
            right = left + 1536
            bottom = 1536
            
            cropped_img = img.crop((left, top, right, bottom))
            
            # 자른 이미지 저장
            output_path = os.path.join(output_folder, f"cropped_{filename}")
            cropped_img.save(output_path)
            return (idx, total, f"✓ {filename}")
        else:
            return (idx, total, f"✗ {filename} (크기: {width}x{height})")
    
    except Exception as e:
        return (idx, total, f"✗ {filename} (에러: {str(e)})")

# 폴더 경로 입력받기
folder_path = input("이미지 폴더 경로를 입력하세요: ").strip()

if not folder_path or not os.path.isdir(folder_path):
    print("유효한 폴더가 아닙니다.")
    exit()

# 이미지 파일 확장자
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

# 폴더 내 모든 이미지 파일 처리
image_files = [f for f in os.listdir(folder_path) 
               if Path(f).suffix.lower() in image_extensions]

if not image_files:
    print(f"'{folder_path}'에서 이미지를 찾을 수 없습니다.")
    exit()

# 출력 폴더 생성
output_folder = os.path.join(folder_path, "cropped")
os.makedirs(output_folder, exist_ok=True)

print(f"찾은 이미지: {len(image_files)}개")
print(f"출력 폴더: {output_folder}")
print(f"사용 CPU 코어: {cpu_count()}개\n")

# 멀티프로세싱으로 처리
start_time = time.time()
num_processes = cpu_count()
total_images = len(image_files)

with Pool(processes=num_processes) as pool:
    args_list = [(idx, total_images, filename, folder_path, output_folder) 
                 for idx, filename in enumerate(image_files, 1)]
    results = pool.imap_unordered(process_image, args_list)
    
    # 진행률 표시
    for idx, (current, total, message) in enumerate(results, 1):
        percentage = (idx / total_images) * 100
        print(f"[{idx:4d}/{total_images:4d}] ({percentage:5.1f}%) {message}")

elapsed_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"완료! 소요 시간: {elapsed_time:.2f}초")
print(f"평균 속도: {total_images/elapsed_time:.2f} 이미지/초")
print(f"자른 이미지는 '{output_folder}'에 저장되었습니다.")
print(f"{'='*60}")