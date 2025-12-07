# 이미지 파일 이름을 000000001.png, 000000002.png, ... 형식으로 변경하는 스크립트
# 자연 정렬을 사용하여 파일 이름 순서대로 변경
# 변경 대상 폴더는 make/2_image_crop.py에서 크롭된 이미지 폴더
# 이는 make/1_make_csv.py에서 생성된 CSV 파일과 일치시키기 위함

import os
from pathlib import Path
from natsort import natsorted

# ===== 설정 =====
IMG_DIR = Path("AerialVL_sequence/long_trajtr/2023-03-18-14-38-32_영상/cropped")
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

# ===== 메인 =====
def main():
    if not IMG_DIR.exists():
        print(f"❌ 폴더를 찾을 수 없습니다: {IMG_DIR}")
        return
    
    # 이미지 파일 찾기 (자연 정렬)
    image_files = [f for f in IMG_DIR.iterdir() 
                   if f.suffix.lower() in IMAGE_EXTENSIONS]
    
    if not image_files:
        print(f"❌ 이미지를 찾을 수 없습니다: {IMG_DIR}")
        return
    
    # 자연 정렬 (001, 002, ..., 010, 011 순서)
    image_files = natsorted(image_files)
    
    print(f"ℹ️  찾은 이미지: {len(image_files)}개")
    print(f"예시: {image_files[0].name} → 000000001.png\n")
    
    # 이미지 이름 변경
    for idx, file_path in enumerate(image_files, 1):
        new_name = f"{idx:09d}.png"  # 000000001, 000000002, ...
        new_path = IMG_DIR / new_name
        
        try:
            file_path.rename(new_path)
            if idx % 100 == 0:
                print(f"  처리 중... {idx}/{len(image_files)}")
        except Exception as e:
            print(f"❌ 오류 {file_path.name}: {e}")
    
    print(f"\n✅ 완료! {len(image_files)}개 파일 이름 변경됨")
    print(f"   범위: 000000001.png ~ {len(image_files):09d}.png")

if __name__ == "__main__":
    main()