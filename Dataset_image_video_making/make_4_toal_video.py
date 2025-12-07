# query 이미지들 -> video 제작 (UAV 비행 영상처럼)
import cv2
import os
import natsort
import sys


# === 경로 설정 ===
image_dir = "t_outputs/match_images"
output_dir = "t_outputs/match_video"
os.makedirs(output_dir, exist_ok=True)

output_video = os.path.join(output_dir, f"match.mp4")

# === 이미지 정렬 ===
images = [f for f in os.listdir(image_dir) if f.endswith(".png")]
images = natsort.natsorted(images)  # 숫자순 정렬

if len(images) == 0:
    raise ValueError(f"{image_dir} 안에 PNG 이미지가 없습니다!")

# === 첫 이미지 크기 확인 ===
first_img = cv2.imread(os.path.join(image_dir, images[0]))
h, w, c = first_img.shape

# === 비디오 저장 설정 ===
fps = 5
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

for i, img_name in enumerate(images):
    img = cv2.imread(os.path.join(image_dir, img_name))
    # 혹시 크기 불일치 시 강제 리사이즈
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    out.write(img)

    if i % 10 == 0:
        print(f"{i}/{len(images)} frames written")

out.release()
print(f"UAV 열화상 비행 영상 저장 완료 → {output_video}")
