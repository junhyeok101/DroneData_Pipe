import os
import h5py

# === 타겟 행 번호 설정 ===
target_row = 3131  # ✅ 원하는 row 번호 지정

save_dir = "t_outputs"
os.makedirs(save_dir, exist_ok=True)

out_path = os.path.join(save_dir, f"dataset_info_row{target_row}.txt")

with open(out_path, "w", encoding="utf-8") as f:

    # Query (thermal)
    query_h5_path = f"t_datasets/{target_row}_datasets/test_queries.h5"
    with h5py.File(query_h5_path, "r") as qf:
        f.write("=== Query (Thermal) ===\n")
        f.write(f"File: {query_h5_path}\n")
        f.write(f"Keys: {list(qf.keys())}\n")

        if "image_data" in qf:
            f.write(f"image_data shape: {qf['image_data'].shape}\n")
            f.write(f"image_data dtype: {qf['image_data'].dtype}\n")

        if "image_name" in qf:
            names = [n.decode("utf-8") for n in qf["image_name"][:5]]
            f.write(f"image_name 개수: {len(qf['image_name'])}\n")
            f.write(f"첫 5개 이름 예시: {names}\n")

        f.write("\n- Thermal 이미지는 512×512 patch\n")
        f.write("- 해상도: 1 m/px\n")
        f.write("- 실제 커버 범위: 512 m × 512 m\n\n")

    # Database (satellite)
    db_h5_path = f"t_datasets/{target_row}_datasets/test_database.h5"
    with h5py.File(db_h5_path, "r") as df:
        f.write("=== Database (Satellite) ===\n")
        f.write(f"File: {db_h5_path}\n")
        f.write(f"Keys: {list(df.keys())}\n")

        if "image_name" in df:
            names = [n.decode("utf-8") for n in df["image_name"][:5]]
            f.write(f"image_name 개수: {len(df['image_name'])}\n")
            f.write(f"첫 5개 이름 예시: {names}\n")

        if "image_size" in df:
            sizes = df["image_size"][:5].tolist()
            f.write(f"image_size shape: {df['image_size'].shape}\n")
            f.write(f"첫 5개 사이즈 예시: {sizes}\n")

        f.write("\n- Satellite 이미지는 WS×WS 크기로 crop 후 사용\n")
        f.write("  ((thermal 512×512과 영역 매칭, 단 실제 m/px은 WS에 따라 달라짐))\n")
        f.write("  • WS = 512  → 1 px = 1.0 m → 실제 커버 512 m\n")
        f.write("  • WS = 1024 → 실제 커버 1024 m → Resize(256) 후 1 px = 4 m\n")
        f.write("  • WS = 1536 → 실제 커버 1536 m → Resize(256) 후 1 px = 6 m\n\n")

    f.write("저장 완료!\n")

print(f"[Row {target_row}] 데이터셋 정보 저장 완료 → {out_path}")
