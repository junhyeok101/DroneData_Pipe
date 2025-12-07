import subprocess

# ===== 여기서 target number 지정 =====
target_number = 2276

scripts = [
    "make_row_1_query.py",
    "make_row_2_database.py",
    "make_row_2.5_refine.py",
    "make_row_3_query_image.py",
    "make_row_4_query_vidio.py",
    "make_row_6_satellite_trajectory.py",
    #"make_row_5_total_image.py",

]

for script in scripts:
    print(f"\n===== Running {script} with target={target_number} =====")
    result = subprocess.run(
        ["python3", script, str(target_number)],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
