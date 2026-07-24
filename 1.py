from pathlib import Path

root_dir = Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan")
output_txt = root_dir / "leaf_folders.txt"

if not root_dir.exists():
    raise FileNotFoundError(f"目录不存在：{root_dir}")

leaf_folders = []

for folder in root_dir.rglob("*"):
    if not folder.is_dir():
        continue

    # 叶子文件夹：当前目录下不存在子文件夹
    has_subdir = any(item.is_dir() for item in folder.iterdir())
    if not has_subdir:
        leaf_folders.append(folder.resolve())

leaf_folders.sort()

with output_txt.open("w", encoding="utf-8") as f:
    for folder in leaf_folders:
        f.write(str(folder) + "\n")

print(f"共找到 {len(leaf_folders)} 个叶子文件夹")
print(f"结果已保存到：{output_txt}")