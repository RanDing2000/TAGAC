import os

# 设置文件夹路径
input_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/collisions"
output_file = "/usr/stud/dira/GraspInClutter/targo/example_targets/acronym_target_list.txt"

# 获取所有以 .urdf 结尾的文件名，去掉后缀
urdf_names = [
    os.path.splitext(f)[0]  # 去掉后缀
    for f in os.listdir(input_dir)
    if f.endswith(".urdf")
]

# 按字典序排序（可选）
urdf_names.sort()

# 写入到目标 txt 文件
with open(output_file, "w") as f:
    for name in urdf_names:
        f.write(name + "\n")

print(f"已写入 {len(urdf_names)} 个目标到 {output_file}")
