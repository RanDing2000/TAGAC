import os
import argparse

def get_filenames_without_extension(directory_path, extension=".npz"):
    """
    获取指定目录下所有文件的名称（不含指定扩展名）。

    Args:
        directory_path (str): 目录的路径。
        extension (str): 要移除的文件扩展名。

    Returns:
        set: 包含处理后文件名的集合。如果目录不存在或无法访问，则返回空集合。
    """
    filenames = set()
    if not os.path.isdir(directory_path):
        print(f"错误: 目录 '{directory_path}' 不存在或无法访问。")
        return filenames
    
    try:
        for f_name in os.listdir(directory_path):
            if os.path.isfile(os.path.join(directory_path, f_name)):
                base_name, ext = os.path.splitext(f_name)
                if ext.lower() == extension.lower(): # 比较扩展名时忽略大小写
                    filenames.add(base_name)
                else:
                    # 如果扩展名不匹配，可以选择是否添加不带扩展名的文件名
                    # 为保持与原始逻辑一致，这里只添加匹配指定扩展名的文件
                    # 如果希望包含所有文件（去除各自的扩展名），可以取消下一行注释
                    # filenames.add(base_name)
                    print(f"提示: 文件 '{f_name}' 在目录 '{directory_path}' 中的扩展名 '{ext}' 与目标扩展名 '{extension}' 不符，已跳过。") 
    except Exception as e:
        print(f"错误: 读取目录 '{directory_path}' 时发生错误: {e}")
    return filenames

def compare_directories(dir1_path, dir2_path, extension=".npz"):
    """
    比较两个目录中的文件名（不含扩展名）。

    Args:
        dir1_path (str): 第一个目录的路径。
        dir2_path (str): 第二个目录的路径。
        extension (str): 要从文件名中移除的扩展名。
    """
    print(f"正在比较目录：")
    print(f"目录 1: {dir1_path}")
    print(f"目录 2: {dir2_path}")
    print(f"比较文件的扩展名: {extension}\n")

    filenames_dir1 = get_filenames_without_extension(dir1_path, extension)
    filenames_dir2 = get_filenames_without_extension(dir2_path, extension)

    if not filenames_dir1 and not os.path.isdir(dir1_path): # 如果目录1读取失败且目录不存在
        print(f"无法继续比较，因为目录1 '{dir1_path}' 处理失败。")
        return
    if not filenames_dir2 and not os.path.isdir(dir2_path): # 如果目录2读取失败且目录不存在
        print(f"无法继续比较，因为目录2 '{dir2_path}' 处理失败。")
        return

    unique_to_dir1 = filenames_dir1 - filenames_dir2
    unique_to_dir2 = filenames_dir2 - filenames_dir1
    common_files = filenames_dir1 & filenames_dir2

    print(f"\n--- 比较结果 ---")
    print(f"在 '{os.path.basename(dir1_path)}' 中独有的文件 ({len(unique_to_dir1)} 个):")
    if unique_to_dir1:
        for name in sorted(list(unique_to_dir1))[:20]: # 最多显示前20个
             print(f"  {name}")
        if len(unique_to_dir1) > 20:
            print(f"  ... (及其他 {len(unique_to_dir1) - 20} 个文件)")
    else:
        print("  无")

    print(f"\n在 '{os.path.basename(dir2_path)}' 中独有的文件 ({len(unique_to_dir2)} 个):")
    if unique_to_dir2:
        for name in sorted(list(unique_to_dir2))[:20]: # 最多显示前20个
            print(f"  {name}")
        if len(unique_to_dir2) > 20:
            print(f"  ... (及其他 {len(unique_to_dir2) - 20} 个文件)")
    else:
        print("  无")

    print(f"\n两个目录共有的文件 ({len(common_files)} 个):")
    if common_files:
        for name in sorted(list(common_files))[:20]: # 最多显示前20个
            print(f"  {name}")
        if len(common_files) > 20:
            print(f"  ... (及其他 {len(common_files) - 20} 个文件)")
    else:
        print("  无")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较两个目录中的文件名（不含扩展名）。")
    parser.add_argument("dir1", help="第一个目录的路径。")
    parser.add_argument("dir2", help="第二个目录的路径。")
    parser.add_argument("--ext", default=".npz", help="要忽略的文件扩展名 (例如, '.npz', '.txt')。默认为 '.npz'。")
    
    args = parser.parse_args()
    
    compare_directories(args.dir1, args.dir2, args.ext)