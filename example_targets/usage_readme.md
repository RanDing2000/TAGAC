# 目标视频录制功能使用说明

## 功能说明
该功能允许用户指定一个包含目标名称的txt文件，程序只会对这些指定的目标进行视频录制，从而减少存储空间的使用并专注于感兴趣的目标。

## 使用方法

1. 创建一个文本文件，每行包含一个目标名称，例如：
   ```
   002_master_chef_can
   003_cracker_box
   004_sugar_box
   ```

2. 在运行`inference_ycb.py`时，添加`--target-file`参数指向该文件：
   ```bash
   python scripts/inference_ycb.py --video-recording=True --target-file=example_targets/target_list.txt
   ```

3. 如果不指定`--target-file`参数但启用了视频录制功能，程序将会录制所有目标的视频。

## 注意事项
- 目标名称必须与系统中实际的目标名称完全匹配，包括大小写和特殊字符。
- 每行只能包含一个目标名称。
- 空行将被忽略。 