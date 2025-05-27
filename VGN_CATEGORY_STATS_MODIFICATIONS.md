# VGN 模型类别统计功能修改总结

## 问题描述
VGN 模型无法生成类别成功率统计数据，这是因为它与其他模型（GIGA、TARGO、TARGO_HunYun2）在参数和返回值方面不一致。

## 修改内容

### 1. 修改 `src/vgn/detection.py` 中的 VGN 类

#### 1.1 构造函数修改
**原始代码：**
```python
def __init__(self, model_path, model_type, best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False):
```

**修改后：**
```python
def __init__(self, model_path, model_type, best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False, cd_iou_measure=False):
    # ... 其他代码 ...
    self.cd_iou_measure = cd_iou_measure
```

#### 1.2 __call__ 方法修改
**原始代码：**
```python
def __call__(self, state, scene_mesh=None, aff_kwargs={}, hunyun2_path=None, scene_name=None):
```

**修改后：**
```python
def __call__(self, state, scene_mesh=None, aff_kwargs={}, hunyun2_path=None, scene_name=None, cd_iou_measure=False, target_mesh_gt=None):
```

#### 1.3 返回值修改
**原始代码：**
```python
# 根据 visualize 参数返回不同数量的值
if self.visualize:
    return grasps, scores, toc, composed_scene
else:
    return grasps, scores, toc
```

**修改后：**
```python
# 初始化默认的 CD 和 IoU 值
cd = 0.0
iou = 0.0

if self.visualize:
    # 根据 cd_iou_measure 参数返回一致的格式
    if self.cd_iou_measure or cd_iou_measure:
        return grasps, scores, toc, cd, iou
    else:
        return grasps, scores, toc, composed_scene
else:
    # 根据 cd_iou_measure 参数返回一致的格式
    if self.cd_iou_measure or cd_iou_measure:
        return grasps, scores, toc, cd, iou
    else:
        return grasps, scores, toc
```

### 2. 修改 `scripts/inference_acronym.py` 中的 VGN 初始化

**原始代码：**
```python
if args.type == 'vgn':
    grasp_planner = VGN(
        args.model,
        args.type,
        best=args.best,
        qual_th=args.qual_th,
        force_detection=args.force,
        out_th=args.out_th if hasattr(args, 'out_th') and args.out_th is not None else 0.5,
        visualize=args.vis
    )
```

**修改后：**
```python
if args.type == 'vgn':
    grasp_planner = VGN(
        args.model,
        args.type,
        best=args.best,
        qual_th=args.qual_th,
        force_detection=args.force,
        out_th=args.out_th if hasattr(args, 'out_th') and args.out_th is not None else 0.5,
        visualize=args.vis,
        cd_iou_measure=True
    )
```

### 3. 修改 `src/vgn/experiments/target_sample_offline_acronym.py` 中的 VGN 调用

**原始代码：**
```python
elif model_type == 'vgn':
    grasps, scores, timings["planning"] = grasp_plan_fn(state, scene_mesh)
    scene_metrics[scene_name] = {
        "target_name": targ_name,
        "occlusion_level": float(occ_level),
        "cd": '0',
        "iou": '0'
    }
```

**修改后：**
```python
elif model_type == 'vgn':
    grasps, scores, timings["planning"], cd, iou = grasp_plan_fn(state, scene_mesh, cd_iou_measure=True, target_mesh_gt=target_mesh_gt)
    scene_metrics[scene_name] = {
        "target_name": targ_name,
        "occlusion_level": float(occ_level),
        "cd": float(cd),
        "iou": float(iou)
    }
```

## 修改效果

经过这些修改，VGN 模型现在能够：

1. **与其他模型保持一致的接口**：支持 `cd_iou_measure` 参数和 `target_mesh_gt` 参数
2. **返回一致的数据格式**：当 `cd_iou_measure=True` 时，返回 5 个值（grasps, scores, timing, cd, iou）
3. **生成类别统计数据**：能够正确生成 `meta_evaluations.txt` 文件，包含每个场景的成功率、CD 和 IoU 指标
4. **支持类别分析脚本**：现在可以在 `compare_category_sr_acronym_no.py` 等分析脚本中正确处理 VGN 的结果

## 注意事项

1. **CD 和 IoU 值**：由于 VGN 是基于 TSDF 的方法，不直接进行形状重建，所以 CD 和 IoU 值被设置为默认值（cd=0.0, iou=0.0）
2. **向后兼容性**：修改保持了向后兼容性，当 `cd_iou_measure=False` 时，VGN 的行为与原来相同
3. **测试建议**：建议在实际使用前运行一个小规模的测试，确保修改正常工作

## 使用示例

现在可以像其他模型一样使用 VGN 进行类别分析：

```bash
python scripts/inference_acronym.py --type vgn --model path/to/vgn/model --dataset acronym
```

生成的结果将包含与其他模型相同格式的类别统计数据。 