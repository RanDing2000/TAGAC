Here is the complete and refined markdown: 

```markdown
# Environment Setup for Shape Completion and Others

## Install PyTorch
To install PyTorch, use the following command:
```bash
pip install torch==1.13.0+cu117
or
pip install torch==1.12.1+cu113
```

---

## 1. Installation for Shape Completion

### 1.1 Building PyTorch Extensions
Build the required PyTorch extensions for Chamfer Distance, PointNet++, and kNN:
```bash
module load cuda/11.3.0
module load compiler/gcc-8.3
cd src/shape_completion
bash install.sh
```

### 1.2 Install PointNet++ and GPU kNN
- **PointNet++:**
  ```bash
  pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
  ```
- **GPU kNN:**
  ```bash
  pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
  ```

---

## 2. Installation for Other Components

### 2.1 Install Minkowski Engine
(Include installation commands or steps if needed here.)
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

### 2.2 Install Other Dependencies
Install the linear attention transformer:
```bash
pip install linear_attention_transformer
```
```
### 2.3 torchscatter
```bash
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
```

You should download data according to GIGA: https://github.com/UT-Austin-RPL/GIGA?tab=readme-ov-file,

Pretrained models are also in the data.zip. They are in data/models. Pretrained models are also in the [data.zip](https://utexas.box.com/s/h3ferwjhuzy6ja8bzcm3nu9xq1wkn94s). They are in `data/models`.


# Data Generation
### 2.1 Generate the TARGO-Syn test
```bash
python scripts/generate_targo_dataset_test.py
```