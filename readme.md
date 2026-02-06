

## 1. 环境准备 (Prerequisites)

*   **OS**: Linux / Windows
*   **GPU**: NVIDIA GPU with CUDA support
*   **Manager**: Anaconda or Miniconda

## 2. 创建 Conda 环境

首先创建一个 Python 3.8 的基础环境并激活：

```bash
conda create -n CityAVOS python=3.8 -y
conda activate CityAVOS
```

## 3. 安装 PyTorch

根据你的显卡驱动版本安装对应的 PyTorch。
> **注意**: 本项目 GroundingDINO 需要编译 CUDA 算子，请确保系统安装的 CUDA Toolkit 版本与 PyTorch 的 CUDA 版本一致或兼容。

这里使用 **CUDA 12.4** 版本：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 4. 安装 Grounded-Segment-Anything

### 4.1 克隆代码库
下载项目代码（如果尚未下载）：

```bash
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
```

### 4.2 安装依赖库
依次安装 Segment Anything (SAM) 和 GroundingDINO：

```bash
# 安装 Segment Anything
python -m pip install -e segment_anything

# 安装 GroundingDINO (需要编译 CUDA 扩展)
pip install --no-build-isolation -e GroundingDINO
```

## 5. 下载模型权重 (Model Weights)

请将以下权重文件下载并放置在 `Grounded-Segment-Anything` 项目根目录下。

### 5.1 SAM & GroundingDINO 权重

**Linux (wget):**
```bash
# SAM ViT-H Checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# GroundingDINO Swin-T Checkpoint
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

**Windows / 手动下载:**
如果 `wget` 不可用，请点击链接下载并手动放入目录：
*   [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
*   [groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

### 5.2 BERT 模型 (离线加载)

由于网络原因或离线运行需求，请下载 `bert-base-uncased` 模型文件。

1.  访问 HuggingFace: [bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main)
2.  下载必要文件（`config.json`, `pytorch_model.bin`, `vocab.txt`, `tokenizer.json` 等）。
3.  将文件放置在 `code/bert-base-uncased` 目录下（或根据代码中的 `text_encoder_type` 路径进行调整）。

目录结构示例：
```text
code/
└── bert-base-uncased/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    └── vocab.txt
```

## 6. 运行 Demo 测试

使用以下命令运行 `grounded_sam_demo.py` 进行测试。该脚本将使用文本提示 "bear" 检测并分割图片中的对象。

```bash
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo1.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "bear" \
  --device "cuda"
```

**参数说明:**
*   `--box_threshold`: 检测框的置信度阈值。
*   `--text_threshold`: 文本匹配的置信度阈值。
*   `--text_prompt`: 你想要检测的物体名称（支持自然语言）。

运行成功后，结果将保存在 `outputs` 文件夹中。

