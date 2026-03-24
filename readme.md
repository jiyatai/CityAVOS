
# CityAVOS

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/38898)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-green.svg)](https://www.python.org/downloads/release/python-380/)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)]()

> **CityAVOS**: Towards autonomous uav visual object search in city space: Benchmark and agentic methodology

CityAVOS is a research project for **UAV (drone) target search tasks in urban spaces**, built upon the [EmbodiedCity](https://github.com/tsinghua-fib-lab/EmbodiedCity) simulator. It integrates large language model (LLM) agents, Grounded-SAM for visual grounding and segmentation, and AirSim for realistic drone control within Unreal Engine 5.3 environments.

📄 **Paper**: Published at **AAAI 2026** — [Read the Paper](https://ojs.aaai.org/index.php/AAAI/article/view/38898)

---

## 🏗️ Architecture Overview

```
CityAVOS
├── main/
│   ├── main.py              # Main entry point
│   └── mykey.py             # AirSim connection test script
├── GroundedSAM/
│   ├── GroundingDINO/        # Visual grounding module
│   ├── weights/              # Model weights directory
│   └── bert-base-uncased/    # BERT model for text encoding
└── llm_agent.py                # LLM-based decision agent
```

---

## 📋 Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Windows |
| **GPU** | NVIDIA GPU with CUDA support |
| **Package Manager** | [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) |
| **Game Engine** | Unreal Engine 5.3 |

---

## 🚀 Installation

### 1. Install External Dependencies

#### EmbodiedCity Simulator

Follow the instructions to install the EmbodiedCity simulator:

👉 [https://github.com/tsinghua-fib-lab/EmbodiedCity](https://github.com/tsinghua-fib-lab/EmbodiedCity)

#### Modified AirSim (Colosseum for UE 5.3)

Install the UE 5.3 compatible version of AirSim:

👉 [https://github.com/CodexLabsLLC/Colosseum/tree/ue-5.3](https://github.com/CodexLabsLLC/Colosseum/tree/ue-5.3)

---

### 2. Create Conda Environment

```bash
conda create -n CityAVOS python=3.8 -y
conda activate CityAVOS
```

---

### 3. Configure AirSim Python Client

```bash
pip install msgpack-rpc-python
pip install airsim
pip install pygame
```

> ✅ **Verification**: Run `python main/mykey.py` to verify that the AirSim environment is correctly installed and connected.

---

### 4. Install Grounded-SAM

Reference: [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)

```bash
cd Grounded-Segment-Anything

# Install Segment Anything
pip install segment_anything

# Install GroundingDINO
cd GroundingDINO
pip install -e .
cd ..
```

---

### 5. Download Model Weights

Download the following model weights and place them in `Grounded-Segment-Anything/weights/`:

| Model | Filename | Download |
|---|---|---|
| GroundingDINO | `groundingdino_swint_ogc.pth` | [Link](https://github.com/IDEA-Research/GroundingDINO/releases) |
| SAM ViT-H | `sam_vit_h_4b8939.pth` | [Link](https://github.com/facebookresearch/segment-anything#model-checkpoints) |

```
Grounded-Segment-Anything/
└── weights/
    ├── groundingdino_swint_ogc.pth
    └── sam_vit_h_4b8939.pth
```

---

### 6. Download BERT Model

Download `bert-base-uncased` and place it under the `Grounded-Segment-Anything/` directory:

```
Grounded-Segment-Anything/
└── bert-base-uncased/
    ├── config.json
    ├── tokenizer.json
    ├── vocab.txt
    └── ...
```

You can download it from [Hugging Face](https://huggingface.co/bert-base-uncased).

---

### 7. Configure LLM Agent

Set up the API key for the LLM agent. Update the key configuration in the `llm_agent` module:

```python
# Example: configure your API key
API_KEY = "your-api-key-here"
```

---

## ▶️ Usage

Make sure the EmbodiedCity simulator (UE 5.3) is running, then execute:

```bash
conda activate CityAVOS
python main.py
```

---

## 📖 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{ji2026towards,
  title={Towards autonomous uav visual object search in city space: Benchmark and agentic methodology},
  author={Ji, Yatai and Zhu, Zhengqiu and Zhao, Yong and Liu, Beidan and Gao, Chen and Zhao, Yihao and Qiu, Sihang and Hu, Yue and Yin, Quanjun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={22},
  pages={18342--18350},
  year={2026}
}
```

---

## 🙏 Acknowledgements

- [EmbodiedCity](https://github.com/tsinghua-fib-lab/EmbodiedCity) — Urban embodied simulation platform
- [Colosseum (AirSim)](https://github.com/CodexLabsLLC/Colosseum) — Open-source drone simulation for UE 5.3
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) — Visual grounding & segmentation
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) — Foundation model for segmentation
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) — Open-set object detection

---

## 📄 License

This project is released under the [MIT License](LICENSE).
```
