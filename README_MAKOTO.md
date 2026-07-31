# 🏊 橘真琴 (Makoto Tachibana) GPT-SoVITS 语音克隆方案

> 基于 GPT-SoVITS v3lora 的日语动漫角色语音克隆——「Free! 男子游泳部」橘真琴

[![Framework](https://img.shields.io/badge/Framework-GPT--SoVITS%20v3lora-blue)](https://github.com/RVC-Boss/GPT-SoVITS)
[![Python](https://img.shields.io/badge/Python-3.9-green)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011-lightgrey)]()
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-orange)]()
[![Voice](https://img.shields.io/badge/Voice-橘真琴%20(日语)-ff69b4)]()
[![Verified](https://img.shields.io/badge/Verified-2026.07.31-brightgreen)](./VERIFICATION_REPORT.md)

---

## 📖 目录

- [项目概述](#项目概述)
- [模型架构](#模型架构)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [训练流程](#训练流程)
  - [Stage 1: GPT 训练 (Text → Semantic)](#stage-1-gpt-训练-text--semantic)
  - [Stage 2: SoVITS v2 训练 (Semantic → Audio)](#stage-2-sovits-v2-训练-semantic--audio)
  - [Stage 3: SoVITS v3 LoRA 微调](#stage-3-sovits-v3-lora-微调)
- [推理与使用](#推理与使用)
- [模型权重清单](#模型权重清单)
- [项目文件结构](#项目文件结构)
- [训练日志与指标](#训练日志与指标)
- [常见问题](#常见问题)
- [参考与致谢](#参考与致谢)

---

## 项目概述

本项目使用 **GPT-SoVITS** 框架，对动漫《Free!》中角色 **橘真琴（Makoto Tachibana）** 的日语语音进行克隆，实现高质量日语 TTS（文本转语音）。

### 核心特性

- 🎯 **角色专属**：针对橘真琴声线进行三阶段微调训练（GPT + SoVITS v2 + LoRA v3）
- 🇯🇵 **日语支持**：完整的日语发音词典（732 个音素），原生支持日文输入
- 🖥️ **Windows 原生部署**：在 Windows 11 笔记本（RTX 5060）上完成全部训练
- 📦 **即插即用**：提供完整模型权重，GPT + SoVITS v2 端到端推理已实测通过（RTF 0.18-0.44x）
- 🧪 **LoRA v3**：成功训练，权重保存在 `SoVITS_weights_v3/`，需配合 v3 推理代码使用

### 语音效果

| 指标 | 说明 |
|------|------|
| **训练数据** | 50 条日语语音片段（来自原声对白） |
| **原始音频** | `dataset/vocals.wav`（约 40 MB，32kHz 单声道，约 10 分钟人声） |
| **采样率** | 32000 Hz |
| **自然度** | 保留橘真琴温柔、沉稳的声线特征 |
| **推理速度** | GPU 实时（RTF 0.27x ~ 0.63x）/ CPU 可接受延迟 |
| **已验证** | 2026-07-31 GPT+SoVITS v2 端到端测试：5/5 通过，详见 [验证报告](./VERIFICATION_REPORT.md) |
| **LoRA 状态** | ✅ 训练完成（258 键, rank=32），使用自定义 `b'03'` 文件头格式 |

---

## 模型架构

GPT-SoVITS 采用**两阶段级联架构**（本项目已验证 GPT + SoVITS v2 端到端工作）：

```
┌─────────────────────────────────────────────────────────┐
│                   推理流程 (Inference)                     │
├─────────────────────────────────────────────────────────┤
│  输入文本 (日语)                                           │
│      │                                                    │
│      ▼                                                    │
│  ┌──────────────┐    ┌──────────────────┐                │
│  │ BERT 编码     │    │ Chinese-HuBERT    │                │
│  │ (语义理解)     │    │ (音素特征提取)      │                │
│  └──────┬───────┘    └────────┬─────────┘                │
│         │                     │                           │
│         └──────────┬──────────┘                           │
│                    ▼                                      │
│  ┌──────────────────────────────────┐                    │
│  │  Stage 1: GPT (Text2Semantic)    │                    │
│  │  文本 → 语义令牌 (Semantic Tokens) │                    │
│  │  模型: makoto-e15.ckpt  ✅ 已验证  │                    │
│  └──────────────┬───────────────────┘                    │
│                 ▼                                         │
│  ┌──────────────────────────────────┐                    │
│  │  Stage 2: SoVITS v2              │                    │
│  │  语义令牌 → Mel频谱 → 波形         │                    │
│  │  模型: makoto_e8_s200.pth ✅ 已验证│                    │
│  └──────────────┬───────────────────┘                    │
│                 ▼                                         │
│  ┌──────────────────────────────────┐                    │
│  │  BigVGAN 声码器                   │                    │
│  │  Mel频谱 → 最终音频波形            │                    │
│  └──────────────┬───────────────────┘                    │
│                 ▼                                         │
│           输出音频 (32kHz WAV)                              │
└─────────────────────────────────────────────────────────┘
```

### 训练阶段对应关系

| 阶段 | 脚本 | 预训练基座 | 微调权重 | 状态 |
|------|------|-----------|---------|------|
| **1. GPT** | `s1_train.py` | `gsv-v2final-pretrained/s1bert25hz-5kh-longer` (v2) | `GPT_weights_v2/makoto-e*.ckpt` | ✅ 已验证 |
| **2. SoVITS v2** | `s2_train.py` | `gsv-v2final-pretrained/s2G2333k.pth` (v2) | `SoVITS_weights_v2/makoto_e*_s*.pth` | ✅ 已验证 |
| **3. SoVITS v3 LoRA** | `s2_train_v3_lora.py` | `pretrained_models/s2Gv3.pth` (v3) | `SoVITS_weights_v3/makoto_e*_s*_l32.pth` | ✅ 训练完成 |

---

## 环境准备

### 系统要求

| 组件 | 要求 |
|------|------|
| **操作系统** | Windows 10/11 (本项目使用 Windows 11 Home China) |
| **GPU** | NVIDIA GPU with CUDA (本项目使用笔记本电脑内置 GPU) |
| **显存** | ≥ 6 GB VRAM（训练）/ ≥ 4 GB VRAM（纯推理） |
| **内存** | ≥ 16 GB RAM |
| **Python** | 3.9（使用内置便携 Python 环境） |
| **磁盘** | ≥ 10 GB（含预训练模型和权重） |

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
```

本项目基于 **GPT-SoVITS-v3lora-20250228** 版本。

#### 2. 安装依赖

```bash
# Windows 用户可直接使用内置 Python 环境
# 或手动安装依赖：
pip install -r requirements.txt

# 关键依赖：
# - torch >= 2.0 (CUDA)
# - pytorch-lightning
# - gradio
# - transformers
# - peft (LoRA 支持)
# - tensorboard
```

#### 3. 下载预训练模型

GPT-SoVITS 需要以下预训练基座模型（放置到 `GPT_SoVITS/pretrained_models/`）：

```
GPT_SoVITS/pretrained_models/
├── chinese-hubert-base/          # HuBERT 特征提取
├── chinese-roberta-wwm-ext-large/ # BERT 语义编码
├── gsv-v2final-pretrained/
│   ├── s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt  # GPT v2 基座
│   ├── s2G2333k.pth              # SoVITS Generator v2 基座
│   └── s2D2333k.pth              # SoVITS Discriminator v2 基座
├── s1v3.ckpt                     # GPT v3 基座（可选）
└── s2Gv3.pth                     # SoVITS v3 基座（LoRA 基座）
```

#### 4. 放置本项目权重

将本项目训练好的权重文件放置到对应目录：

```
GPT_SoVITS/
├── GPT_weights_v2/
│   ├── makoto-e5.ckpt            # ~155 MB
│   ├── makoto-e10.ckpt           # ~155 MB
│   └── makoto-e15.ckpt           # ~155 MB (推荐)
├── SoVITS_weights_v2/
│   ├── makoto_e4_s100.pth        # ~85 MB
│   └── makoto_e8_s200.pth        # ~85 MB (推荐)
└── SoVITS_weights_v3/
    ├── makoto_e1_s100_l32.pth    # ~76 MB
    └── makoto_e2_s200_l32.pth    # ~76 MB (推荐)
```

---

## 数据准备

### 音频预处理流水线

```
原始音频 (vocals.wav)
    │
    ├── [UVR5] 人声分离（提取纯人声）
    │
    ├── [Audio Slicer] 静音检测切片
    │    └── 输出: output/slicer_opt/ (50 个片段)
    │
    ├── [Denoiser] 背景降噪
    │    └── 输出: output/denoise_opt/ (50 个片段)
    │
    └── [ASR] 语音识别 (Faster-Whisper Large v3) + 特征提取
         ├── logs/makoto/2-name2text.txt (文本 + 音素标注)
         ├── logs/makoto/6-name2semantic.tsv (语义令牌)
         ├── logs/makoto/3-bert/ (BERT 特征, 50 个 .pt)
         ├── logs/makoto/4-cnhubert/ (HuBERT 特征, 50 个 .pt)
         └── logs/makoto/5-wav32k/ (重采样 32kHz 音频)
```

### 关键参数配置

```yaml
# GPT Stage 训练配置 (tmp_s1.yaml)
train:
  batch_size: 4
  epochs: 15
  precision: 16-mixed
  exp_name: makoto
  half_weights_save_dir: GPT_weights_v2

model:
  n_layer: 24          # Transformer 层数
  embedding_dim: 512
  hidden_dim: 512
  head: 16              # 注意力头数
  phoneme_vocab_size: 732   # 日语+多语种音素

data:
  max_sec: 54           # 最大音频长度（秒）
  num_workers: 4

pretrained_s1: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
```

```json
// SoVITS Stage 训练配置 (tmp_s2.json)
{
  "train": {
    "batch_size": 4,
    "epochs": 8,
    "learning_rate": 0.0001,
    "fp16_run": true,
    "lora_rank": "32"
  },
  "data": {
    "sampling_rate": 32000,
    "n_mel_channels": 128,
    "hop_length": 640
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "n_heads": 2,
    "n_layers": 6,
    "semantic_frame_rate": "25hz"
  }
}
```

---

## 训练流程

### Stage 1: GPT 训练 (Text → Semantic)

将文本序列映射为语义令牌（Semantic Tokens），是两阶段训练的第一步。

```bash
cd GPT_SoVITS

# 使用 s1longer v2 配置训练（本项目使用）
python s1_train.py -c configs/s1longer-v2.yaml

# 或使用自定义配置
python s1_train.py -c path/to/your/s1_config.yaml
```

**本项目训练参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| 预训练基座 | `s1bert25hz-5kh-longer` (v2) | 5kHz 长上下文版本 |
| Batch Size | 4 | 受限于笔记本 GPU 显存 |
| Epochs | 15 | 每 5 epoch 保存一次 |
| 优化器 | AdamW | lr=0.01, warmup=2000 steps |
| 精度 | FP16 Mixed | 节省显存 |

**输出权重**：`GPT_weights_v2/makoto-e{5,10,15}.ckpt`

---

### Stage 2: SoVITS v2 训练 (Semantic → Audio)

将语义令牌转换为 Mel 频谱图，再通过声码器生成音频波形。

```bash
cd GPT_SoVITS

# SoVITS v2 训练
python s2_train.py
# 配置通过 configs/s2.json 或命令行参数指定
```

**本项目训练参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| 预训练基座 | `s2G2333k.pth` (v2) | SoVITS v2 Generator |
| Batch Size | 4 | 每 GPU |
| Epochs | 8 | 每 4 epoch 保存一次 |
| 学习率 | 1e-4 | 余弦衰减 |
| 采样率 | 32000 Hz | |
| Mel 通道数 | 128 | |

**输出权重**：`SoVITS_weights_v2/makoto_e{4,8}_s{100,200}.pth`

---

### Stage 3: SoVITS v3 LoRA 微调

在 v3 预训练基座（`s2Gv3.pth`）基础上，使用 LoRA（Low-Rank Adaptation, rank=32）在 CFM 模块上进行参数高效微调。

> 📌 **注意**：LoRA 权重使用了自定义文件头格式（`b'03'` 标记的 v3lora 格式），必须用 `process_ckpt.load_sovits_new()` 加载，不能直接用 `torch.load()`。这是框架设计行为，并非文件损坏。

```bash
cd GPT_SoVITS-v3lora-20250228
# 运行前需修改 TEMP/tmp_s2.json 中的 3 个字段（详见下方配置说明）
..\runtime\python.exe GPT_SoVITS\s2_train_v3_lora.py -c TEMP\tmp_s2.json
```

**训练前必须修改的配置**（`TEMP/tmp_s2.json`）：

| 字段 | 旧值 (v2 全量训练) | 新值 (v3 LoRA 训练) |
|------|-------------------|-------------------|
| `train.pretrained_s2G` | `…gsv-v2final-pretrained/s2G2333k.pth` | `GPT_SoVITS/pretrained_models/s2Gv3.pth` |
| `save_weight_dir` | `SoVITS_weights_v2` | `SoVITS_weights_v3` |
| `model.version` | `v2` | `v3` |
| `train.save_every_epoch` | `4` | `1`（确保每 epoch 都保存） |

**本项目训练参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| LoRA Rank | 32 | 低秩矩阵维度，作用于 CFM 的 `to_k/to_q/to_v/to_out` |
| 预训练基座 | `s2Gv3.pth` | SoVITS v3 架构（733 MB），`strict=False` 加载 |
| 训练数据 | `logs/makoto/` | 与 Stage 1/2 共享同一数据集 |
| Epochs | 2 | LoRA 快速收敛，实测 89 秒完成 |
| Batch Size | 4 | 每 step 4 条音频 |
| 保存间隔 | 每 epoch | 输出 `makoto_e{1,2}_s{25,50}_l32.pth` |

**输出权重**：`SoVITS_weights_v3/makoto_e1_s25_l32.pth` 和 `makoto_e2_s50_l32.pth`（各 72 MB, 258 键）

> 💡 **加载方式**：LoRA 权重使用 `b'03'` 文件头（v3lora 标记），加载时必须使用 `process_ckpt.load_sovits_new()`，它会自动将 `03` 替换为标准的 `PK`（zip）头。直接用 `torch.load()` 会报 `UnpicklingError`。

---

## 推理与使用

### 方式一：WebUI（推荐）

启动 Gradio Web 界面：

```bash
# 主界面（数据预处理 + 训练 + 推理）
python webui.py

# 或仅启动推理界面
python GPT_SoVITS/inference_webui.py
```

在 WebUI 中：
1. 选择 GPT 权重：`GPT_weights_v2/makoto-e15.ckpt`
2. 选择 SoVITS 权重：`SoVITS_weights_v2/makoto_e8_s200.pth`
3. 输入日语文本 → 点击生成

### 方式二：Python API

```python
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
import soundfile as sf

# 配置模型路径
config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
config.t2s_weights_path = "GPT_weights_v2/makoto-e15.ckpt"
config.vits_weights_path = "SoVITS_weights_v2/makoto_e8_s200.pth"
config.version = "v2"

# 初始化 TTS
tts = TTS(config)

# 构建推理输入（run() 接收 dict）
inputs = {
    "text": "おはよう、春ちゃん。元気？",   # 目标文本
    "text_lang": "ja",                       # 文本语言
    "ref_audio_path": "output/denoise_opt/vocals.wav_0000040640_0000210560.wav",  # 参考音频
    "prompt_text": "o [ h a y o o ] h a r u ...",   # 参考音频的音素标注
    "prompt_lang": "ja",                     # 参考音频语言
    "top_k": 15,                             # top-k 采样
    "temperature": 1.0,                      # 温度参数
    "seed": 42,                              # 随机种子
}

# run() 返回生成器，逐个 yield (sr, audio)，最后一个为最终结果
gen = tts.run(inputs)
sr, audio = None, None
for chunk in gen:
    if isinstance(chunk, tuple) and len(chunk) == 2:
        sr, audio = chunk

# 保存音频
sf.write("output.wav", audio, sr)
print(f"生成完毕: {len(audio)/sr:.2f} 秒 @ {sr}Hz")
```

> 💡 **提示**：`run()` 返回一个生成器（generator），逐段 yield 音频进度。遍历到最后一步即可获得完整合成音频。

### 方式三：HTTP API 服务

```bash
# 启动 API 服务 (端口 9880)
python api_v2.py
```

```python
import requests
import json

# 调用 API
response = requests.post(
    "http://localhost:9880/tts",
    json={
        "text": "橘真琴っていうんだ。よろしくね。",
        "text_lang": "ja",
        "gpt_weights": "GPT_weights_v2/makoto-e15.ckpt",
        "sovits_weights": "SoVITS_weights_v2/makoto_e8_s200.pth",
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### 推理配置说明

```yaml
# tts_infer.yaml 关键配置
custom:
  device: cuda          # cuda / cpu
  is_half: true         # FP16 推理 (GPU 推荐开启)
  version: v2           # v1 / v2 / v3
  t2s_weights_path: GPT_weights_v2/makoto-e15.ckpt
  vits_weights_path: SoVITS_weights_v2/makoto_e8_s200.pth
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base

# 参数说明
# top_k: 5-15 (越大越多样，越小越稳定)
# temperature: 0.6-1.0 (越高越有变化)
```

---

## 模型权重清单

### 本项目训练权重（橘真琴专属）

| 权重文件 | 大小 | 阶段 | 说明 |
|---------|------|------|------|
| `GPT_weights_v2/makoto-e5.ckpt` | 148 MB | GPT Epoch 5 | 中期检查点 |
| `GPT_weights_v2/makoto-e10.ckpt` | 148 MB | GPT Epoch 10 | 较优 |
| **`GPT_weights_v2/makoto-e15.ckpt`** | **148 MB** | **GPT Epoch 15** | **🏆 推荐（最终）** |
| `SoVITS_weights_v2/makoto_e4_s100.pth` | 81 MB | SoVITS v2 E4S100 | 中期检查点 |
| **`SoVITS_weights_v2/makoto_e8_s200.pth`** | **81 MB** | **SoVITS v2 E8S200** | **🏆 推荐（最终）** |
| `SoVITS_weights_v3/makoto_e1_s25_l32.pth` | 72 MB | LoRA v3 E1 S25 | ✅ 2026-07-31 重新训练 |
| **`SoVITS_weights_v3/makoto_e2_s50_l32.pth`** | **72 MB** | **LoRA v3 E2 S50** | **🏆 LoRA 推荐（最终）** |

> 📌 **LoRA 加载说明**：LoRA 权重使用 v3lora 自定义文件头格式（`b'03'`），必须用 `process_ckpt.load_sovits_new()` 加载。此函数自动处理头部字节替换（`03`→`PK`），正确解析出 258 个权重键（LoRA rank=32）。直接用 `torch.load()` 会报错，这是框架设计行为，并非文件损坏。

### 预训练基座模型（需自行下载）

| 模型文件 | 大小 | 用途 |
|---------|------|------|
| `chinese-hubert-base/` | ~400 MB | HuBERT 音频特征提取 |
| `chinese-roberta-wwm-ext-large/` | ~1.3 GB | BERT 日语文本编码 |
| `s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt` | ~310 MB | GPT v2 基座 |
| `s2G2333k.pth` | ~170 MB | SoVITS v2 Generator 基座 |
| `s2D2333k.pth` | ~170 MB | SoVITS v2 Discriminator 基座 |
| `s2Gv3.pth` | ~170 MB | SoVITS v3 Generator 基座（LoRA 用） |

---

## 项目文件结构

```
GPT_SoVITS/                              # 本项目工作目录
│
├── 📂 dataset/
│   └── vocals.wav                       # 原始人声音频（~40 MB）
│
├── 📂 output/
│   ├── slicer_opt/                      # 切片后音频（50 段 WAV）
│   └── denoise_opt/                     # 降噪后音频（50 段 WAV）
│
├── 📂 logs/
│   ├── makoto/                          # 训练数据集
│   │   ├── 2-name2text.txt              # 音素标注（50 条）
│   │   ├── 6-name2semantic.tsv          # 语义令牌序列
│   │   ├── 3-bert/                      # BERT 特征（50 个 .pt）
│   │   ├── 4-cnhubert/                  # HuBERT 特征（50 个 .pt）
│   │   ├── 5-wav32k/                    # 重采样 32kHz 音频
│   │   └── config.json                  # SoVITS 训练配置
│
├── 📂 GPT_weights_v2/                   # ★ GPT Stage 训练权重
│   ├── makoto-e5.ckpt                   #   Epoch 5  (~155 MB)
│   ├── makoto-e10.ckpt                  #   Epoch 10 (~155 MB)
│   └── makoto-e15.ckpt                  #   Epoch 15 (~155 MB) ★推荐
│
├── 📂 SoVITS_weights_v2/                # ★ SoVITS Stage 训练权重
│   ├── makoto_e4_s100.pth               #   Epoch 4, Step 100  (~85 MB)
│   └── makoto_e8_s200.pth               #   Epoch 8, Step 200  (~85 MB) ★推荐
│
├── 📂 SoVITS_weights_v3/                # ★ SoVITS v3 LoRA 权重
│   ├── makoto_e1_s100_l32.pth           #   Epoch 1, Step 100, Rank 32 (~76 MB)
│   └── makoto_e2_s200_l32.pth           #   Epoch 2, Step 200, Rank 32 (~76 MB) ★推荐
│
├── 📂 GPT_SoVITS/                       # GPT-SoVITS 框架代码
│   ├── configs/                         # 训练/推理配置文件
│   │   ├── s1.yaml / s2.json           # Stage 1 & 2 默认配置
│   │   ├── train.yaml                  # 训练超参
│   │   └── tts_infer.yaml              # 推理配置
│   ├── pretrained_models/              # 预训练基座模型（需下载）
│   ├── s1_train.py                     # GPT 训练入口
│   ├── s2_train.py                     # SoVITS v2 训练入口
│   ├── s2_train_v3_lora.py            # SoVITS v3 LoRA 训练入口
│   ├── inference_webui.py             # 推理 WebUI
│   ├── AR/                             # GPT (AR) 模型模块
│   ├── module/                         # SoVITS 模型模块
│   ├── text/                           # 文本处理（含日语支持）
│   └── feature_extractor/             # 音频特征提取
│
├── 📂 TEMP/
│   ├── tmp_s1.yaml                     # 实际使用的 GPT 训练配置
│   └── tmp_s2.json                     # 实际使用的 SoVITS 训练配置
│
├── config.py                           # 全局配置
├── weight.json                         # 预训练模型路径注册表
├── webui.py                            # 主 WebUI 入口
├── api.py / api_v2.py                 # HTTP API 服务
└── requirements.txt                    # Python 依赖
```

---

## 训练日志与指标

### 训练环境

| 项目 | 详情 |
|------|------|
| 主机名 | LAPTOP-KU14NIU8 |
| 操作系统 | Windows 11 Home China (Build 26200) |
| Python | 3.9 (便携版) |
| CUDA | 可用（笔记本 GPU） |
| 训练精度 | FP16 Mixed |

### GPT Stage 训练过程

| Epoch | 保存文件 | 说明 |
|-------|---------|------|
| 5 | `makoto-e5.ckpt` | 初步收敛，音色开始偏向目标 |
| 10 | `makoto-e10.ckpt` | 语义准确性明显提升 |
| 15 | `makoto-e15.ckpt` | 最终模型，推荐推理使用 |

### SoVITS Stage 训练过程

| Epoch/Step | 保存文件 | 说明 |
|-----------|---------|------|
| E4 S100 | `makoto_e4_s100.pth` | 中期检查点，基本音色已形成 |
| E8 S200 | `makoto_e8_s200.pth` | 最终模型，音质最佳 |

### LoRA 微调过程

| Epoch/Step | 保存文件 | 说明 |
|-----------|---------|------|
| E1 S100 | `makoto_e1_s100_l32.pth` | LoRA 快速收敛 |
| E2 S200 | `makoto_e2_s200_l32.pth` | 最终 LoRA，推荐配合 v2 权重使用 |

---

## 常见问题

### Q1: 可以在 CPU 上运行吗？

可以，但推理速度较慢。在 `config.py` 中设置 `infer_device = "cpu"` 或环境变量 `is_half=False`。训练建议使用 GPU。

### Q2: 如何训练自己的角色语音？

替换 `dataset/vocals.wav` 为你的人声音频，然后按照本文档的流程从数据预处理开始执行：
1. WebUI → 音频处理（UVR5 + 切片 + 降噪）
2. WebUI → 数据集制作（ASR + 特征提取）
3. 修改 `tmp_s1.yaml` 和 `tmp_s2.json` 中的路径
4. 依次运行三个训练阶段

### Q3: 为什么选择 v2 预训练基座而不是 v3？

v2 基座模型（`s1bert25hz-5kh-longer`）在日语语音上表现更稳定。v3 LoRA 是在 v2 权重基础上微调，结合了 v2 的稳定性和 v3 的架构优势。

### Q4: GPT 和 SoVITS 权重需要配套使用吗？

不强制配套，但推荐同一训练轮次配套使用。已验证可用的组合：
- **GPT + SoVITS v2**（端到端推荐）：`makoto-e15.ckpt` + `makoto_e8_s200.pth` — 5/5 日语测试句全部通过，RTF 0.18-0.44x
- **GPT + SoVITS v3 + LoRA**（实验性）：`makoto-e15.ckpt` + `s2Gv3.pth` + `makoto_e2_s50_l32.pth` — LoRA 权重已成功训练，但需框架代码适配才能在推理中加载 CFM 模块

### Q5: 显存不足怎么办？

- 减小 `batch_size`（当前为 4，可降至 2 或 1）
- 开启 `grad_ckpt` (gradient checkpointing)
- 使用 LoRA 模式训练（比全量微调省显存）

### Q6: 支持中文输入吗？

本项目主要优化日语。虽然框架支持中文和多语种混合，但模型是在日语数据上微调的，中文效果可能不如日语自然。

### Q7: LoRA 权重为什么不能用 `torch.load()` 加载？

报错 `UnpicklingError: unpickling stack underflow` 是因为 LoRA 权重使用了**自定义二进制文件头**。`my_save2()` 函数在保存时故意将 pickle 文件的前两个字节替换为 `b'03'`（v3lora 格式标记）。标准 `torch.load()` 不认识这个头部，导致反序列化失败。

**正确的加载方式**：
```python
from process_ckpt import load_sovits_new
ckpt = load_sovits_new("SoVITS_weights_v3/makoto_e2_s50_l32.pth")
# load_sovits_new 自动将 b'03' 头替换为 b'PK'（zip 标准头），然后正常反序列化
```

这是框架设计行为（参见 `process_ckpt.py` 的 `head2version` 字典和注释），并非文件损坏。旧的 v1/v2 权重使用 `b'00'`/`b'01'`/`b'02'` 等头部标记不同版本。

### Q8: 如何验证 LoRA 权重是否有效？

```python
from process_ckpt import load_sovits_new
ckpt = load_sovits_new("SoVITS_weights_v3/makoto_e2_s50_l32.pth")
print(len(ckpt["weight"]))  # 应为 258
print(ckpt["lora_rank"])    # 应为 32
print(ckpt["info"])         # 应为 "2epoch_50iteration"
```

### Q9: 项目从 D: 盘搬到 C: 盘后训练报错？

检查 `runtime\lib\site-packages\users.pth`，里面可能有旧盘符的绝对路径。将路径改为新的 C: 盘位置即可。

---

## 参考与致谢

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 开源语音克隆框架
- [Free! (アニメ)](https://iwatobi-sc.com/) — 橘真琴角色出处
- [Chinese-HuBERT](https://github.com/TencentGameMate/chinese_speech_pretrain) — 音频特征提取
- [Chinese-RoBERTa](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) — 文本编码
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) — 声码器

---

## 📜 License

本项目模型权重基于 GPT-SoVITS 框架训练，遵循原框架的 [MIT License](LICENSE)。

> ⚠️ **使用声明**：本模型仅供学习和研究使用。请勿用于非法用途或侵犯角色版权。使用本模型生成的音频请标明 AI 生成来源。

---

<p align="center">
  <b>🏊 橘真琴 — いつもそばにいる、優しい声 🏊</b>
</p>
