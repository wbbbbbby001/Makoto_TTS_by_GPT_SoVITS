# 🏊 橘真琴 GPT-SoVITS 验证报告

> **验证日期**：2026-07-31（两轮验证：初验 + LoRA 重训练后复验）
> **验证方式**：端到端自动化测试
> **结论**：GPT + SoVITS v2 ✅ 完全可用 | LoRA v3 ✅ 训练成功，权重有效

---

## 验证环境

| 项目 | 值 |
|------|-----|
| **操作系统** | Windows 11 Home China (Build 26200) |
| **Python** | 3.9.13（项目内置 runtime 环境） |
| **PyTorch** | 2.8.0+cu128 |
| **CUDA** | 12.8 |
| **GPU** | NVIDIA GeForce RTX 5060 Laptop GPU |
| **gradio** | 4.24.0 |
| **transformers** | 4.43.0 |
| **pytorch_lightning** | 2.1.3 |
| **项目版本** | GPT-SoVITS-v3lora-20250228 |

---

## 验证结果总览

| # | 验证项 | 结果 | 备注 |
|---|--------|------|------|
| 1 | Python 环境 | ✅ 通过 | Python 3.9.13，PyTorch 2.8.0，CUDA 可用 |
| 2 | GPU 检测 | ✅ 通过 | RTX 5060 Laptop GPU，12.8 算力 |
| 3 | 依赖库导入 | ✅ 通过 | torch, numpy, gradio, transformers, pytorch_lightning |
| 4 | GPT-SoVITS 模块导入 | ✅ 通过 | config, text.japanese, cnhubert, T2S, SynthesizerTrn, TTS |
| 5 | 关键文件完整性 | ✅ 通过 | 19 个权重/配置/数据文件全部存在，大小正常 |
| 6 | GPT 模型加载 | ✅ 通过 | makoto-e15.ckpt（295 个权重键，n_layer=24, phoneme_vocab=732） |
| 7 | SoVITS v2 加载 | ✅ 通过 | makoto_e8_s200.pth（673 个权重键） |
| 8 | SoVITS v3 LoRA 加载 | ❌ **失败** | 两个文件均损坏，详见下方 |
| 9 | 端到端推理（日语） | ✅ 通过 | 3/3 测试句全部成功，RTF 0.27x ~ 0.63x |

---

## 🔄 LoRA 权重格式说明（初验误判，复验已纠正）

### 初验判断（已被推翻）

初验时用 `torch.load()` 加载 LoRA 权重时报 `UnpicklingError`，初步判断为文件损坏。

### 复验发现（正确结论）

LoRA 权重使用了 **GPT-SoVITS v3lora 的自定义文件头格式**，并非损坏。具体机制：

1. `process_ckpt.my_save2()` 保存时将 pickle 文件前 2 字节替换为 `b'03'`（v3lora 标记）
2. `process_ckpt.load_sovits_new()` 加载时将 `b'03'` 替换回 `b'PK'`（标准 zip 头），然后正常反序列化
3. `torch.load()` 不认识 `b'03'` 头，所以报 `unpickling stack underflow`

**正确加载方式**：
```python
from process_ckpt import load_sovits_new
ckpt = load_sovits_new("SoVITS_weights_v3/makoto_e2_s50_l32.pth")
# 成功: 258 keys, lora_rank=32, info="2epoch_50iteration"
```

### LoRA 重训练

初验后重新训练了 LoRA（修改 `save_every_epoch: 4→1` 确保保存），同时排查并修复了两个环境问题：

1. **`users.pth` 路径过时**：项目从 D: 盘搬到了 C: 盘，`runtime\lib\site-packages\users.pth` 仍指向 D: 盘，导致模块导入跨盘符失败
2. **训练配置残留**：`TEMP/tmp_s2.json` 被覆盖为 v2 全量训练配置，需修改 3 个字段（`pretrained_s2G`→s2Gv3, `save_weight_dir`→SoVITS_weights_v3, `model.version`→v3）

重训练结果：2 epochs / 89 秒完成，输出 `makoto_e1_s25_l32.pth` + `makoto_e2_s50_l32.pth`（各 258 键，rank=32）。

---

## ✅ 端到端推理验证详情

### 配置

| 参数 | 值 |
|------|-----|
| GPT 权重 | `GPT_weights_v2/makoto-e15.ckpt` |
| SoVITS 权重 | `SoVITS_weights_v2/makoto_e8_s200.pth` |
| 版本 | v2 |
| 精度 | FP16 |
| 设备 | CUDA (RTX 5060) |
| top_k | 15 |
| temperature | 1.0 |

### 测试结果

### 初验（3 句，2026-07-31 第一轮）

| # | 输入文本 | 含义 | 耗时 | 音频长度 | RTF | 结果 |
|---|---------|------|------|---------|-----|------|
| 1 | `おはよう、春ちゃん。元気？` | 早上好小春，还好吗 | 4.72s | 7.46s | 0.63x | ✅ |
| 2 | `橘真琴っていうんだ。よろしくね。` | 我叫橘真琴，请多关照 | 2.53s | 6.90s | 0.37x | ✅ |
| 3 | `僕、橘真琴。一緒に泳ごうよ。` | 我是橘真琴，一起游泳吧 | 2.94s | 10.74s | 0.27x | ✅ |

### 复验（5 句，2026-07-31 第二轮，LoRA 重训练后）

| # | 输入文本 | 场景 | 耗时 | 音频长度 | RTF | 结果 |
|---|---------|------|------|---------|-----|------|
| 1 | `おはよう、春ちゃん。今日もいい天気だね。` | 日常问候 | 4.3s | 9.8s | 0.44x | ✅ |
| 2 | `僕は橘真琴。水泳部の部長をやってるんだ。` | 角色介绍 | 1.8s | 9.5s | 0.18x | ✅ |
| 3 | `一緒に泳ごう！水の中は気持ちいいよ。` | 邀请游泳 | 1.4s | 6.3s | 0.22x | ✅ |
| 4 | `大丈夫？無理しないで、ゆっくり休もう。` | 关心他人 | 1.9s | 10.7s | 0.18x | ✅ |
| 5 | `ありがとう、みんな。最高の夏だったね。` | 感谢告别 | 1.6s | 8.9s | 0.18x | ✅ |

> **RTF (Real Time Factor)** 平均 0.24x，推理速度约为实时的 4 倍。RTX 5060 Laptop GPU 下游刃有余。

### 生成音频文件

测试生成的音频文件保存在 `TEMP/` 目录：
- `makoto_test_1.wav` — 7.46 秒
- `makoto_test_2.wav` — 6.90 秒
- `makoto_test_3.wav` — 10.74 秒

---

## 教程发现与修正

### 1. `run()` API（已修正）
- **❌ 错误**：`tts.run(text=text, text_lang="ja")` 关键字参数形式
- **✅ 正确**：`tts.run(dict)` — 接收包含 `text/text_lang/ref_audio_path/prompt_text/prompt_lang` 的字典，返回生成器需遍历消费

### 2. LoRA 权重格式（重要纠正）
- **初验误判**：以为是文件损坏（`UnpicklingError`）
- **正确结论**：v3lora 自定义文件头格式（`b'03'`），必须用 `load_sovits_new()` 加载

### 3. `users.pth` 盘符（新增）
- 项目迁移后需更新 `runtime\lib\site-packages\users.pth` 中的绝对路径

### 4. LoRA 训练配置（新增）
- 需修改 3 个字段：`pretrained_s2G`→s2Gv3, `save_weight_dir`→SoVITS_weights_v3, `model.version`→v3
- `save_every_epoch` 建议设为 1，避免小 epoch 数时跳过保存

---

## 结论

| 组件 | 状态 | 说明 |
|------|------|------|
| **GPT 训练** | ✅ 正常工作 | `makoto-e15.ckpt` 可用 |
| **SoVITS v2 训练** | ✅ 正常工作 | `makoto_e8_s200.pth` 可用 |
| **GPT + SoVITS v2 推理** | ✅ **5/5 通过** | RTF 0.18-0.44x，GPU 实时推理 |
| **SoVITS v3 LoRA 训练** | ✅ 重训练完成 | `makoto_e2_s50_l32.pth` 有效（258 键, rank=32） |
| **LoRA 推理集成** | ⚠️ 需代码适配 | 当前 `TTS.py` 加载 v2 架构（无 CFM），LoRA 需要 v3 架构（`SynthesizerTrnV3`） |

### 环境修复清单

如果在重训练或迁移项目时遇到问题，检查以下文件：

| 文件 | 可能的问题 | 修复方式 |
|------|-----------|---------|
| `runtime\lib\site-packages\users.pth` | 包含旧盘符路径（如 D:） | 改为新盘符 |
| `TEMP\tmp_s2.json` | 配置被覆盖为 v2 全量训练参数 | 改 3 个字段：`pretrained_s2G`→s2Gv3, `save_weight_dir`→SoVITS_weights_v3, `model.version`→v3, `save_every_epoch`→1 |

**总结**：橘真琴语音克隆方案核心管线（GPT + SoVITS v2）完全可用，5 句不同语气和场景的日语测试全部通过，GPU 推理速度达到实时的 4 倍以上。LoRA v3 权重已成功训练，可在未来的框架更新中启用。
