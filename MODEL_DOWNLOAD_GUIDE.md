# HuggingFace 模型下载指南

## 📦 需要下载的模型

| 模型ID | 用途 | 大小 | 文件数 |
|--------|------|------|--------|
| `facebook/wav2vec2-lv-60-espeak-cv-ft` | 多语言音素识别（支持英语、法语等） | ~1.2GB | 多个文件 |

## 🚀 下载方法（三选一）

### 方法1：自动下载（最简单）⭐️

运行pipeline时会自动下载，无需手动操作：

```bash
# 直接运行，第一次会自动下载
dvc repro
```

**优点**：无需额外操作
**缺点**：第一次运行时需要等待下载

---

### 方法2：使用 huggingface-cli（推荐提前下载）

```bash
# 1. 安装 huggingface-cli
pip install -U "huggingface_hub[cli]"

# 2. 下载模型到默认缓存目录（推荐）
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft

# 或者下载到指定目录
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft \
  --local-dir ./models/wav2vec2-lv-60-espeak-cv-ft \
  --local-dir-use-symlinks False
```

**优点**：可以提前下载，支持断点续传
**缺点**：需要额外安装工具

---

### 方法3：使用提供的Python脚本

```bash
# 运行下载脚本
python scripts/download_models.py
```

**优点**：简单直接，显示下载进度
**缺点**：需要Python环境已配置好

---

## 🌐 国内用户加速下载

### 使用HuggingFace镜像站

```bash
# 方法A：设置环境变量（临时）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft
hf download facebook/wav2vec2-lv-60-espeak-cv-ft --local-dir ./models/facebook/wav2vec2-lv-60-espeak-cv-ft

# 方法B：永久设置（添加到 ~/.zshrc 或 ~/.bash_profile）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc
source ~/.zshrc
```

### 使用代理

```bash
# 设置代理
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890

# 然后下载
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft
```

---

## 📂 模型存储位置

### 默认缓存位置

```
~/.cache/huggingface/hub/models--facebook--wav2vec2-lv-60-espeak-cv-ft/
```

### 查看已下载的模型

```bash
# 查看缓存目录
ls -lh ~/.cache/huggingface/hub/

# 查看模型大小
du -sh ~/.cache/huggingface/hub/models--facebook--wav2vec2-lv-60-espeak-cv-ft/
```

---

## 🔍 验证模型是否下载成功

### 方法1：使用Python验证

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# 如果已下载，会直接加载；否则会开始下载
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

print("✅ 模型加载成功！")
```

### 方法2：检查文件是否存在

```bash
# 检查模型目录
ls ~/.cache/huggingface/hub/ | grep wav2vec2-lv-60-espeak-cv-ft
```

---

## 📋 完整下载流程（推荐）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 安装 huggingface-cli
pip install -U "huggingface_hub[cli]"

# 3. （可选）设置镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 4. 下载模型
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft

# 5. 验证下载
python scripts/download_models.py
```

---

## ⚠️ 常见问题

### Q1: 下载速度很慢怎么办？

**A**: 使用镜像站或代理：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q2: 下载中断了怎么办？

**A**: 使用 `huggingface-cli` 支持断点续传，重新运行命令即可继续下载。

### Q3: 如何删除已下载的模型？

**A**: 删除缓存目录：
```bash
rm -rf ~/.cache/huggingface/hub/models--facebook--wav2vec2-lv-60-espeak-cv-ft/
```

### Q4: 模型占用空间太大怎么办？

**A**: 这个模型是必需的，约1.2GB。如果空间不足，建议清理其他不需要的模型：
```bash
# 查看所有缓存的模型
ls ~/.cache/huggingface/hub/

# 删除不需要的模型
rm -rf ~/.cache/huggingface/hub/models--<model-name>/
```

### Q5: 能否使用其他模型？

**A**: 可以，但需要修改 `params.yaml` 中的 `model_name` 参数。确保新模型：
- 支持音素输出（不是文本）
- 支持你需要的语言
- 输入格式为16kHz音频

---

## 🎯 快速命令总结

```bash
# 最简单：自动下载（运行pipeline时）
dvc repro

# 推荐：提前下载
pip install -U "huggingface_hub[cli]"
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft

# 国内加速
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft

# 使用脚本
python scripts/download_models.py
```

---

## 📊 下载时间估算

| 网络速度 | 预计下载时间 |
|---------|-------------|
| 10 MB/s | ~2分钟 |
| 5 MB/s  | ~4分钟 |
| 1 MB/s  | ~20分钟 |
| 500 KB/s | ~40分钟 |

**建议**：如果网络较慢，使用镜像站或在网络条件好的时候提前下载。
