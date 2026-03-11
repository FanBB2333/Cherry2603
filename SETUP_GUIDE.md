# 数据准备和运行指南

## 一、需要准备的数据文件

### 必需的数据结构

您需要准备以下数据文件（这些文件需要您自己下载或准备）：

| 文件路径 | 说明 | 格式要求 |
|---------|------|---------|
| `data/raw/en/wav/*.wav` | 英语音频文件 | 单声道，16kHz采样率，WAV格式 |
| `data/raw/en/transcripts.txt` | 英语转录文本 | Tab分隔：`文件名\t转录文本` |
| `data/raw/fr/wav/*.wav` | 法语音频文件（可选） | 单声道，16kHz采样率，WAV格式 |
| `data/raw/fr/transcripts.txt` | 法语转录文本（可选） | Tab分隔：`文件名\t转录文本` |

### transcripts.txt 格式示例

```
commonvoice_000001	I can hardly believe it.
commonvoice_000002	Please call me tomorrow.
commonvoice_000003	The weather is nice today.
```

注意：
- 文件名不包含扩展名（.wav）
- 使用Tab字符分隔（不是空格）
- 每行一个音频文件

### 推荐的数据来源

1. **Common Voice** (https://commonvoice.mozilla.org/)
   - 多语言开源语音数据集
   - 已包含转录文本
   - 需要转换为所需格式

2. **LibriSpeech** (https://www.openslr.org/12/)
   - 英语语音数据集
   - 高质量录音

## 二、环境设置

### 1. 安装系统依赖

```bash
# macOS
brew install espeak-ng

# Ubuntu/Debian
sudo apt-get install espeak-ng

# 验证安装
espeak-ng --version
```

### 2. 安装Python依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装DVC
pip install dvc
```

### 3. 初始化Git和DVC

```bash
# 初始化Git（如果还没有）
git init

# 初始化DVC
dvc init

# 添加远程存储（可选，用于共享数据）
# dvc remote add -d myremote /path/to/remote/storage
# 或使用云存储：
# dvc remote add -d myremote s3://mybucket/path
```

## 三、运行Pipeline

### 步骤1：准备数据

创建数据目录结构：

```bash
mkdir -p data/raw/en/wav
mkdir -p data/raw/fr/wav
```

将您的音频文件放入相应的目录，并创建transcripts.txt文件。

### 步骤2：配置参数

编辑 `params.yaml`：

```yaml
# 开始时只用一种语言测试
languages:
  - en

# 可以先用较少的SNR级别测试
snr_levels:
  - 10
  - 5
  - 0
  - -5
```

### 步骤3：运行Pipeline

```bash
# 运行完整pipeline
dvc repro

# 查看pipeline状态
dvc dag

# 查看metrics
dvc metrics show

# 查看生成的图表
open plots/per_vs_snr.png  # macOS
# 或 xdg-open plots/per_vs_snr.png  # Linux
```

### 步骤4：添加新语言

编辑 `params.yaml` 添加新语言：

```yaml
languages:
  - en
  - fr  # 添加法语
```

再次运行：

```bash
# DVC会自动检测变化，只运行必要的阶段
dvc repro

# 查看更新后的结果
dvc metrics show
open plots/per_vs_snr.png
```

## 四、Pipeline阶段说明

| 阶段 | 输入 | 输出 | 说明 |
|-----|------|------|------|
| create_manifest | 原始音频+文本 | clean.jsonl | 创建清洁音频的manifest |
| add_noise | clean.jsonl | snr_X.jsonl + 噪声音频 | 在不同SNR级别添加噪声 |
| inference_clean | clean.jsonl | predictions/clean.jsonl | 对清洁音频进行音素识别 |
| inference_noisy | snr_X.jsonl | predictions/snr_X.jsonl | 对噪声音频进行音素识别 |
| evaluate_clean | predictions/clean.jsonl | metrics/clean.json | 计算清洁音频的PER |
| evaluate_noisy | predictions/snr_X.jsonl | metrics/snr_X.json | 计算噪声音频的PER |
| plot_results | 所有metrics | per_vs_snr.png | 生成PER vs SNR曲线图 |

## 五、常见问题

### 1. espeak-ng找不到

```bash
# 检查是否安装
which espeak-ng

# 如果没有，重新安装
brew install espeak-ng  # macOS
```

### 2. 内存不足

如果数据集很大，可以：
- 减少音频文件数量
- 使用较小的batch size
- 在inference.py中添加批处理逻辑

### 3. 音频格式错误

确保音频文件：
- 是单声道（mono）
- 采样率为16kHz
- 格式为WAV

转换命令（使用ffmpeg）：
```bash
ffmpeg -i input.wav -ar 16000 -ac 1 output.wav
```

### 4. DVC缓存问题

```bash
# 清理DVC缓存
dvc gc

# 强制重新运行某个阶段
dvc repro -f <stage_name>
```

## 六、提交到Git

```bash
# 添加代码和配置文件
git add scripts/ params.yaml dvc.yaml requirements.txt .gitignore README.md
git add .dvc/config .dvc/.gitignore

# 提交
git commit -m "Initial pipeline implementation"

# 推送到GitHub
git remote add origin <your-github-repo-url>
git push -u origin main
```

## 七、预期输出

成功运行后，您应该看到：

1. **Manifests**: `data/manifests/<lang>/*.jsonl`
2. **Metrics**: `metrics/<lang>/*.json`
3. **Plot**: `plots/per_vs_snr.png` - 显示每种语言和平均PER vs SNR的曲线

## 八、性能优化建议

1. **使用GPU**: 如果有GPU，inference会快很多
2. **并行处理**: 可以修改脚本支持多进程
3. **缓存模型**: 第一次运行会下载模型，后续运行会使用缓存
4. **减少数据量**: 测试时可以只用少量音频文件

## 九、扩展建议

1. 添加更多语言（西班牙语、德语等）
2. 尝试不同的噪声类型（不只是白噪声）
3. 测试不同的ASR模型
4. 添加更多评估指标（WER, CER等）
