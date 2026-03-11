# 数据下载和准备指南

## 📦 推荐数据源

### 1. Mozilla Common Voice（推荐）⭐️

**官网**：https://commonvoice.mozilla.org/

**特点**：
- 免费开源
- 支持100+语言
- 已包含转录文本
- 音频质量好

#### 下载步骤

1. **访问网站并注册**
   ```
   https://commonvoice.mozilla.org/datasets
   ```

2. **选择语言和版本**
   - 选择你需要的语言（如：English, French）
   - 选择最新版本（如：Common Voice Corpus 17.0）

3. **下载数据集**
   - 点击下载按钮
   - 选择格式：推荐下载 `.tar.gz` 格式
   - 文件大小：英语约30GB（完整版），可以下载较小的validated子集

4. **解压数据**
   ```bash
   # 解压下载的文件
   tar -xzf cv-corpus-17.0-2024-03-15-en.tar.gz
   ```

#### 数据结构

解压后的目录结构：
```
cv-corpus-17.0-2024-03-15/en/
├── clips/              # 音频文件目录
│   ├── common_voice_en_00001.mp3
│   ├── common_voice_en_00002.mp3
│   └── ...
├── validated.tsv       # 验证过的数据（推荐使用）
├── train.tsv          # 训练集
├── dev.tsv            # 验证集
└── test.tsv           # 测试集
```

#### 转换为项目所需格式

使用提供的转换脚本：

```bash
# 安装ffmpeg（用于音频转换）
brew install ffmpeg  # macOS

# 转换英语数据（处理前100个样本用于测试）
python scripts/convert_common_voice.py \
  --cv-dir ~/Downloads/cv-corpus-17.0-2024-03-15/en \
  --output-dir data/raw \
  --lang en \
  --max-samples 100

# 转换法语数据
python scripts/convert_common_voice.py \
  --cv-dir ~/Downloads/cv-corpus-17.0-2024-03-15/fr \
  --output-dir data/raw \
  --lang fr \
  --max-samples 100
```

转换后的目录结构：
```
data/raw/
├── en/
│   ├── wav/
│   │   ├── en_000000.wav
│   │   ├── en_000001.wav
│   │   └── ...
│   └── transcripts.txt
└── fr/
    ├── wav/
    │   ├── fr_000000.wav
    │   ├── fr_000001.wav
    │   └── ...
    └── transcripts.txt
```

---

### 2. LibriSpeech（英语专用）

**官网**：https://www.openslr.org/12/

**特点**：
- 英语语音数据集
- 高质量录音
- 约1000小时音频

#### 下载步骤

```bash
# 下载测试集（较小，约350MB）
wget https://www.openslr.org/resources/12/test-clean.tar.gz

# 解压
tar -xzf test-clean.tar.gz
```

---

### 3. 自己准备数据

如果你有自己的音频数据，需要准备成以下格式：

#### 目录结构

```
data/raw/<lang>/
├── wav/
│   ├── file001.wav
│   ├── file002.wav
│   └── ...
└── transcripts.txt
```

#### 音频要求

- **格式**：WAV
- **采样率**：16000 Hz
- **声道**：单声道（mono）
- **位深度**：16-bit（推荐）

#### 转换命令

如果你的音频不符合要求，使用ffmpeg转换：

```bash
# 转换单个文件
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# 批量转换（在音频目录中）
for file in *.mp3; do
  ffmpeg -i "$file" -ar 16000 -ac 1 "${file%.mp3}.wav"
done
```

#### transcripts.txt 格式

```
file001	This is the first sentence.
file002	This is the second sentence.
file003	Hello world.
```

**注意**：
- 文件名不包含扩展名（.wav）
- 使用Tab字符分隔（不是空格）
- 每行一个音频文件
- 使用UTF-8编码

---

## 🚀 快速开始（完整流程）

### 方案A：使用Common Voice（推荐）

```bash
# 1. 下载Common Voice数据集
# 访问 https://commonvoice.mozilla.org/datasets
# 下载英语和法语数据集

# 2. 解压数据
tar -xzf cv-corpus-17.0-2024-03-15-en.tar.gz
tar -xzf cv-corpus-17.0-2024-03-15-fr.tar.gz

# 3. 安装ffmpeg
brew install ffmpeg

# 4. 转换数据（先用少量数据测试）
python scripts/convert_common_voice.py \
  --cv-dir ~/Downloads/cv-corpus-17.0-2024-03-15/en \
  --output-dir data/raw \
  --lang en \
  --max-samples 50

# 5. 验证数据
ls data/raw/en/wav/
cat data/raw/en/transcripts.txt | head

# 6. 运行pipeline
dvc repro
```

### 方案B：使用示例数据（快速测试）

如果只是想测试pipeline是否工作，可以创建少量示例数据：

```bash
# 创建目录
mkdir -p data/raw/en/wav

# 使用espeak-ng生成示例音频
echo "This is a test sentence." | espeak-ng -w data/raw/en/wav/test001.wav
echo "Hello world." | espeak-ng -w data/raw/en/wav/test002.wav
echo "How are you today?" | espeak-ng -w data/raw/en/wav/test003.wav

# 创建transcripts.txt
cat > data/raw/en/transcripts.txt << EOF
test001	This is a test sentence.
test002	Hello world.
test003	How are you today?
EOF

# 转换为16kHz mono
for file in data/raw/en/wav/*.wav; do
  ffmpeg -i "$file" -ar 16000 -ac 1 -y "${file%.wav}_16k.wav"
  mv "${file%.wav}_16k.wav" "$file"
done

# 运行pipeline
dvc repro
```

---

## 📊 数据量建议

| 用途 | 建议样本数 | 预计处理时间 |
|-----|-----------|-------------|
| 快速测试 | 10-20 | 5-10分钟 |
| 开发调试 | 50-100 | 20-30分钟 |
| 完整实验 | 500-1000 | 2-4小时 |
| 论文级别 | 5000+ | 数小时到数天 |

**建议**：先用少量数据（50-100个样本）测试整个pipeline，确认无误后再处理完整数据集。

---

## ⚠️ 常见问题

### Q1: Common Voice下载很慢怎么办？

**A**:
- 使用下载工具（如wget、aria2c）支持断点续传
- 选择较小的子集（如只下载validated.tsv对应的数据）
- 考虑使用镜像站（如果有）

### Q2: 音频转换失败怎么办？

**A**: 检查ffmpeg是否正确安装：
```bash
ffmpeg -version
```

### Q3: transcripts.txt格式错误？

**A**: 确保：
- 使用Tab字符分隔（不是空格）
- 文件名不包含扩展名
- 使用UTF-8编码保存

验证方法：
```bash
# 查看分隔符（应该看到\t）
cat -A data/raw/en/transcripts.txt | head
```

### Q4: 数据太大，磁盘空间不足？

**A**:
- 只处理部分数据（使用--max-samples参数）
- 删除原始Common Voice数据（转换后）
- 使用外部存储

---

## 📋 数据准备检查清单

完成以下检查后再运行pipeline：

- [ ] 音频文件在 `data/raw/<lang>/wav/` 目录
- [ ] 所有音频文件是16kHz、单声道、WAV格式
- [ ] `transcripts.txt` 文件存在
- [ ] `transcripts.txt` 使用Tab分隔
- [ ] 文件名与音频文件匹配（不含扩展名）
- [ ] 至少有10个样本用于测试

验证命令：
```bash
# 检查音频文件
ls data/raw/en/wav/*.wav | wc -l

# 检查transcripts.txt
wc -l data/raw/en/transcripts.txt

# 检查音频格式
file data/raw/en/wav/*.wav | head
soxi data/raw/en/wav/*.wav | head
```

---

## 🎯 推荐工作流程

1. **下载小数据集测试**（50-100样本）
2. **运行完整pipeline验证**
3. **检查输出结果**（metrics和plots）
4. **确认无误后处理完整数据集**
5. **添加新语言**（只需修改params.yaml）

这样可以快速发现问题，避免浪费时间处理大量数据后才发现配置错误。
