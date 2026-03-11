# 项目完成总结

## 一、已完成的项目文件

| 文件路径 | 说明 | 状态 |
|---------|------|------|
| `params.yaml` | Pipeline参数配置文件 | ✅ 已创建 |
| `dvc.yaml` | DVC pipeline定义文件 | ✅ 已创建 |
| `requirements.txt` | Python依赖列表 | ✅ 已创建 |
| `.gitignore` | Git忽略文件配置 | ✅ 已创建 |
| `.dvcignore` | DVC忽略文件配置 | ✅ 已创建 |
| `README.md` | 项目说明文档 | ✅ 已创建 |
| `SETUP_GUIDE.md` | 详细设置和运行指南 | ✅ 已创建 |
| `scripts/create_manifest.py` | 创建manifest的脚本 | ✅ 已创建 |
| `scripts/add_noise.py` | 添加噪声的脚本 | ✅ 已创建 |
| `scripts/inference.py` | 运行音素识别的脚本 | ✅ 已创建 |
| `scripts/evaluate.py` | 评估PER的脚本 | ✅ 已创建 |
| `scripts/plot_results.py` | 绘制结果图表的脚本 | ✅ 已创建 |

## 二、需要您准备的数据文件

| 文件/目录 | 说明 | 格式要求 | 优先级 |
|----------|------|---------|--------|
| `data/raw/en/wav/*.wav` | 英语音频文件 | 单声道，16kHz，WAV格式 | 🔴 必需 |
| `data/raw/en/transcripts.txt` | 英语转录文本 | Tab分隔：`文件名\t文本` | 🔴 必需 |
| `data/raw/fr/wav/*.wav` | 法语音频文件 | 单声道，16kHz，WAV格式 | 🟡 可选 |
| `data/raw/fr/transcripts.txt` | 法语转录文本 | Tab分隔：`文件名\t文本` | 🟡 可选 |

### transcripts.txt 格式示例

```
file001	This is the first sentence.
file002	This is the second sentence.
file003	Hello world.
```

**注意**：
- 文件名不包含`.wav`扩展名
- 使用Tab字符分隔（不是空格）
- 每行对应一个音频文件

### 推荐数据来源

1. **Mozilla Common Voice** (https://commonvoice.mozilla.org/)
   - 免费、开源、多语言
   - 已包含转录文本
   - 需要注册账号下载

2. **LibriSpeech** (https://www.openslr.org/12/)
   - 英语语音数据集
   - 高质量录音

## 三、需要安装的系统依赖

| 软件 | 安装命令 | 说明 |
|-----|---------|------|
| espeak-ng | `brew install espeak-ng` (macOS) | 文本转音素工具 |
| espeak-ng | `sudo apt-get install espeak-ng` (Ubuntu) | 文本转音素工具 |
| Python 3.9+ | 系统自带或从python.org下载 | Python运行环境 |
| DVC | `pip install dvc` | 数据版本控制工具 |

## 四、运行步骤（简明版）

### 步骤1：环境准备

```bash
# 1. 安装espeak-ng
brew install espeak-ng  # macOS

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装Python依赖
pip install -r requirements.txt

# 4. 初始化DVC
dvc init
```

### 步骤2：准备数据

```bash
# 创建数据目录
mkdir -p data/raw/en/wav

# 将您的音频文件放入 data/raw/en/wav/
# 创建 data/raw/en/transcripts.txt 文件
```

### 步骤3：运行Pipeline

```bash
# 运行完整pipeline
dvc repro

# 查看结果
dvc metrics show
open plots/per_vs_snr.png
```

### 步骤4：添加新语言（可选）

```bash
# 1. 准备新语言数据
mkdir -p data/raw/fr/wav
# 添加法语音频和transcripts.txt

# 2. 修改params.yaml，添加 'fr' 到 languages 列表

# 3. 重新运行（只会运行新增的部分）
dvc repro
```

## 五、Pipeline工作流程

```
原始数据 (data/raw/)
    ↓
[create_manifest] → 创建manifest (data/manifests/)
    ↓
[add_noise] → 生成噪声音频 (data/noisy/)
    ↓
[inference] → 运行音素识别 (data/predictions/)
    ↓
[evaluate] → 计算PER指标 (metrics/)
    ↓
[plot_results] → 生成图表 (plots/per_vs_snr.png)
```

## 六、预期输出

运行成功后，您将得到：

1. **Manifest文件**: `data/manifests/<lang>/clean.jsonl` 和 `snr_*.jsonl`
2. **噪声音频**: `data/noisy/<lang>/snr_*/`
3. **预测结果**: `data/predictions/<lang>/`
4. **评估指标**: `metrics/<lang>/*.json`
5. **可视化图表**: `plots/per_vs_snr.png`

图表将显示：
- 每种语言的PER vs SNR曲线
- 跨语言平均PER曲线
- 展示ASR系统在不同噪声水平下的性能

## 七、Git提交建议

```bash
# 添加代码文件
git add scripts/ params.yaml dvc.yaml requirements.txt
git add .gitignore .dvcignore README.md SETUP_GUIDE.md
git add .dvc/config .dvc/.gitignore

# 提交
git commit -m "Add DVC pipeline for phoneme ASR robustness evaluation"

# 推送到GitHub
git remote add origin https://github.com/yourusername/Cherry2603.git
git push -u origin main
```

## 八、快速测试建议

为了快速验证pipeline是否工作，建议：

1. **使用少量数据**: 先用5-10个音频文件测试
2. **减少SNR级别**: 在`params.yaml`中只用3-4个SNR级别
3. **单一语言**: 先只测试英语

示例配置（快速测试）：
```yaml
languages:
  - en

snr_levels:
  - 10
  - 0
  - -10
```

## 九、常见问题快速解决

| 问题 | 解决方案 |
|-----|---------|
| espeak-ng找不到 | `which espeak-ng` 检查安装，重新安装 |
| 音频格式错误 | 使用ffmpeg转换：`ffmpeg -i input.wav -ar 16000 -ac 1 output.wav` |
| 内存不足 | 减少音频文件数量，或使用CPU而非GPU |
| DVC stage失败 | 查看错误信息，使用 `dvc repro -f <stage>` 强制重新运行 |

## 十、项目特点

✅ **完全可复现**: 使用固定随机种子和DVC版本控制
✅ **语言无关**: 添加新语言只需修改参数，无需改代码
✅ **原子操作**: Manifest采用原子写入，确保数据完整性
✅ **增量计算**: DVC自动检测变化，只运行必要的阶段
✅ **模块化设计**: 每个阶段独立，易于测试和维护

---

**下一步**: 请按照"运行步骤"准备数据并运行pipeline。如有问题，请参考`SETUP_GUIDE.md`获取详细说明。
