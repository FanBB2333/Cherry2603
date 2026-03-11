# 语音识别鲁棒性实验报告

## 实验设置

- **数据集**: Common Voice (英语)
- **样本数量**: 100条语音
- **模型**: facebook/wav2vec2-lv-60-espeak-cv-ft
- **评估指标**: 音素错误率 (PER)
- **噪声类型**: 白高斯噪声
- **信噪比 (SNR)**: 20dB, 10dB, 0dB, -10dB

## 实验结果

| 条件 | PER | 相对增长 |
|------|-----|---------|
| Clean (无噪声) | 0.634 | - |
| SNR 20dB | 0.652 | +2.8% |
| SNR 10dB | 0.709 | +11.8% |
| SNR 0dB | 0.837 | +32.0% |
| SNR -10dB | 0.969 | +52.8% |

## 结果分析

1. **基线性能**: 在无噪声条件下，模型的PER为0.634，表明约63.4%的音素识别错误
2. **噪声影响**: 随着SNR降低，PER显著上升，在-10dB时达到0.969
3. **鲁棒性**: 在20dB和10dB的轻度噪声下，模型保持相对稳定；但在0dB以下，性能急剧下降

详细的PER-SNR曲线见 `plots/per_vs_snr.png`

## 数据产出

- 清洁音频清单: `data/manifests/en/clean.jsonl`
- 噪声音频清单: `data/manifests/en/snr_{20,10,0,-10}.jsonl`
- 预测结果: `data/predictions/en/`
- 评估指标: `metrics/en/`
- 可视化结果: `plots/per_vs_snr.png`
