# Structure Comparison

Reference: `songci_main_5mb.txt`

Generated: `lab3_runs/main_songci5mb_gpu_baseline_5k/generated_samples.txt`

## Reference
{
  "file": "songci_main_5mb.txt",
  "num_poems": 20773,
  "num_lines": 157179,
  "avg_lines_per_poem": 7.566504597313821,
  "avg_raw_line_length": 10.741250421493966,
  "avg_stripped_line_length": 8.908880957379802,
  "punctuation_ratio": 0.17059182052468194,
  "lines_per_poem_distribution": {
    "8": 6546,
    "6": 4060,
    "9": 2061,
    "5": 1680,
    "10": 1451,
    "7": 1363,
    "4": 1045,
    "12": 727,
    "11": 484,
    "1": 457
  },
  "raw_line_length_distribution": {
    "8": 36438,
    "6": 13620,
    "16": 13444,
    "15": 11303,
    "12": 11006,
    "14": 9938,
    "11": 9707,
    "7": 7381,
    "5": 7163,
    "13": 6208,
    "4": 5765,
    "17": 5603,
    "9": 4523,
    "10": 4110,
    "18": 3699
  },
  "stripped_line_length_distribution": {
    "7": 35606,
    "5": 13631,
    "13": 12221,
    "6": 12204,
    "12": 12147,
    "14": 10994,
    "11": 10839,
    "9": 10799,
    "10": 10536,
    "4": 7161,
    "3": 5769,
    "8": 4600,
    "15": 4031,
    "17": 2884,
    "2": 1893
  },
  "line_ending_distribution": {
    "。": 157149,
    "）": 14,
    "�": 5,
    "𧣴": 3,
    "翩": 1,
    "行": 1,
    "“": 1,
    "，": 1,
    "收": 1,
    "□": 1
  }
}

## Generated
{
  "file": "lab3_runs/main_songci5mb_gpu_baseline_5k/generated_samples.txt",
  "num_poems": 100,
  "num_lines": 1540,
  "avg_lines_per_poem": 15.4,
  "avg_raw_line_length": 10.502597402597402,
  "avg_stripped_line_length": 8.582467532467533,
  "punctuation_ratio": 0.18282428589093608,
  "lines_per_poem_distribution": {
    "13": 15,
    "14": 14,
    "17": 10,
    "12": 9,
    "11": 9,
    "16": 9,
    "15": 9,
    "18": 6,
    "19": 5,
    "21": 4
  },
  "raw_line_length_distribution": {
    "8": 167,
    "5": 131,
    "6": 125,
    "7": 116,
    "15": 110,
    "10": 105,
    "14": 103,
    "11": 100,
    "12": 83,
    "9": 78,
    "16": 76,
    "13": 73,
    "4": 66,
    "17": 48,
    "19": 32
  },
  "stripped_line_length_distribution": {
    "7": 202,
    "6": 141,
    "4": 130,
    "5": 129,
    "11": 120,
    "8": 119,
    "12": 118,
    "10": 115,
    "9": 106,
    "13": 83,
    "3": 72,
    "14": 67,
    "15": 42,
    "2": 26,
    "16": 26
  },
  "line_ending_distribution": {
    "。": 1458,
    "，": 11,
    "、": 3,
    "风": 2,
    "天": 2,
    "一": 2,
    "雨": 2,
    "春": 2,
    "何": 2,
    "花": 2
  }
}

## Comparison
{
  "line_count_l1": 1.8004399942232705,
  "raw_line_length_l1": 0.39232222043475456,
  "stripped_line_length_l1": 0.3203158728038006,
  "line_ending_l1": 0.10645533345512351,
  "avg_lines_per_poem_gap": 7.83349540268618,
  "avg_raw_line_length_gap": -0.23865301889656365,
  "avg_stripped_line_length_gap": -0.32641342491226943,
  "punctuation_ratio_gap": 0.012232465366254136
}
