#!/usr/bin/env bash
set -euo pipefail
python3 -c 'import sys, lab3_attention_benchmark; sys.argv=["lab3_attention_benchmark.py", *sys.argv[2:]]; lab3_attention_benchmark.main()' dummy "$@"
