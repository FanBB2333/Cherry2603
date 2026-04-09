#!/usr/bin/env bash
set -euo pipefail
python3 -c 'import sys, lab3_train; sys.argv=["lab3_train.py", *sys.argv[2:]]; lab3_train.main()' dummy "$@"
