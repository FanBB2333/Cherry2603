#!/usr/bin/env bash
set -euo pipefail
python3 -c 'import sys, lab3_generate_from_checkpoint; sys.argv=["lab3_generate_from_checkpoint.py", *sys.argv[2:]]; lab3_generate_from_checkpoint.main()' dummy "$@"
