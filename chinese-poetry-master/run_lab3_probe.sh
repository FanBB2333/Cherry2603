#!/usr/bin/env bash
set -euo pipefail
python3 -c 'import sys, lab3_probe_newline; sys.argv=["lab3_probe_newline.py", *sys.argv[2:]]; lab3_probe_newline.main()' dummy "$@"
