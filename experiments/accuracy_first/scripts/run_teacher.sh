#!/usr/bin/env bash
set -euo pipefail
python -m experiments.accuracy_first.train.train_teacher --config experiments/accuracy_first/configs/teacher.yaml
