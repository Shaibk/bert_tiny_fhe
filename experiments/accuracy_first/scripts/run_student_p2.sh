#!/usr/bin/env bash
set -euo pipefail
python -m experiments.accuracy_first.train.train_student --config experiments/accuracy_first/configs/student_p2.yaml
