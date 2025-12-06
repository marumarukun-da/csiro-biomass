#!/bin/bash
set -e
python src/download.py competition_download || { echo "Kaggle push failed"; exit 1; }
