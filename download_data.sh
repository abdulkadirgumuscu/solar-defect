#!/usr/bin/env bash
# Download the public ELPV dataset (2,624 electroluminescence images of solar cells).
# License: CC BY-NC-SA 4.0, so the images are not stored in this repository.
set -e
mkdir -p data
if [ -d data/elpv-dataset ]; then
  echo "data/elpv-dataset already exists"
else
  git clone --depth 1 https://github.com/zae-bayern/elpv-dataset data/elpv-dataset
fi
echo "Images: $(ls data/elpv-dataset/src/elpv_dataset/data/images | wc -l)"
