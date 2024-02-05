#!/bin/bash

conda create --name asc_final python=3.8
conda activate asc_final

yes | pip install soundata==0.1.1 --no-deps
yes | pip install pandas
yes | pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
yes | pip install torchlibrosa==0.0.9
yes | pip install torch-audiomentations
yes | pip install audiomentations==0.24.0

