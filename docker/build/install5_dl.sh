#!/bin/bash

# Add any additional installation commands here
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python opencv-python-headless
# Install ffmpeg_quality_metrics version 3.3.0
pip install ffmpeg_quality_metrics==3.3.0
pip install moviepy==1.0.3
pip uninstall -y pillow
pip install pillow==9.5.0
pip install tensorboard