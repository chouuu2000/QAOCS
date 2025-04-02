#!/bin/bash

apt install python3.8-venv

# add-apt-repository ppa:deadsnakes/ppa -y
# apt install python3.10-venv -y

### BUILD LIBVMAF
  git config --global http.postBuffer 524288000
  git clone https://github.com/chouuu2000/vmaf.git && \
  cd vmaf && \
  PYTHONPATH=/vmaf/python/src:/vmaf:$PYTHONPATH PATH=/vmaf:/vmaf/src/libvmaf:$PATH make -j 24 && \
  make install

