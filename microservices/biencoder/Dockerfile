# Base Image
# FROM tensorflow/tensorflow:latest-gpu
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Upgrade PIP
RUN python3 -m pip install --upgrade pip

# Install IPython kernel
RUN python3 -m pip install ipykernel && \
    python3 -m ipykernel install --user
    
# Install utils
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        sudo git zsh curl wget \
        gcc g++ build-essential cmake build-essential \
        unzip libjpeg-dev libpng-dev libpq-dev \
        imagemagick gnuplot gnuplot-x11 libssl-dev libzmq3-dev graphviz

WORKDIR /home/app

COPY . .

RUN pip install --no-cache-dir --upgrade -r /home/app/requirements.txt

# RUN pip install -e git+https://github.com/facebookresearch/BLINK.git#egg=BLINK