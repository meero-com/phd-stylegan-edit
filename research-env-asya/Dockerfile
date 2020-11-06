FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /


# install packages for `rd-tools` and for `raw-converter`
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        libdc1394-22 \
        libjpeg-turbo-progs \
        libimage-exiftool-perl \
        libexiv2-dev \
        libz-dev \
        libexpat-dev \
        libjpeg-dev \
        cmake \
        language-pack-en \
        libsm6 \
        libxext6 \
        libxrender-dev \
        curl \
        git \
        python3-pip \
        python3.7 \
        python3.7-dev && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*


    
# use python3 by default
RUN  update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.7 2 \
    && update-alternatives  --set python /usr/bin/python3.7 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2 \
    && update-alternatives  --set python3 /usr/bin/python3.7 \
    && python -m pip install --upgrade setuptools pip wheel

# pip packages
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm -r requirements.txt

# install jupyterlab
COPY jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json
RUN apt-get -y update \
    && apt-get install -y curl \
    && curl -sL https://deb.nodesource.com/setup_14.x | bash - \
    && apt install -y nodejs \
    && jupyter labextension install @jupyterlab/toc \
    && pip install jupyterlab-nvdashboard \
    && jupyter labextension install jupyterlab-nvdashboard

# install zsh and oh-my-zsh
RUN apt update \
    && apt install -y zsh \
    && sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

