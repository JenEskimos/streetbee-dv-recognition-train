FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  build-essential \
  libssl-dev \
  libffi-dev \
  python3-dev \
  python3-venv \
  curl \
  protobuf-compiler \
  python3-tk \
  unzip \
  git

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED 1
ENV HOME /root
RUN pip3 install --upgrade pip

RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app/
RUN pip3 install -r requirements.txt
ADD . /app/

WORKDIR /app/mmdetection
RUN pip install --no-cache-dir -e .
WORKDIR /app