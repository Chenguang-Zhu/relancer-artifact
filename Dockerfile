# Dockerfile

FROM continuumio/anaconda3

MAINTAINER  Chenguang Zhu <cgzhu@utexas.edu>

ENV LANG C.UTF-8

# Install sofware properties common
RUN \
  apt-get update && \
  apt-get install -y software-properties-common

# Install python3
RUN \
  apt-get update && \
  apt-get install -y python3 python3-dev python3-pip python3-virtualenv && \
  pip3 install goto-statement

# Install git
RUN \
  apt-get install -y git && \
  git --version

# Install sudo
RUN \
  apt-get update && \
  apt-get install -y sudo

# Install some text editors
RUN \
  apt-get update && \
  apt-get install -y emacs vim && \
  rm -rf /var/lib/apt/lists/*

# Set git user info
RUN \
  git config --global user.email "artifact@example.com" && \
  git config --global user.name "Artifact"

# Set up working environment
RUN \
  mkdir -p /home/relancer
COPY . /home/relancer
WORKDIR /home/relancer

# Create conda environment
RUN \
conda env create -f environment.yml

ENTRYPOINT /bin/bash
