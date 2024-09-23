FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

MAINTAINER Sean Gasiorowski "sgaz@slac.stanford.edu"

ENV PYTHONUNBUFFERED=1

ARG SCRATCH_VOLUME=/scratch
ENV SCRATCH_VOLUME=/scratch
RUN echo creating ${SCRATCH_VOLUME} && mkdir -p ${SCRATCH_VOLUME}
VOLUME ${SCRATCH_VOLUME}

WORKDIR /work
ADD requirements.txt /work/requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git wget build-essential

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ADD . /work