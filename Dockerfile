# Edit the base image here, e.g., to use 
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/) 
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
RUN  pip install --upgrade pip

# Copy all required files so that they are available within the docker image 
# All the codes, weights, anything you need to run the algorithm!
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm entrypoint.sh /opt/algorithm/
COPY --chown=algorithm:algorithm model.pth /opt/algorithm/
COPY --chown=algorithm:algorithm resnet50-19c8e357.pth  /home/algorithm/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
COPY --chown=algorithm:algorithm training_utils /opt/algorithm/training_utils

# Install required python packages via pip - please see the requirements.txt and adapt it to your needs
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

# Entrypoint to run, entypoint.sh files executes process.py as a script
ENTRYPOINT ["bash", "entrypoint.sh"]

## ALGORITHM LABELS: these labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=noduledetection
# These labels are required and describe what kind of hardware your algorithm requires to run for grand-challenge.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=2
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=12G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=10G


