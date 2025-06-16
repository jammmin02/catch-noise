FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /workspace

COPY . /workspace

RUN pip install --upgrade pip && \
    pip install -r Docker/requirements.txt
