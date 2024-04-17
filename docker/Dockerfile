FROM nvcr.io/nvidia/pytorch:23.08-py3

# Download and setup conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && chmod +x Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3
ENV PATH /workspace/miniconda3/bin:$PATH
RUN conda update conda
RUN conda create -n incasem --no-default-packages python=3.9
RUN conda init
SHELL ["conda", "run", "-n", "incasem", "/bin/bash", "-c"]

# Install necessary packages
RUN pip install -e git+https://github.com/kirchhausenlab/incasem.git@docker#egg=incasem
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install nvidia-pyindex
RUN pip install nvidia-dlprof[pytorch]

# Set LD_LIBRARY_PATH to include torch libraries so it does not use default container library
COPY entrypoint.sh ./

RUN mkdir -p /workspace/src/incasem/data

ENTRYPOINT ["./entrypoint.sh"]
