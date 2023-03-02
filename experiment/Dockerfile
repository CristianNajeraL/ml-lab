FROM continuumio/miniconda3:latest

COPY conda.yaml .

RUN conda env create -f conda.yaml && \
    conda clean -a

ENV PATH /opt/conda/envs/pycaret-with-docker/bin:$PATH

COPY main.py .

CMD ["python", "main.py"]
