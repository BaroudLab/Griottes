FROM jupyter/datascience-notebook:latest
WORKDIR /home/jovyan
COPY . .
RUN pip install -e . &&\
    pip install opencv-python-headless
CMD python -V && jupyter notebook