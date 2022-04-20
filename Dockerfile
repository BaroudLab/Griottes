FROM jupyter/scipy-notebook:latest
WORKDIR /home/jovyan
RUN git clone https://github.com/BaroudLab/Griottes.git &&\
    cd Griottes &&\
    pip install .
EXPOSE 8888
WORKDIR /home/jovyan/Griottes/example_notebooks/
CMD ["jupyter", "lab"]