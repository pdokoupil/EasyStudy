FROM continuumio/miniconda3:4.12.0

# RUN mkdir -p /app/src
WORKDIR /app
#ENV HOME=/app

COPY conda_requirements.txt conda_requirements.txt
COPY pip_requirements.txt pip_requirements.txt

RUN conda update conda --yes \ 
    && conda config --add channels conda-forge \
    && conda create --name app_env python=3.9 \
    && conda install --yes --file conda_requirements.txt

# Some packages are not available in conda
RUN pip install --upgrade pip
RUN pip install -r pip_requirements.txt

EXPOSE 5555

ENTRYPOINT [ "python" ]
CMD ["-m", "gunicorn", "-w", "4", "--bind", "0.0.0.0:5555", "app:create_app()"]
