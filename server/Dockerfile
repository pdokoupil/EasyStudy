FROM continuumio/miniconda3:4.12.0

WORKDIR /app

COPY conda_requirements.txt conda_requirements.txt
COPY pip_requirements.txt pip_requirements.txt

RUN conda update conda --yes \ 
    && conda config --add channels conda-forge \
    && conda create --name app_env python=3.9 \
    && conda install --yes --file conda_requirements.txt

# Some packages are not available in conda
RUN pip install --upgrade pip
RUN pip install -r pip_requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python" ]
CMD ["-m", "gunicorn", "-w", "1", "--bind", "0.0.0.0:5000", "--timeout", "0", "--preload", "--log-level", "debug", "--access-logfile", "/app/access_logfile.txt", "--error-logfile", "/app/error_logfile.txt", "app:create_app()"]