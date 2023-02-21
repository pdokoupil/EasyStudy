FROM continuumio/miniconda3:4.12.0

# RUN mkdir -p /app/src
WORKDIR /app
#ENV HOME=/app

COPY requirements.txt requirements.txt

RUN conda update conda --yes \ 
    && conda config --add channels conda-forge \
    && conda create --name app_env python=3.9 \
    && conda install --yes --file requirements.txt

# Some packages are not available in conda
RUN pip install --upgrade pip
RUN pip install Flask-PluginKit
RUN pip install Flask-Session
RUN pip install is-safe-url
RUN pip install tensorflow
RUN pip install tensorflow-recommenders

EXPOSE 5555

ENTRYPOINT [ "python" ]
CMD ["-m", "gunicorn", "-w", "1", "--bind", "0.0.0.0:5555", "app:create_app()"]
