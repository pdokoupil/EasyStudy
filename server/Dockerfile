FROM continuumio/miniconda3:4.12.0

WORKDIR /app

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
RUN pip install python3-discogs-client
RUN pip install cachelib
#RUN pip install lenskit
RUN pip install pytest

EXPOSE 5000

ENTRYPOINT [ "python" ]
CMD ["-m", "gunicorn", "-w", "1", "--bind", "0.0.0.0:5000", "--timeout", "0", "--preload", "--log-level", "debug", "--access-logfile", "/app/access_logfile.txt", "--error-logfile", "/app/error_logfile.txt", "app:create_app()"]