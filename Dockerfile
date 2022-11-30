FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade wheel
RUN apt install ffmpeg libsndfile1-dev
VOLUME /app
WORKDIR /app
COPY requirements.txt /app/
COPY *.py /app/
RUN pip install -r requirements.txt
#CMD python subtitler.py