FROM pytorch
RUN pip install --upgrade pip
VOLUME /app
WORKDIR /app
COPY requirements.txt /app/
COPY *.py /app/
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
#CMD python subtitler.py