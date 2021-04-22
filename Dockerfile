FROM python

# EXPOSE 8080/tcp

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev\
    && apt-get install -y nocache\
    && apt-get install enchant -y\
    && apt-get install tesseract-ocr -y\
    && apt-get install tesseract-ocr-eng -y\
    && apt-get install libsm6 -y\
    && apt-get install libxrender1 -y\
    && apt-get install libfontconfig1 -y\
    && apt-get install libice6 -y\
    && apt-get install libgl1 -y\
    && apt-get install libarchive13 -y\
    && apt-get install libtesseract-dev -y\
    && apt-get install libglib2.0-0 -y\
    && apt-get install libglib2.0-dev -y


RUN pip install --upgrade pip

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["gunicorn","wsgi:app"]