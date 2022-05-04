FROM python:3.7

ARG SPACY_MODEL

WORKDIR /home/app

COPY . .

RUN pip install --no-cache-dir --upgrade -r /home/app/requirements.txt

RUN python -m spacy download $SPACY_MODEL