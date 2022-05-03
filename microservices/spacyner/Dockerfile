FROM python:3.7

WORKDIR /home/app

COPY . .

RUN pip install --no-cache-dir --upgrade -r /home/app/requirements.txt

RUN python -m spacy download it_core_news_sm