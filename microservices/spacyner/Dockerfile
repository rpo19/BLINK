FROM python:3.7

ARG SPACY_MODEL

WORKDIR /home/app

COPY . .

RUN pip install --no-cache-dir --upgrade -r /home/app/requirements.txt

RUN python -m spacy download $SPACY_MODEL

CMD python main.py --host 0.0.0.0 --port 80 --model $SPACY_MODEL --tint http://tint:8012/tint