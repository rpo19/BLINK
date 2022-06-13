FROM python:3.7

WORKDIR /home/app

COPY . .

RUN pip install --no-cache-dir --upgrade -r /home/app/requirements.txt

CMD python main.py --host 0.0.0.0 --port 80 --model $SPACY_MODEL --tint http://tint:8012/tint