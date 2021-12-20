import argparse
from fastapi import FastAPI
import uvicorn
import blink.ner as NER
from typing import List

app = FastAPI()

@app.post('/api/ner')
async def encode_mention(texts: List[str]):
    samples = _annotate(ner_model, texts)
    return samples

def _annotate(ner_model, input_sentences):
    ner_output_data = ner_model.predict(input_sentences)

    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentences[mention["sent_idx"]][
            : mention["start_pos"]
        ].lower()
        record["context_right"] = sentences[mention["sent_idx"]][
            mention["end_pos"] :
        ].lower()
        record["mention"] = mention["text"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30304", help="port to listen at",
    )

    args = parser.parse_args()

    print('Loading NER model...')
    # Load NER model
    ner_model = NER.get_model()
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
