import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from blink.main_dense import load_biencoder, _process_biencoder_dataloader
from typing import List
import json
from tqdm import tqdm
import torch
import numpy as np
import base64

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

class Mention(BaseModel):
    label:str
    label_id:int
    context_left: str
    context_right:str
    mention: str
    start_pos:int
    end_pos: int
    sent_idx:int

app = FastAPI()

@app.post('/api/blink/biencoder/mention')
async def encode_mention(samples: List[Mention]):
    samples = [dict(s) for s in samples]
    dataloader = _process_biencoder_dataloader(
        samples, biencoder.tokenizer, biencoder_params
    )
    encodings = _run_biencoder_mention(biencoder, dataloader)
    assert encodings[0].dtype == 'float32'
    #assert np.array_equal(encodings[0], vector_decode(vector_encode(encodings[0]), np.float32))
    ## dtype float32
    encodings = [vector_encode(e) for e in encodings]
    return {'samples': samples, 'encodings': encodings}

def _run_biencoder_mention(biencoder, dataloader):
    biencoder.model.eval()
    encodings = []
    for batch in tqdm(dataloader):
        context_input, _, _ = batch
        with torch.no_grad():
            context_encoding = biencoder.encode_context(context_input).numpy()
            context_encoding = np.ascontiguousarray(context_encoding)
        encodings.extend(context_encoding)
    return encodings

def load_models(args):
    # load biencoder model
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)
    return biencoder, biencoder_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # biencoder
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        default="models/biencoder_wiki_large.bin",
        help="Path to the biencoder model.",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        default="models/biencoder_wiki_large.json",
        help="Path to the biencoder configuration.",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )

    parser.add_argument(
        "--port", type=int, default="30300", help="port to listen at",
    )

    args = parser.parse_args()

    print('Loading biencoder...')
    biencoder, biencoder_params = load_models(args)
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
