import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from blink.main_dense import load_biencoder, _process_biencoder_dataloader
from typing import List

class Mention(BaseModel):
    context_left: str
    context_right:str
    mention: str
    # start_pos:int
    # end_pos: int
    # sent_idx:int

app = FastAPI()

@app.post('/api/blink/biencoder/encode/mention')
async def encode_mention(samples: List[Mention]):
    dataloader = _process_biencoder_dataloader(
        samples, biencoder.tokenizer, biencoder_params
    )
    encodings = _run_biencoder_mention(biencoder, dataloader)
    return encodings

def _run_biencoder_mention(biencoder, dataloader):
    biencoder.model.eval()
    encodings = []
    for batch in tqdm(dataloader):
        context_input, _, _ = batch
        with torch.no_grad():
            context_encoding = biencoder.encode_context(context_input).numpy()
            context_encoding = np.ascontiguousarray(context_encoding)
        encodings.extend([e.tolist() for e in context_encoding])
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

    biencoder, biencoder_params = load_models(args)


    main(args)