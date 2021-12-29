import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Union, List
import spacy
import sys
import preprocess
import itertools

class Item(BaseModel):
    text: Union[List[str], str]

app = FastAPI()

@app.post('/api/ner')
async def encode_mention(item: Item):
    samples = []
    sentences = []

    if isinstance(item.text, str):
        text = item.text
        text, mapping, mapping_back = preprocess.preprocess(text)
        input_sentences = nlp(text).sents
        process_sent = lambda x: (x, x.text)
        get_mapping_back = itertools.repeat(mapping_back)
    else:
        input_sentences = item.text
        input_sentences, mapping, mapping_back = tuple(zip(*map(lambda x: preprocess.preprocess(x), input_sentences)))
        process_sent = lambda x: (nlp(x), x)
        get_mapping_back = mapping_back

    for i, sentence, _mapping_back in zip(itertools.count(), input_sentences, get_mapping_back):
        doc, sentence = process_sent(sentence)

        if hasattr(doc, 'start_char') and hasattr(doc, 'end_char'):
            start_pos_original, end_pos_original = preprocess.mapSpan((doc.start_char, doc.end_char), _mapping_back)
            sentences.append({
                    'text': sentence,
                    'start_pos': doc.start_char,
                    'end_pos': doc.end_char,
                    'start_pos_original': start_pos_original,
                    'end_pos_original': end_pos_original
                })
        else:
            start_pos_original, end_pos_original = preprocess.mapSpan((0, len(sentence)), _mapping_back)
            sentences.append({
                    'start_pos': 0,
                    'text': sentence,
                    'end_pos': len(sentence),
                    'start_pos_original': start_pos_original,
                    'end_pos_original': end_pos_original
                })

        for ent in doc.ents:
            start_pos_original, end_pos_original = preprocess.mapSpan((ent.start_char, ent.end_char), _mapping_back)
            sample = {
                'label': 'unknown',
                'label_id': -1,
                'context_left': sentence[:ent.start_char],
                'context_right': sentence[ent.end_char:],
                'mention': ent.text,
                'start_pos': ent.start_char,
                'end_pos': ent.end_char,
                'start_pos_original': start_pos_original,
                'end_pos_original': end_pos_original,
                'sent_idx': i,
                'ner_type': ent.label_
            }

            samples.append(sample)

    return {
        'ents': samples,
        'sentences': sentences,
        'preprocess': {
            'mapping': mapping
        }
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30304", help="port to listen at",
    )
    parser.add_argument(
        "--model", type=str, default="en_core_web_sm", help="spacy model to load",
    )

    args = parser.parse_args()

    print('Loading spacy model...')
    # Load spacy model
    try:
        nlp = spacy.load(args.model, exclude=['tok2vec', 'morphologizer', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    except Exception as e:
        print('ERROR.')
        print(e)
        if "Can't find model" in str(e):
            print('Maybe you did not download the model. To download it run ```python -m spacy download $MODEL```.')
        sys.exit(1)
    nlp.enable_pipe('senter')
    
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
