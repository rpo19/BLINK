import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Union, List
import spacy
import sys
import itertools
import json
import requests
# from multiprocessing import Pool
from entity import EntityMention

from gatenlp import Document

class Item(BaseModel):
    text: str

app = FastAPI()

@app.post('/api/ner')
async def encode_mention(item: Item):
    samples = []
    sentences = []

    doc = Document(item.text)
    sentence_set = doc.annset('sentences')
    entity_set = doc.annset('entities')

    spacy_out = spacy_pipeline(doc.text)

    import pdb
    pdb.set_trace()

    tint_out = nlp_tint(doc.text)

    # sentences
    for sent in spacy_out.sents:
        # TODO keep track of entities in this sentence?
        sentence_set.add(sent.start_char, sent.end_char, "sentence")

    for ent in spacy_out.ents:
        # TODO keep track of sentences
        # sentence_set.overlapping(ent.start_char, ent.end_char)
        entity_set.add(ent.start_char, ent.end_char, ent.label_, {"ner": {"type": ent.label_, "score": 1.0}})

    for ent in tint_out:
        if ent.type_ == 'DATE':
            entity_set.add(ent.begin, ent.end, ent.type_, {"ner": {"type": ent.type_, "score": 1.0}})
        else:
            entity_set.add(ent.begin, ent.end, ent.type_, {"ner": {
                "type": ent.type_,
                "score": 1.0,
                "normalized_date": ent.attrs['normalized_date']
                }})

    # for i, sentence in zip(itertools.count(), input_sentences):
    #     doc, sentence = process_sent(sentence)

    #     if hasattr(doc, 'start_char') and hasattr(doc, 'end_char'):
    #         sent_start_pos = doc.start_char
    #         sent_end_pos = doc.end_char
    #         sentences.append({
    #                 'text': sentence,
    #                 'start_pos': sent_start_pos,
    #                 'end_pos': sent_end_pos,
    #             })
    #     else:
    #         sent_start_pos = 0
    #         sent_end_pos = len(sentence)
    #         sentences.append({
    #                 'start_pos': sent_start_pos,
    #                 'text': sentence,
    #                 'end_pos': sent_end_pos,
    #             })

    #     # spacy ents # pos are already ok
    #     for ent in doc.ents:
    #         sample = {
    #             'label': 'unknown',
    #             'label_id': -1,
    #             'context_left': sentence[:ent.start_char],
    #             'context_right': sentence[ent.end_char:],
    #             'mention': ent.text,
    #             'start_pos': ent.start_char,
    #             'end_pos': ent.end_char,
    #             'sent_idx': i,
    #             'ner_type': ent.label_
    #         }

    #         samples.append(sample)

    #     # TODO make it async inside spacy
    #     res_tint = nlp_tint(sentence)
    #     for ent in res_tint:
    #         if ent.type_ == 'DATE':
    #             # only dates
    #             start_pos = sent_start_pos + ent.begin
    #             end_pos = sent_start_pos + ent.end
    #             sample = {
    #                 'label': 'unknown',
    #                 'label_id': -1,
    #                 'context_left': sentence[:ent.begin],
    #                 'context_right': sentence[ent.end:],
    #                 'mention': ent.text,
    #                 # tint pos starts from the beginning of the sentence: to fix
    #                 'start_pos': start_pos,
    #                 'end_pos': end_pos,
    #                 'sent_idx': i,
    #                 'ner_type': ent.type_,
    #                 'normalized_date': ent.attrs['normalized_date']
    #             }

    #             samples.append(sample)
    #         else:
    #             # TODO only dates for now
    #             continue

    # return {
    #     'ents': samples,
    #     'sentences': sentences,
    # }

    return doc.to_dict()

def nlp_tint(text):
    global args

    # TODO async
    # tint_async = pool.apply_async(tint, (x, args.tint))
    # res_tint = tint_async.get()

    ents, res = tint(text, baseurl=args.tint)

    if res.ok:
        ents = EntityMention.group_from_tint(ents, '', False, doc=text)
    else:
        # tint error # TODO
        return []

    return ents

def tint(text, format_='json', baseurl='http://127.0.0.1:8012/tint'):
    if len(text) > 0:
        payload = {
            'text': text,
            'format': format_
        }
        res = requests.post(baseurl, data=payload)
        if res.ok:
            # success
            return json.loads(res.text), res
        else:
            return None, res
    else:
        print('WARNING: Tint Server Wrapper got asked to call tint with empty text.', file=sys.stderr)
        return None, None

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
    parser.add_argument(
        "--tint", type=str, default="http://127.0.0.1:8012/tint", help="tint URL",
    )

    # pool to run tint in parallel # TODO
    #pool = Pool(1)

    args = parser.parse_args()

    print('Loading spacy model...')
    # Load spacy model
    try:
        spacy_pipeline = spacy.load(args.model, exclude=['tok2vec', 'morphologizer', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    except Exception as e:
        print('ERROR.')
        print(e)
        if "Can't find model" in str(e):
            print('Maybe you did not download the model. To download it run ```python -m spacy download $MODEL```.')
        sys.exit(1)
    spacy_pipeline.enable_pipe('senter')

    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
