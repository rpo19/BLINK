import redis


title2id = {}
id2title = {}
id2text = {}
wikipedia_id2local_id = {}
local_id2wikipedia_id = {}
id2url = {}

def load_models(args):
    print('Loading entities')
    local_idx = 0
    with open(args.entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1

    for k,v in wikipedia_id2local_id.items():
        local_id2wikipedia_id[v] = k

    id2url = {
        v: "https://en.wikipedia.org/wiki?curid=%s" % k
        for k, v in wikipedia_id2local_id.items()
    }

def populate():
    # db 0 --> id2title
    r = redis.Redis(host='localhost', port=6379, db=0)
    for k, v in id2title.items():
        r.set(k, v)
    r.close()

    # db 1 --> id2text
    r = redis.Redis(host='localhost', port=6379, db=1)
    for k, v in id2text.items():
        r.set(k, v)
    r.close()

    # db 2 --> local_id2wikipedia_id
    r = redis.Redis(host='localhost', port=6379, db=2)
    for k, v in local_id2wikipedia_id.items():
        r.set(k, v)
    r.close()

    # ...



if __name__ == '__main__':
parser = argparse.ArgumentParser()
    # indexer
    parser.add_argument(
        "--redis-host", type=str, default="127.0.0.1", help="redis host",
    )
    parser.add_argument(
        "--redis-port", type=int, default="6379", help="redis port",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )

    print('Loading entities...')
    load_models()

    print('Populating redis...')
    populate()

