import argparse
import redis

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

def populate(args, entity_encodings):
    r = redis.Redis(host='localhost', port=6379, db=3)

    assert entity_encodings[0].numpy().dtype == 'float32'

    max_shape = entity_encodings.shape[0]
    for i, tensor in enumerate(entity_encodings):
        string_encoded = vector_encode(tensor.numpy())
        r.set(i, string_encoded)
        if i % 1000 == 0:
            print('\r{}/{}'.format(i, max_shape), end = '')
    print()


def load_models(args):
    return torch.load(entity_encoding)

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
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
        default="models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    
    args = parser.parse_args()

    print('Loading entities...')
    entity_encodings = load_models(args)

    print('Populating redis...')
    populate(args, entity_encodings)
