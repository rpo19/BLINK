from tqdm import trange
import psycopg
import argparse
from annoy import AnnoyIndex
import numpy as np
import base64

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # indexer
    parser.add_argument(
        "--postgres", type=str, default="", help="postgresql url (e.g. postgresql://username:password@localhost:5432/mydatabase)",
    )
    parser.add_argument(
        "--table-name",
        dest="table_name",
        type=str,
        default=None,  # ALL WIKIPEDIA!
        help="Postgres table name.",
    )
    parser.add_argument(
        "--dims",
        dest="dims",
        type=int,
        default=1024,
        help="vector dimension.",
    )
    parser.add_argument(
        "--trees",
        dest="trees",
        type=int,
        default=10,
        help="number of trees.",
    )
    parser.add_argument(
        "--save",
        dest="save",
        type=str,
        default=None,  # ALL WIKIPEDIA!
        help="Where to save the index.",
    )

    args = parser.parse_args()

    assert args.table_name is not None, 'Error: table-name is required!'
    assert args.save is not None, 'Error: --save is required!'

    connection = psycopg.connect(args.postgres)

    with connection.cursor() as cur:
        cur.execute('select count(*) from {};'.format(args.table_name))
        n_rows = cur.fetchone()[0]

    named_cursor = connection.cursor(name='embeddings')

    named_cursor.execute('select id, embedding from {};'.format(args.table_name))

    index = AnnoyIndex(args.dims, 'dot')

    for i in trange(0, n_rows):
        v_id, v_enc = named_cursor.fetchone()
        v_vector = vector_decode(v_enc)

        index.add_item(v_id, v_vector)

    print('Building with {} trees'.format(args.trees))
    index.build(args.trees)
    print('Saving to {}'.format(args.save))
    index.save(args.save)
