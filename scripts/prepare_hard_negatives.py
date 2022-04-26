
import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser
from tqdm import trange

def main(params):
    input_samples = utils.read_dataset("train", params["data_path"], compression='gzip',
        max=params['max_dataset'], sample=params['sample_dataset'], seed=params['sample_dataset_seed'])

    batch_size = params['batch_size']

    x = 0
    for i in trange(0, len(input_samples), batch_size):
        batch = input_samples[i:i+batch_size]
        x+=len(batch)

    assert x == len(input_samples)

if __name__ == '__main__':
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()
    parser.add_argument(
        "--max-dataset", default=None, type=int, dest='max_dataset',
        help="Limit the dataset to this size."
    )
    parser.add_argument(
        "--batch-size", default=None, type=int, dest='batch_size',
        help="Batch size for creating hard negs dataset."
    )
    parser.add_argument(
        "--sample-dataset", default=None, type=int, dest='sample_dataset',
        help="Sample the dataset to this size."
    )
    parser.add_argument(
        "--sample-dataset-seed", default=None, type=int, dest='sample_dataset_seed',
        help="Sample with this seed."
    )
    parser.add_argument(
        #TODO int for how many hard negatives?
        "--hard-negatives", action="store_true", help="Whether to use hard-negatives.",
        dest='hard_negatives', default=False
    )
    parser.add_argument(
        "--biencoder-url",
        default=None,
        type=str,
        help="The url of the biencoder.",
        dest='biencoder_url'
    )
    parser.add_argument(
        "--indexer-url",
        default=None,
        type=str,
        help="The url of the indexer from where to extract hard negatives.",
        dest='indexer_url'
    )
    parser.add_argument(
        "--entities-path",
        default=None,
        type=str,
        help="Path of the entities from which to get descriptions.",
        dest='entities_path'
    )

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)