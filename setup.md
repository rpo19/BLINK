## Conda environment setup

### Create conda env
```
conda create -n blink37 -y python=3.7
```

### Activate conda env
Remember to activte it whenever you'll use BLINK.
```
conda activate blink37
```

### Install gcc (if under Linux)
It could be required to compile python libraries. If you are under Windows you may need to have a `C` compiler installed.
```
apt update && apt install gcc
```

### Install BLINK requirements
```
cd BLINK
pip install -r requirements.txt
```

### Download BLINK models
```
chmod +x download_blink_models.sh 
./download_blink_models.sh 
```

### Download FAISS indexes
You probably won't need both at the same time.
```
wget http://dl.fbaipublicfiles.com/BLINK/faiss_flat_index.pkl
wget http://dl.fbaipublicfiles.com/BLINK/faiss_hnsw_index.pkl
```

### Downgrade gensim
```
pip install gensim==3.8.3
```

## Datasets
### Download BLINK datasets
```
bash scripts/get_train_and_benchmark_data.sh
```
Now you should see the datasets downloaded inside `data/train_and_benchmark_data/`.

### Patch AIDA dataset
AIDA is patched to include also NER types from CoNLL. In order to do this it is required to re-construct AIDA starting from CoNLL using a patched version of the creation jar.
- Follow the instructions at [aida-patch/README.md](aida-patch/README.md) to obtain the dataset `AIDA-YAGO2-dataset.tsv` with NER types too.
- Copy and rename the dataset to `data/train_and_benchmark_data/basic_data/test_datasets/AIDA/AIDA-YAGO2-dataset-ner.tsv`.
- Run `python scripts/create_BLINK_benchmark_data_with_NME_and_ner.py` to convert the dataset in the format accepted by BLINK. 

