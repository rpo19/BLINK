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

<!-- ### Downgrade gensim
```
pip install gensim==3.8.3
``` -->

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
- Run `python scripts/create_BLINK_benchmark_data_with_NME.py` to convert all the dataset but AIDA to the format accepted by BLINK. 
- Run `python scripts/create_BLINK_benchmark_data_with_NME_and_ner.py` to convert AIDA to the format accepted by BLINK. 

### Get BLINK linking scores
Run
```
python blink/run_benchmark_save_scores_indexer.py
```
You should find the scores inside `data/scores`.

### Prepare the NIL prediction dataset
First run
```
python blink/main_dense.py --save-id2title --save-wikipedia-id2local-id
```
to obtain two required files; the script should terminate with an error but only after having saved `id2title.pickle` and `wikipedia_id2local_id.pickle`.


Then Run
```
python scripts/prepare_nil_dataset.py
```
You should find the dataset at `data/nil_dataset.pickle`.

### Get NER types from Wikidata
Get the types and update the nil dataset with
```
python scripts/get_types_from_wikidata.py
```
Since Wikidata stops answering if called repeatedly, only one call per second is performed. Therefore this step requires few hours.

### Run feature selection notebook
Improve the notebook.

### Train and test NIL prediction models
Using the scripts. For the Feature ablation study.

### Evaluate on ALL the datasets

### KBP Simulation

### HITL Simulation

### KBP Simulation with NIL prediction