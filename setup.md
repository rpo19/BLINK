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

Probably you won't be able to correctly obtain all the types at the first run. Re-run it until there are no more missing types.

### Feature selection
Use the notebook `notebooks/feature_selection.ipynb` for the feature selection.

### Feature ablation study
The feature ablation study is run on the AIDA dataset only.

Run it with:
```
python scripts/feature_ablation_study.py
```
At the end look inside the folder `output/feature_ablation_study`: you should find:
- the trained NIL prediction models named `*_model.pickle`;
- the performance report for each model named `*_report.txt`;
- the ROC curve for each model named `*_roc.png`;
- the density plot of targets and predictions for each model named `*_kde.png`;
- the density plot of correct and wrong predictions for each model named `*_kde_correct_errors.png`.
Finally there should be a summary of all the models' results together named `feature_ablation_summary.csv`. 

### Evaluate on ALL the datasets
Run
```
python scripts/evaluate_all_datasets_cross.py # cross encoder
python scripts/evaluate_all_datasets_bi.py # bi encoder
```
The results should be respectively inside `output/evaluate_all_datasets/cross/` and `output/evaluate_all_datasets/bi/`. They are similar to the results of the feature ablation study except that the summary is named `evaluation_summary.csv`. Note that these scripts require the models previously trained in the features ablation study.

### Human-in-the-loop Simulation
Use the notebook `notebooks/hitl_simulation.ipynb` to reproduce the Human-in-the-loop simulation.

### Get the encodings
The'll be used in the KBP simulations.
Run
```
python scripts/get_encodings_fast.py  './data/BLINK_benchmark/AIDA-YAGO2_train_ner.jsonl' './output/AIDA-YAGO2_train_encodings.jsonl'
python scripts/get_encodings_fast.py  './data/BLINK_benchmark/AIDA-YAGO2_testa_ner.jsonl' './output/AIDA-YAGO2_testa_encodings.jsonl'
python scripts/get_encodings_fast.py  './data/BLINK_benchmark/AIDA-YAGO2_testb_ner.jsonl' './output/AIDA-YAGO2_testb_encodings.jsonl'
```

### KBP Simulation (with NIL prediction)
Run the script `scripts/kbp_simulation.py` as follows:
```
medoid=true python scripts/kbp_simulation.py
medoid=false python scripts/kbp_simulation.py
```
The first time it uses the clusters' medoid for indexing; the second one it uses all the points.
It will run the KBP simulation "without NIL prediction" and then "with NIL prediction".

The results are available inside `output/kbp_simulation/medoid` and `output/kbp_simulation/all`. The results regarding
only the novel entities representation (that do not include the NIL prediction)
are named `kbp_nel_no_nil_summary*.txt` in the format of latex tables.

The rest of the results are similar to the ones of the feature ablation study
except that the summaries are is named `kbp_simulation_summary.csv` and
`kbp_simulation_oracle_summary.csv`. Even here several NIL prediction models
(among the previously trained during the feature ablation study) are tested
(effectively this is another feature ablation study for the NIL prediction but
in a kbp context) in a kbp context.

The "oracle" results are calculated only on the most confident samples; in this
setup, it is assumed the most uncertain decisions are left to a human (hitl).
