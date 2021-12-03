# BLINK setup conda

### Create conda env
```
conda create -n blink37 -y python=3.7
```

### Activate conda env
Remember to activte it whenever you'll use BLINK.
```
conda activate blink37
```

### Install gcc
```
apt update && apt install gcc
```

### Enter BLINK folder
```
cd BLINK
```

### Install BLINK requirements
```
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