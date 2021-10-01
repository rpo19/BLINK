#!/bin/bash
python blink/main_dense.py --faiss_index flat --index_path models/faiss_flat_index.pkl --test_mentions data/BLINK_benchmark/AIDA-YAGO2_testa.jsonl --fast --save_encodings output/encodings/AIDA-YAGO2_testa_encodings.jsonl --keep_all --consider_all
python blink/main_dense.py --faiss_index flat --index_path models/faiss_flat_index.pkl --test_mentions data/BLINK_benchmark/AIDA-YAGO2_testb.jsonl --fast --save_encodings output/encodings/AIDA-YAGO2_testb_encodings.jsonl --keep_all --consider_all
python blink/main_dense.py --faiss_index flat --index_path models/faiss_flat_index.pkl --test_mentions data/BLINK_benchmark/AIDA-YAGO2_train.jsonl --fast --save_encodings output/encodings/AIDA-YAGO2_train_encodings.jsonl --keep_all --consider_all
