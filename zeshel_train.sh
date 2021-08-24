#!/bin/bash
# based on paper appendix A.2 and params read from blink/biencoder/train_biencoder.py
#PYTHONPATH=. python blink/biencoder/train_biencoder.py --evaluate --output_eval_file examples/zeshel/data/zeshel/output_eval.txt --train_batch_size 128 --learning_rate 2e-5 --num_train_epochs 5 --max_context_length 128 --output_path examples/zeshel/data/zeshel/output/output_eval.txt --seed 42 --data_path examples/zeshel/data/zeshel/blink_format # cuda out of mem
# trying with batch_size 64
PYTHONPATH=. python blink/biencoder/train_biencoder.py --evaluate --output_eval_file examples/zeshel/data/zeshel/output_eval.txt --train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 5 --max_context_length 128 --output_path examples/zeshel/data/zeshel/output/output_eval.txt --seed 42 --data_path examples/zeshel/data/zeshel/blink_format
# failed even with 32
