#!/bin/bash 

cd $CODE
python -m scripts.generate_gnn_predictions -d ${TEST_FOLDER} -o ${OUT_FOLDER}_tmp -w ${CODE}/logs/model0904_f1_r -f logits 
python -m scripts.logits_to_preds -r ${OUT_FOLDER}_tmp -t ${OUT_FOLDER}