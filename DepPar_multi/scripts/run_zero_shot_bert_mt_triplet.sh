#!/usr/bin/env bash

dep_par_dir="/nas/clear/users/meryem/Datasets/DepParsing/data2.2_more/"
results_dir="/nas/clear/users/meryem/Results/DepPar/"
xintr_dir="/nas/clear/users/meryem/Datasets/XSTS"
stsb_dir="/nas/clear/users/meryem/sentence-transformers/STS2017-extended"

python main.py --data-dir $dep_par_dir --output-dir $results_dir --load-checkpoint --do-train --do-eval --use-rnn \
--do-predict --train-langs "en" --test-langs "en,ar,de,es,zh" --use-multi-task --xintr-path $xintr_dir --stsb-path $stsb_dir