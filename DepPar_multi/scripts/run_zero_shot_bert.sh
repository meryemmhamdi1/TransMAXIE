#!/usr/bin/env bash

dep_par_dir="/nas/clear/users/meryem/Datasets/DepParsing/data2.2_more/"
results_dir="/nas/clear/users/meryem/Results/DepPar/"


python main.py --data-dir $dep_par_dir --output-dir $results_dir --load-checkpoint --do-train --do-eval \
--do-predict --train-langs "en" --test-langs "en,ar,de,es,zh" --use-rnn