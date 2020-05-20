#!/usr/bin/env bash

dep_par_dir="/nas/clear/users/meryem/Datasets/DepParsing/data2.2_more/"
results_dir="/nas/clear/users/meryem/Results/DepPar/"


python main.py --data-dir $dep_par_dir --output-dir $results_dir --load-checkpoint --do-train --do-eval --do-predict \
--train-langs "en" --test-langs "en,ar,de,es,zh" --use-alignment --alignment-choice "gd"  --use-rnn \
--alignment-dict "{'en':'', 'ar':'/nas/clear/users/meryem/Results/Alignment/CLBT/models/trained/gd/gd.en-ar.new.model/best_mapping.pkl', 'de': '/nas/clear/users/meryem/Results/Alignment/CLBT/models/downloaded/gd/gd.de-en/best_mapping.pkl', 'es': '/nas/clear/users/meryem/Results/Alignment/CLBT/models/downloaded/gd/gd.es-en/best_mapping.pkl', 'zh': '/nas/clear/users/meryem/Results/Alignment/CLBT/models/trained/gd/gd.en-zh.model/best_mapping.pkl'}"