#!/usr/bin/env bash

dep_par_dir="/nas/clear/users/meryem/Datasets/DepParsing/data2.2_more/"
results_dir="/nas/clear/users/meryem/Results/DepPar/"


python main.py --data-dir $dep_par_dir --output-dir $results_dir --load-checkpoint --do-train --do-eval --do-predict \
--train-langs "en" --test-langs "en,ar,de,es,zh" --use-alignment --alignment-choice "schuster" --use-rnn \
--alignment-dict "{'en':'', 'ar':'/nas/clear/users/meryem/MUSE/dumped/debug/ar_en_new/best_mapping.pth', 'de': '/nas/clear/users/meryem/MUSE/dumped/debug/de_en/best_mapping.pth', 'es': '/nas/clear/users/meryem/MUSE/dumped/debug/es_en/best_mapping.pth', 'zh': '/nas/clear/users/meryem/MUSE/dumped/debug/zh_en_new/best_mapping.pth'}"