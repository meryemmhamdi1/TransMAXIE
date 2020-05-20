#!/usr/bin/env bash

dep_par_dir="/Users/d22admin/USCGDrive/ISI/BETTER/4.Implementation/TransMAXIE/DepPar_multi/sample_data/"
results_dir="/Users/d22admin/USCGDrive/ISI/BETTER/5.Results/DepPar_multi/"


python main.py --data-dir $dep_par_dir --output-dir $results_dir --load-checkpoint --do-train --do-eval \
--do-predict --train-langs "en" --test-langs "en" --batch-size 16