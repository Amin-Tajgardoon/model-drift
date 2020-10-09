#!/bin/bash
#

CODE_DIR="~\proposal\HIDENIC_overtime_analysis"
data_dir="~\proposal\data"
out_dir="~\proposal\data"
exp_type="mv_test"
source_train_type="single_site"
n_threads=24

python -u $CODE_DIR/drift_detection.py \
    --exp_type $exp_type \
    --dir_path $data_dir \
    --out_dir $out_dir \
    --source_train_type $source_train_type \
    --n_threads $n_threads \