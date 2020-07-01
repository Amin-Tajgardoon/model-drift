@echo off

set CODE_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\code\HIDENIC_overtime_analysis"

set DATA_DIR=E:\Data\HIDENIC_EXTRACT_OUTPUT_DIR\POP_SIZE_0\ITEMID_REP
set OUTPUT_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\output\HIDENIC_overtime_analysis\with_month_intervals"
mkdir %OUTPUT_DIR%

set TRAIN_TYPES= first_years
set MODEL_TYPES=rf lr rbf-svm 1class_svm 1class_svm_novel iforest
set SEED=0
set TARGETS=mort_icu los_3
set test_month_interval=2

python -u %CODE_DIR%/experiments.py ^
    --data_dir %DATA_DIR% ^
    --output_dir %OUTPUT_DIR% ^
    --representation pca ^
    --train_types %TRAIN_TYPES% ^
    --model_types %MODEL_TYPES% ^
    --random_seed %SEED% ^
    --target_list %TARGETS% ^
    --test_month_interval %test_month_interval% ^
    --n_threads 6 ^