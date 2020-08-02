@echo off

set CODE_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\code\HIDENIC_overtime_analysis"

set DATA_DIR=E:\Data\HIDENIC_EXTRACT_OUTPUT_DIR\POP_SIZE_0\ITEMID_REP
set OUTPUT_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\output\HIDENIC_overtime_analysis\overall_overtime_fselect"
mkdir %OUTPUT_DIR%

set LOAD_FILTERED_DATA=1
set TRAIN_TYPES=overall_overtime
set TRAIN_YEARS=2008 2009 2010
set MODEL_TYPES=lr rf nb rbf-svm
set SEED=0
set TARGETS=mort_icu los_3
set representation=raw pca
set save_data=1

python -u %CODE_DIR%/experiments.py ^
    --data_dir %DATA_DIR% ^
    --output_dir %OUTPUT_DIR% ^
    --representation %representation% ^
    --train_types %TRAIN_TYPES% ^
    --model_types %MODEL_TYPES% ^
    --random_seed %SEED% ^
    --target_list %TARGETS% ^
    --load_filtered_data %LOAD_FILTERED_DATA% ^
    --save_data %save_data% ^
    --train_years %TRAIN_YEARS% ^
    --feature_selection 1 ^
    --K 100 200 500 1000 2000 ^
    --n_threads 1 ^