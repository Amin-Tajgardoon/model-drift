@echo off

set CODE_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\code\HIDENIC_overtime_analysis"

set DATA_DIR=E:\Data\HIDENIC_EXTRACT_OUTPUT_DIR\POP_SIZE_0\ITEMID_REP
set OUTPUT_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\output\HIDENIC_overtime_analysis"

set TRAIN_TYPES= rolling rolling_limited first_years
set MODEL_TYPES=lr rf
set SEED=0
set TARGETS=mort_icu los_3

python -u %CODE_DIR%/experiments.py ^
    --data_dir %DATA_DIR% ^
    --output_dir %OUTPUT_DIR% ^
    --representation pca ^
    --train_types %TRAIN_TYPES% ^
    --model_types %MODEL_TYPES% ^
    --random_seed %SEED% ^
    --target_list %TARGETS% ^