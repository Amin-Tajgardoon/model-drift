@echo off

set CODE_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\code\HIDENIC_overtime_analysis"

set DATA_DIR=E:\Data\HIDENIC_EXTRACT_OUTPUT_DIR\POP_SIZE_0\ITEMID_REP
set TRAIN_TYPES=rolling
set MODEL_TYPES=lr rf
set /a SEED=0
set TARGETS=mort_icu

python -u %CODE_DIR%/experiments.py ^
    --data_dir %DATA_DIR% ^
    --representation raw ^
    --train_types %TRAIN_TYPES% ^
    --model_types %MODEL_TYPES% ^
    --random_seed %SEED% ^
    --target_list %TARGETS% ^