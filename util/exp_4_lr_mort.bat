@echo off

set CODE_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\code\HIDENIC_overtime_analysis"

set DATA_DIR=E:\Data\HIDENIC_EXTRACT_OUTPUT_DIR\POP_SIZE_0\ITEMID_REP
set TRAIN_TYPE=first_years
set MODEL_TYPE=lr
set /a SEED=0
set TARGET=mort_icu

python -u %CODE_DIR%/experiments.py ^
    --representation raw ^
    --train_type %TRAIN_TYPE% ^
    --modeltype %MODEL_TYPE% ^
    --random_seed %SEED% ^
    --data_dir %DATA_DIR% ^
    --target %TARGET% ^