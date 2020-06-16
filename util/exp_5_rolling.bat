@echo off

set CODE_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\code\HIDENIC_overtime_analysis"

set DATA_DIR=E:\Data\HIDENIC_EXTRACT_OUTPUT_DIR\POP_SIZE_0\ITEMID_REP
set TRAIN_TYPE=rolling
set MODEL_TYPES=rf lr
set /a SEED=0
set TARGETS=los_3 mort_icu

for %%t in (%TARGETS%) do (
    python -u %CODE_DIR%/experiments.py ^
    --representation raw ^
    --train_type %TRAIN_TYPE% ^
    --model_types %MODEL_TYPES% ^
    --random_seed %SEED% ^
    --target %%t ^
)