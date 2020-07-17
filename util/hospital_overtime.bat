@echo off

set CODE_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\code\HIDENIC_overtime_analysis"

set DATA_DIR=E:\Data\HIDENIC_EXTRACT_OUTPUT_DIR\POP_SIZE_0\ITEMID_REP
set OUTPUT_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\output\HIDENIC_overtime_analysis\test\hospital_overtime"
mkdir %OUTPUT_DIR%

set LOAD_FILTERED_DATA=1
set TRAIN_TYPES=hospital_overtime
set TRAIN_HOSP=UPMCPUH
set MODEL_TYPES=nb lr rf rbf-svm
set SEED=0
set TARGETS=los_3
set representation=raw
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
    --n_threads 6 ^
    --train_hospitals %TRAIN_HOSP% ^