@echo off

set CODE_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\code\HIDENIC_overtime_analysis"

set DATA_DIR=E:\Data\HIDENIC_EXTRACT_OUTPUT_DIR\POP_SIZE_0\ITEMID_REP
set OUTPUT_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\output\HIDENIC_overtime_analysis\single_site"
mkdir %OUTPUT_DIR%

set LOAD_FILTERED_DATA=1
set TRAIN_TYPES=single_site
set SITE_NAMES=UPMCPUH UPMCSHY CTICU MICU
set MODEL_TYPES=rf lr nb rbf-svm
set SEED=0
set TARGETS=mort_icu los_3
set REPS=raw pca

    FOR %%t IN (%SITE_NAMES%) DO (
        FOR %%r IN (%REPS%) DO (
            python -u %CODE_DIR%/experiments.py --data_dir %DATA_DIR% --output_dir %OUTPUT_DIR% --representation %%r --train_types %TRAIN_TYPES% --site_name %%t --model_types %MODEL_TYPES% --random_seed %SEED% --target_list %TARGETS% --load_filtered_data %LOAD_FILTERED_DATA% --n_threads 6
    )
)