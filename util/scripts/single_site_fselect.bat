@echo off

set CODE_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\code\HIDENIC_overtime_analysis"

set DATA_DIR=E:\Data\HIDENIC_EXTRACT_OUTPUT_DIR\POP_SIZE_0\ITEMID_REP
set OUTPUT_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\output\HIDENIC_overtime_analysis\single_site_fselect"
mkdir %OUTPUT_DIR%

set LOAD_FILTERED_DATA=1
set TRAIN_TYPES=single_site
set SITE_NAMES=MICU UPMCPUH UPMCSHY CTICU
set MODEL_TYPES=nb rf lr rbf-svm
set TRAIN_YEARS=2008 2009 2010
set SEED=0
set TARGETS=mort_icu los_3
set REPS=raw pca
set save_data=1
set num_features=100 200 500 1000
set n_threads=4

FOR %%t IN (%SITE_NAMES%) DO (
    FOR %%r IN (%REPS%) DO (
        python -u %CODE_DIR%/experiments.py --data_dir %DATA_DIR% --output_dir %OUTPUT_DIR% --representation %%r --train_types %TRAIN_TYPES% --site_name %%t --model_types %MODEL_TYPES% --random_seed %SEED% --target_list %TARGETS% --load_filtered_data %LOAD_FILTERED_DATA% --save_data %save_data% --train_years %TRAIN_YEARS% --feature_selection 1 --K %num_features% --n_threads %n_threads%
    )
)