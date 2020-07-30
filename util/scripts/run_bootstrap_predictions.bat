@echo off

set CODE_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\code\HIDENIC_overtime_analysis"
set data_dir="C:\Users\mot16\Box Sync\Projects\Thesis\output\HIDENIC_overtime_analysis\with_month_intervals"

set source_train_type=first_years
set run_bootstrap=0
set n_bootstrap=100
set generate_stats=1
set stat_test=mannwhitneyu

python -u %CODE_DIR%/bootstrap_predictions.py ^
    --data_dir %data_dir% ^
    --source_train_type %source_train_type% ^
    --run_bootstrap %run_bootstrap% ^
    --n_bootstrap %n_bootstrap% ^
    --generate_stats %generate_stats% ^
    --stat_test %stat_test% ^