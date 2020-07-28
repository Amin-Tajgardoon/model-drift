@echo off

set CODE_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\code\HIDENIC_overtime_analysis"
set dir_path="C:\Users\mot16\Box Sync\Projects\Thesis\output\HIDENIC_overtime_analysis\hospital_overtime_fselect"

set run_bootstrap=1
set n_bootstrap=100
set generate_stats=1
set stat_test=mannwhitneyu

python -u %CODE_DIR%/bootstrap_predictions.py ^
    --dir_path %dir_path% ^
    --run_bootstrap %run_bootstrap% ^
    --n_bootstrap %n_bootstrap% ^
    --generate_stats %generate_stats% ^
    --stat_test %stat_test% ^