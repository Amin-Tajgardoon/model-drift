@echo off

set CODE_DIR="C:\Users\mot16\Box Sync\Projects\Thesis\code\HIDENIC_overtime_analysis"
set dir_path="C:\Users\mot16\Box Sync\Projects\Thesis\output\HIDENIC_overtime_analysis\hospital_overtime_fselect"

set exp_type="error_rate_change_detection"

python -u %CODE_DIR%/drift_detection.py ^
    --exp_type %exp_type% ^
    --
    --dir_path %dir_path% ^