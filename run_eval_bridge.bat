@echo off
REM ============================================================
REM Step 2: Evaluate ViT_M2F2Det_Bridge on ALL benchmark datasets
REM ============================================================
cd /d "E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy"

set PYTHON=C:\Users\Supor2\.conda\envs\M2F2_Det\python.exe
set CHECKPOINT=./checkpoints/stage_1/bridge_phase1.pth
set DATA_ROOT=./dataset

echo ============================================================
echo Evaluating ViT_M2F2Det_Bridge on ALL datasets
echo ============================================================
echo Checkpoint:  %CHECKPOINT%
echo Data root:   %DATA_ROOT%
echo ============================================================

%PYTHON% vit_module/eval_all_datasets.py ^
    --model-type bridge ^
    --checkpoint "%CHECKPOINT%" ^
    --data-root "%DATA_ROOT%" ^
    --batch-size 64

echo.
echo Evaluation complete!
pause
