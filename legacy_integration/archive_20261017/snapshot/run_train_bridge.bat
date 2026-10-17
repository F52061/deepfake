@echo off
REM ============================================================
REM Step 1: Train ViT_M2F2Det_Bridge (BridgeAdapter version)
REM ============================================================
cd /d "E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy"

set PYTHON=C:\Users\Supor2\.conda\envs\M2F2_Det\python.exe
set VIT_CKPT=E:\Cross-domain_authentication_verification\PDI\results\Ama1_aps1_1\net_050.pth
set TRAIN_TXT=./dataset/data_2023/ffpp_train_split.txt
set DATA_ROOT=./dataset
set SAVE_PATH=./checkpoints/stage_1/bridge_phase1.pth

echo ============================================================
echo Training ViT_M2F2Det_Bridge
echo ============================================================
echo ViT checkpoint: %VIT_CKPT%
echo Train data:     %TRAIN_TXT%
echo Save to:        %SAVE_PATH%
echo ============================================================

%PYTHON% vit_module/train_bridge_phase1.py ^
    --vit-ckpt "%VIT_CKPT%" ^
    --train-txt "%TRAIN_TXT%" ^
    --data-root "%DATA_ROOT%" ^
    --save-path "%SAVE_PATH%" ^
    --epochs 10 ^
    --lr 1e-4 ^
    --batch-size 32

echo.
echo Training complete! Model saved to: %SAVE_PATH%
pause
