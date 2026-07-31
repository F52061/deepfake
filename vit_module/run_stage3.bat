@echo off
REM ============================================================
REM Stage-3: LoRA Fine-tuning (2-GPU fp16, native dispatch)
REM ============================================================
cd /d "E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy"

REM Use 2 cards (fp16 7B ~14GB over 2x11.8GB).
set CUDA_VISIBLE_DEVICES=2,3

echo ============================================================
echo Stage-3 LoRA Fine-tuning (2-GPU fp16, %CUDA_VISIBLE_DEVICES%)
echo ============================================================

C:\Users\Supor2\.conda\envs\M2F2_Det\python.exe -u vit_module/run_stage3.py

echo.
echo Stage-3 training finished!
pause
