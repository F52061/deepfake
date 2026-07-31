@echo off
REM ============================================================
REM Stage-3: LoRA Fine-tuning (2-GPU fp16, native dispatch)
REM ============================================================
cd /d "E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy"

REM Use 3 cards GPU 1,2,3 (leave GPU 0 free).
REM Note: CUDA_VISIBLE_DEVICES=1,2,3 means torch sees logical 0,1,2
REM mapping to physical 1,2,3. dispatch_model uses logical indices.
set CUDA_VISIBLE_DEVICES=1,2,3

echo ============================================================
echo Stage-3 LoRA Fine-tuning (3-GPU fp16, %CUDA_VISIBLE_DEVICES%)
echo ============================================================

C:\Users\Supor2\.conda\envs\M2F2_Det\python.exe -u vit_module/run_stage3.py

echo.
echo Stage-3 training finished!
pause
