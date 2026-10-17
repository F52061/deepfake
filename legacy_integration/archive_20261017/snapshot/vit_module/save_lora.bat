@echo off
cd /d "E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy"
set CUDA_VISIBLE_DEVICES=0,2,3
echo ============================================================
echo Saving LoRA weights from final checkpoint
echo ============================================================
C:\Users\Supor2\.conda\envs\M2F2_Det\python.exe -u vit_module/save_lora_from_checkpoint.py
echo.
echo Save complete!
pause
