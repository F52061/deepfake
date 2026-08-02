@echo off
cd /d "E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy"
set CUDA_VISIBLE_DEVICES=0,2,3
echo ============================================================
echo Hybrid Merge: HF LoRA + bridge_v2 detector
echo ============================================================
C:\Users\Supor2\.conda\envs\M2F2_Det\python.exe -u vit_module/hybrid_merge.py
echo.
pause
