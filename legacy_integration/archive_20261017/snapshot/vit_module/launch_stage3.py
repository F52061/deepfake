"""真正独立的训练启动器 — 用 subprocess.CREATE_NEW_CONSOLE 在新窗口启动。

用法（在任意终端中）:
    C:\Users\Supor2\.conda\envs\M2F2_Det\python.exe vit_module/launch_stage3.py

会在新窗口启动 bat，关闭本终端不影响训练。
"""
import subprocess, os, time

bat_path = r'E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy\vit_module\run_stage3.bat'
work_dir = r'E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy'

p = subprocess.Popen(
    ['cmd.exe', '/c', 'start', 'Stage3-Training', '/min', bat_path],
    cwd=work_dir,
    creationflags=subprocess.CREATE_NEW_CONSOLE,
)
print(f'Launcher PID: {p.pid}')
print('New console window opened - training will run independently.')
print('You can close this terminal safely.')
time.sleep(2)
