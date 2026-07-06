@echo off
cd /d "%~dp0"
python core/server.py --host 0.0.0.0 --port 18789
pause
