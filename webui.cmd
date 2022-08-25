@echo off
call C:\ProgramData\miniconda3\Scripts\activate.bat ldm
python "%CD%"\scripts\webui.py

:PROMPT
python scripts/webui.py