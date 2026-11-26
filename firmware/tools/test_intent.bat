@echo off
REM Test TextCNN + logic.c (code C that cua firmware) tren may tinh, khong can ESP.
REM Lenh trong chuong trinh: ":det cup,book,phone" doi vat trong anh gia lap | ":empty" anh trong | ":q" thoat
cd /d "%~dp0..\.."
set PYTHONIOENCODING=utf-8
Scripts\python.exe firmware\tools\intent_host.py %*
pause
