@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo   Flash firmware ESP32-S3 (N16R8: flash 16MB, PSRAM Octal 8MB)
echo ============================================
rem Uu tien Python cua venv du an (thu muc goc repo, 2 cap tren); khong co thi dung python cua may
set "PY=python"
if exist "%~dp0..\..\Scripts\python.exe" set "PY=%~dp0..\..\Scripts\python.exe"
echo Python: %PY%
"%PY%" -c "import esptool" >nul 2>&1
if errorlevel 1 (
  echo Chua co esptool -> dang cai...
  "%PY%" -m pip install esptool
  if errorlevel 1 (echo Cai esptool loi. & pause & exit /b 1)
)
echo.
set /p PORT="Cong COM cua board (vd COM5): "
if "%PORT%"=="" (echo Chua nhap cong COM & pause & exit /b 1)

"%PY%" -m esptool --chip esp32s3 -p %PORT% -b 460800 --before default-reset --after hard-reset ^
  write-flash --flash-mode dio --flash-size 16MB --flash-freq 80m ^
  0x0 bootloader.bin 0x8000 partition-table.bin 0x10000 esp32s3_infer.bin
if errorlevel 1 (
  echo.
  echo Flash LOI. Thu: doi cap / cong USB, hoac giu BOOT - nhan RESET - tha BOOT roi chay lai.
  echo Neu van loi checksum, thu toc do thap: sua -b 460800 thanh -b 115200 trong file nay.
)
pause
