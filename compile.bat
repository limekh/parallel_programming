@echo off
echo ========================================
echo   Matrix Multiplication - Compile (CUDA)
echo ========================================
echo.

REM Проверка наличия nvcc
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found!
    echo Please install CUDA Toolkit from:
    echo https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

echo Compiling with NVCC...
nvcc -O3 -o matrix_cuda.exe src/matrix_cuda.cu

if %errorlevel% equ 0 (
    echo.
    echo [V] Compilation successful!
    echo    File: matrix_cuda.exe
    echo.
    echo To run: matrix_cuda.exe run input.txt output.txt 16 0
) else (
    echo.
    echo [X] Compilation failed!
    echo    Make sure CUDA Toolkit is properly installed
)
echo.
pause