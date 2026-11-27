@echo off
setlocal
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0"
cl /nologo /O2 /EHsc /std:c++17 /LD region.cpp /Fe:region.dll
if errorlevel 1 exit /b 1
cl /nologo /O2 /EHsc /std:c++17 /LD synthesis.cpp /Fe:synthesis.dll
set compile_status=%errorlevel%
popd
exit /b %compile_status%
