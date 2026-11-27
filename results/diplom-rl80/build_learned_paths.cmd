@echo off
setlocal
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0"
cl /nologo /O2 /EHsc /std:c++17 /LD learned_paths.cpp /Fe:learned_paths.dll
if errorlevel 1 exit /b 1
cl /nologo /O2 /EHsc /std:c++17 /DPATH_TRAIN /LD learned_paths.cpp /Fo:path_counts.obj /Fe:path_counts.dll
set build_result=%errorlevel%
popd
exit /b %build_result%
