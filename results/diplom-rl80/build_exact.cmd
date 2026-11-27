@echo off
setlocal
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0"
cl /nologo /O2 /EHsc /std:c++17 /LD exact_mission.cpp /Fe:exact_mission.dll
if errorlevel 1 exit /b 1
cl /nologo /O2 /EHsc /std:c++17 /DMISSION_TRAINING_CHECK /LD exact_mission.cpp /Fo:mission_training_check.obj /Fe:mission_training_check.dll
set build_result=%errorlevel%
popd
exit /b %build_result%
