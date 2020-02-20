@echo off
set env_name=py38_env
set project_name=prac_kmeans
set execpath=%cd%

call conda.bat activate %env_name%

:: Move script path to current
cd /d %~dp0
cd ../

@echo on
python -m %project_name% conf/config.json

call conda.bat deactivate

@echo off
cd /d %execpath%
