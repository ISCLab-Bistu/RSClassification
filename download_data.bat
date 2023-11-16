@echo off

REM Set the download URL
set DOWNLOAD_URL=https://github.com/ISCLab-Bistu/RSClassification/releases/download/dataV1.0.0/data.zip

REM Set the name of the file to save
set FILE_NAME=data.zip

REM Download the file
powershell -Command "Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile '%FILE_NAME%'"

echo Download completed: %FILE_NAME%
pause
