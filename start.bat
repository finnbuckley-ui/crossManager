@echo off
echo Starting TikTok Clip Factory...
cd /d %~dp0
if not exist ".venv\Scripts\python.exe" (
	echo Creating Python 3.11 virtual environment...
	py -3.11 -m venv .venv
)

cd backend
start cmd /k "..\.venv\Scripts\python.exe -m pip install -r requirements.txt && ..\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000"
cd ..\frontend
start cmd /k "npm install && npm run dev"
timeout /t 4
start http://localhost:5173
