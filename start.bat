@echo off
REM start.bat - Startet wgp.py mit aktivierter venv auf Port 8734

REM Pfad zur virtuellen Umgebung (anpassen, falls anders)
set "VENV_PATH=D:\claudeAgent\wan2gp\venv"

REM venv aktivieren
call "%VENV_PATH%\Scripts\activate.bat"

REM wgp.py starten auf Port 8734
python wgp.py --port 8734

REM Konsole blockiert, bis Python beendet
pause
