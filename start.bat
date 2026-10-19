@echo off
REM start.bat - Startet wgp.py mit aktivierter venv auf Port 8734

set "WGP_ROOT=D:\claudeAgent\wan2gp"
set "VENV_PATH=%WGP_ROOT%\venv"

REM ins Projektverzeichnis wechseln (sonst findet wgp.py models.txt etc. nicht)
cd /d "%WGP_ROOT%"

REM venv aktivieren
call "%VENV_PATH%\Scripts\activate.bat"

REM wgp.py starten: --server-port ist der korrekte Parameter (--port gibt es nicht)
python wgp.py --server-port 8734 --listen

REM Konsole blockiert, bis Python beendet
pause
