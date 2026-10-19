#!/bin/bash
cd "$(dirname "$0")/.."
clear
echo "=========================================================================================="
echo "                                 WAN2GP UPDATE / UPGRADE                                  "
echo "=========================================================================================="
python3 setup.py status
echo "1. Update (git pull + install requirements)"
echo "2. Upgrade (Upgrade Torch, Triton, Sage Attention, etc.)"
echo "3. Platform Migration (Upgrade to Py 3.11/Torch 2.10)"
echo "4. Exit"
echo "------------------------------------------------------------------------------------------"
read -p "Select an option (1-4): " choice

if [ "$choice" == "1" ]; then
    python3 setup.py update
elif [ "$choice" == "2" ]; then
    python3 setup.py upgrade
elif [ "$choice" == "3" ]; then
    echo "[!] This will rebuild your environment with Python 3.11/Torch 2.10"
    python3 setup.py migrate
else
    exit 0
fi
read -p "Press Enter to exit..."