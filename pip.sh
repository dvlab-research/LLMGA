sudo apt-get update
sudo apt install tmux
sudo apt install libgl1-mesa-glx
pip install --upgrade pip  # enable PEP 660 support
pip install -e ".[train]"
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
