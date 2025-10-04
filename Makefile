# Makefile for social_distance_monitor (CPU-only, WSL-friendly)
# Usage:
#   make system-deps   # install required ubuntu packages (requires sudo)
#   make all           # create venv, install deps, download models
#   make run           # run the app (append ARGS="--video test.mp4" etc)
#   make models        # download model files only
#   make venv          # create virtualenv and install deps
#   make calib-skeleton# create a sample calib.json to edit
#   make clean         # remove venv
#   make dist-clean    # remove venv and downloaded models

PYTHON := python3
VENV_DIR := venv
PIP := $(VENV_DIR)/bin/pip
PY := $(VENV_DIR)/bin/$(PYTHON)
REQ := requirements.txt

PROTO := MobileNetSSD_deploy.prototxt
MODEL := MobileNetSSD_deploy.caffemodel
CALIB := calib.json

# Use mirrors that are known to work (raw gist for prototxt, sourceforge mirror for caffemodel)
PROTO_URL := https://gist.githubusercontent.com/mm-aditya/797a3e7ee041ef88cd4d9e293eaacf9f/raw/MobileNetSSD_deploy.prototxt
MODEL_URL := https://sourceforge.net/projects/ip-cameras-for-vlc/files/MobileNetSSD_deploy.caffemodel/download

# Allow the user to pass extra args to the python script: make run ARGS="--camera 0"
ARGS ?=

# Windows detection (basic)
ifeq ($(OS),Windows_NT)
    VENV_DIR := venv
    PIP := $(VENV_DIR)\Scripts\pip.exe
    PY := $(VENV_DIR)\Scripts\python.exe
endif

.PHONY: all venv install models run clean dist-clean help system-deps calib-skeleton

all: venv models

# Create virtualenv and install python deps
venv:
	@echo "==> Creating virtualenv (if missing) at $(VENV_DIR)"
	@if [ ! -d "$(VENV_DIR)" ]; then $(PYTHON) -m venv $(VENV_DIR); else echo "venv exists"; fi
	@echo "==> Upgrading pip + installing requirements"
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r $(REQ)

install: venv
	@echo "==> Installing requirements into existing venv..."
	@$(PIP) install -r $(REQ)

# Download model files (won't overwrite existing files)
models:
	@echo "==> Downloading model files (if missing)..."
	@if [ -f "$(PROTO)" ]; then echo "$(PROTO) already exists"; else curl -L -o $(PROTO) $(PROTO_URL); fi
	@if [ -f "$(MODEL)" ]; then echo "$(MODEL) already exists"; else curl -L -o $(MODEL) $(MODEL_URL); fi
	@echo "==> Model files ready: $(PROTO) $(MODEL)"

# Optional: install system deps on Ubuntu/WSL (run once with sudo)
system-deps:
	@echo "==> Installing system packages (requires sudo)."
	@echo "You may be prompted for your password."
	sudo apt update
	sudo apt install -y curl x11-xserver-utils libxcb-xinerama0 libxkbcommon-x11-0 \
		libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render0 \
		libgl1-mesa-glx libgl1-mesa-dri libglib2.0-0 || true
	@echo "==> system packages installed (if available)."

# Create a small calib.json skeleton you can edit
calib-skeleton:
	@echo "==> Creating sample calib.json (edit image_points/world_points to match your scene)"
	@printf '{\n  "image_points": [[150,420], [820,420], [820,770], [150,770]],\n  "world_points": [[0,0], [3,0], [3,2], [0,2]],\n  "H": null\n}\n' > $(CALIB)
	@echo "Saved: $(CALIB) - edit image_points/world_points and re-run with --load-calib"

# Run program using the venv python
run: all
	@echo "==> Running social_distance_monitor.py"
	@echo "If using WSLg, ensure DISPLAY is ':0' (export DISPLAY=:0) and WSL was restarted."
	@echo "To allow GUI with sudo, run 'xhost +local:root' on your Windows host (careful: security)."
	@$(PY) social_distance_monitor.py $(ARGS)

clean:
	@echo "==> Removing venv directory $(VENV_DIR)"
	@rm -rf $(VENV_DIR) || true

dist-clean: clean
	@echo "==> Removing model files $(PROTO) $(MODEL) and calib.json"
	@rm -f $(PROTO) $(MODEL) $(CALIB) || true

push:
	git add .
	git commit -m "new"
	git push origin main --force
	
help:
	@sed -n '1,200p' Makefile
	@echo ""
	@echo "Common usage:"
	@echo "  make system-deps      # install OS packages (run once, needs sudo)"
	@echo "  make all              # create venv and download models"
	@echo "  make run ARGS=\"--video test.mp4 --load-calib --show-birdeye\""
	@echo "  make calib-skeleton   # write a sample calib.json you can edit"
	@echo "  make dist-clean       # clean everything"
