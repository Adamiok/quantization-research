# Quantization Research

Aim: To compare PTQ and QAT on a convolutional neural network with a small sample size

Part of my project on quantization research.

## Get Started

Trained models are already provided, alternatively you can train them yourself.

## Linux/MacOS
```sh
git clone https://github.com/Adamiok/quantization-research.git
cd quantization-research
python -m venv .venv && . .venv/bin/activate
pip install --upgrade pip

# REQUIRED: install pytorch according to: https://pytorch.org/get-started/locally/

pip install -r requirements.txt fbgemm-gpu-genai

# Train models (optional)
rm -rf models && python train.py

# Test models
python test.py

# Visualise results (requires an interactive environment)
# Use 'results.json' as filename, unless manually renamed
python parse.py
```

## Windows

```bat
git clone https://github.com/Adamiok/quantization-research.git
cd quantization-research
python -m venv .venv && .venv\Scripts\activate
python -m pip install --upgrade pip

# REQUIRED: install pytorch according to: https://pytorch.org/get-started/locally/

pip install -r requirements.txt

# Train models (optional)
rd /S /Q models && python train.py

# Test models
python test.py

# Visualise results (requires an interactive environment)
# Use 'results.json' as filename, unless manually renamed
python parse.py
```