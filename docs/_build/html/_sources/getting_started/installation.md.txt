# Installation Guide

This guide will help you install Porcupy with either CPU or GPU support.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- (Optional) CUDA-compatible GPU for GPU acceleration

## Installing Porcupy

### Basic Installation (CPU only)

```bash
pip install porcupy
```

### Installation with GPU Support

For GPU acceleration, you'll need to install the appropriate version of CuPy for your CUDA version:

```bash
# For CUDA 11.x
pip install porcupy[cuda11x]

# For CUDA 12.x
pip install porcupy[cuda12x]
```

### Development Installation

To install from source for development:

```bash
# Clone the repository
git clone https://github.com/SammanSarkar/Porcupy.git
cd Porcupy

# Install in development mode with all dependencies
pip install -e ".[dev]"

# For GPU development
pip install -e ".[gpu-dev]"
```

## Verifying Installation

You can verify your installation by running:

```python
import porcupy
print(f"Porcupy version: {porcupy.__version__}")
print(f"GPU available: {porcupy.GPU_AVAILABLE}")
```

## Troubleshooting

### Common Issues

1. **CUDA Installation Issues**
   - Ensure you have the correct CUDA toolkit installed
   - Verify your GPU is CUDA-compatible
   - Check that your CUDA version matches the CuPy version

2. **Import Errors**
   - Make sure you've activated the correct Python environment
   - Try uninstalling and reinstalling the package

3. **GPU Not Detected**
   - Check that your GPU drivers are properly installed
   - Verify that CUDA is in your system PATH

For additional help, please [open an issue](https://github.com/SammanSarkar/Porcupy/issues) on GitHub.
