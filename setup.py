from setuptools import setup, find_packages
import os

# Read the contents of README.md for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
install_requires = [
    'numpy>=1.19.0',
    'scipy>=1.5.0',
    'matplotlib>=3.3.0',
]

extras_require = {
    'gpu': ['cupy-cuda11x>=9.0.0'],  # For GPU acceleration
    'docs': [
        'sphinx>=4.0',
        'sphinx-rtd-theme',
        'myst-parser',
    ],
    'test': [
        'pytest>=6.0.0',
        'pytest-cov>=2.0.0',
        'pytest-xdist>=2.0.0',
    ],
    'dev': [
        'ipython>=7.0.0',
        'jupyter>=1.0.0',
        'black>=21.0',
        'flake8>=3.9.0',
        'mypy>=0.900',
    ],
}

# All test and development dependencies
extras_require['all'] = (
    extras_require['gpu']
    + extras_require['docs']
    + extras_require['test']
    + extras_require['dev']
)

setup(
    name="porcupy",
    version="0.2.0",
    author="Samman Sarkar",
    author_email="sarkar.samman4231@gmail.com", 
    description="A Python library for the Crested Porcupine Optimizer for research and optimization tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SammanSarkar/Porcupy",
    packages=find_packages(exclude=["tests*"]),
    package_data={
        "porcupy": ["py.typed"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require=extras_require,
    keywords=[
        'optimization',
        'metaheuristics',
        'evolutionary algorithms',
        'swarm intelligence',
        'machine learning',
        'artificial intelligence',
    ],
    project_urls={
        'Documentation': 'https://porcupy-cpo.readthedocs.io/',
        'Source': 'https://github.com/SammanSarkar/Porcupy',
        'Bug Reports': 'https://github.com/SammanSarkar/Porcupy/issues',
    },
    license="MIT",
)
