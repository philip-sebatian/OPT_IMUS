#!/usr/bin/env python3
"""
Setup script for the Optimus routing system.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Optimus: Advanced Vehicle Routing with Refill Optimization"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="optimus-routing",
    version="1.0.0",
    author="Optimus Team",
    author_email="optimus@example.com",
    description="Advanced Vehicle Routing with Refill Optimization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/optimus",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "viz": [
            "matplotlib>=3.3",
            "networkx>=2.5",
            "plotly>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "optimus-demo=optimus.examples.basic_usage:main",
            "optimus-split=optimus.examples.split_delivery_demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="routing optimization vehicle logistics cuopt nvidia",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/optimus/issues",
        "Source": "https://github.com/your-repo/optimus",
        "Documentation": "https://optimus.readthedocs.io/",
    },
)
