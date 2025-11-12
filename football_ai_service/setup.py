"""
Setup script за Football AI Prediction Service
"""

from setuptools import setup, find_packages

# Четене на requirements от requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Четене на README за описание
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="football-ai-service",
    version="1.0.0",
    author="Football AI Team",
    author_email="contact@football-ai.com",
    description="AI-powered football match predictions using ESPN data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/football-ai/football-ai-service",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "football-ai-server=api.main:app",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.md"],
    },
)
