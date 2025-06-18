from setuptools import setup, find_packages

# Read README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

setup(
    name="foch-cim10-mapper",
    version="0.1.0",
    author="Foch Hospital Team",
    author_email="contact@foch.fr",
    description="A Python package for mapping French medical text to CIM-10 codes using NER and semantic similarity",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/foch-hospital/foch-cim10-mapper",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: French",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch[cuda]",
        ],
    },
    include_package_data=True,
    package_data={
        "foch_cim10_mapper": ["data/*.csv", "config/*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "foch-cim10=foch_cim10_mapper.cli:main",
        ],
    },
)