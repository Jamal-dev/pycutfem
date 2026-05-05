from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")
REQUIREMENTS = [
    line.strip()
    for line in Path(__file__).with_name("requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="pycutfem",
    version="0.1",
    description="Research-oriented CutFEM toolkit for scientific computing workflows",
    long_description=README,
    long_description_content_type="text/markdown",
    keywords=["continuum mechanics", "scientific computing"],
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    python_requires=">=3.9",
    extras_require={
        "dev": ["pytest"],
        "pardiso": ["pypardiso>=0.4.7"],
    },
)
