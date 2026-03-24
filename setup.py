from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="pycutfem",
    version="0.1",
    description="Research-oriented CutFEM toolkit for scientific computing workflows",
    long_description=README,
    long_description_content_type="text/markdown",
    keywords=["continuum mechanics", "scientific computing"],
    packages=find_packages(),
    extras_require={
        "pardiso": ["pypardiso>=0.4.7"],
    },
)
