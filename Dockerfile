FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EIGEN_INCLUDE_DIR=/usr/include/eigen3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        git \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/pycutfem

COPY requirements.txt setup.py README.md ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY pycutfem ./pycutfem
COPY examples/__init__.py ./examples/__init__.py
COPY examples/utils ./examples/utils
COPY examples/biofilms/benchmarks/stoter ./examples/biofilms/benchmarks/stoter
COPY examples/biofilms/benchmarks/three_constituent ./examples/biofilms/benchmarks/three_constituent
COPY tests ./tests

RUN python -m pip install -e .

CMD ["python", "-m", "pytest", "-q", "tests/test_three_constituent_benchmark_suite.py", "tests/test_three_constituent_one_domain_cpp.py"]
