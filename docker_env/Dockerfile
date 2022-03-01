# ---- Base Installations ----
FROM python:3.9.7 as base

RUN apt-get update && apt-get install python-dev -y && \
	apt-get install swig -y && apt install default-jre -y && \
	pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir packaging

# ---- Base Sklearn ----
FROM base as base_python
# Base requriements to run the benchmark framework (will get newest versions)
RUN pip install --no-cache-dir pandas==1.3.5 && pip install --no-cache-dir scikit-learn==1.0.1

# ---- Mgmt Container ----
FROM base as mgmt
COPY mgmt_requirements.txt mgmt_requirements.txt
RUN pip install --no-cache-dir -r mgmt_requirements.txt


# ------ Library Specific Images ------
# ---- Baselines ----
# -- constant_predictors --
FROM base_python as constant_predictors


# ---- ML ----
# -- scikit-learn --
FROM base_python as sklearn
RUN pip install --no-cache-dir scikit-learn==1.0.1

# -- xgboost --
FROM base_python as xgboost
RUN pip install --no-cache-dir xgboost==1.5.1

# -- ktrain --
FROM base as ktrain
RUN pip install --no-cache-dir scikit-learn==0.23.2
RUN pip install --no-cache-dir ktrain==0.28.3  && pip --no-cache-dir install tensorflow

# ---- AutoML ----
# -- autosklearn --
FROM base_python as autosklearn
RUN pip install --no-cache-dir auto-sklearn==0.14.2

# -- flaml --
FROM base_python as flaml
RUN pip install --no-cache-dir flaml==0.9.1

# -- gama --
FROM base as gama
RUN pip install --no-cache-dir gama==21.0.1

# -- h2o --
FROM base_python as h2o
RUN pip install --no-cache-dir h2o==3.34.0.3 && pip install --no-cache-dir psutil

# -- tpot --
FROM base_python as tpot
RUN pip install --no-cache-dir tpot==0.11.7 && pip install --no-cache-dir torch

# -- autogluon --
FROM python:3.7  as autogluon
# Using 3.7 instead of 3.8 as 3.8 support is experimental
RUN apt-get update && apt-get install python-dev -y && \
	pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir packaging && \
    pip install --no-cache-dir -U setuptools wheel && pip install --no-cache-dir -U "mxnet<2.0.0"
RUN pip install --no-cache-dir autogluon==0.3.1

# -- autopytorch --
FROM base as autopytorch
RUN pip install --no-cache-dir autoPyTorch==0.1.1

# ---- RecSys ----
# -- surprise --
FROM base_python as surprise
RUN pip install --no-cache-dir scikit-surprise==1.1.1

# -- lenskit --
FROM base as lenskit
# Add specific extra depnendices for specific algorithms to use
RUN pip install --no-cache-dir lenskit[sklearn,hpf,implicit,tf]==0.13.1

# -- spotlight --
FROM base_python as spotlight
RUN pip install torch && pip install -e git+https://github.com/maciejkula/spotlight.git@v0.1.6#egg=spotlight

# -- AutoRec --
FROM python:3.8 as autorec
# autorec needs python <3.9 for tensorflow < 2.5
RUN pip install --no-cache-dir --upgrade pip && pip install Cython && pip install --no-cache-dir autorec==0.0.2

# ---- AutoRecSys ----
# -- autosurprise --
FROM base_python as autosurprise
RUN pip install --no-cache-dir auto-surprise==0.1.7





