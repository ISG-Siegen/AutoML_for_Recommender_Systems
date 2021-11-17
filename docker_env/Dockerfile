# ---- Base Installations ----
FROM python:3.9.7 as base

RUN apt-get update && apt-get install python-dev -y && \
	apt-get install swig -y && apt install default-jre -y && \
	pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir packaging

# ---- Base Sklearn ----
FROM base as base_python
# Base requriements to run the benchmark framework (will get newest versions)
RUN pip install --no-cache-dir pandas && pip install --no-cache-dir scikit-learn

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
RUN pip install --no-cache-dir scikit-learn

# -- xgboost --
FROM base_python as xgboost
RUN pip install --no-cache-dir xgboost

# -- ktrain --
FROM base as ktrain
RUN pip install --no-cache-dir scikit-learn==0.23.2
RUN pip install --no-cache-dir ktrain  && pip --no-cache-dir install tensorflow

# ---- AutoML ----
# -- autosklearn --
FROM base_python as autosklearn
RUN pip install --no-cache-dir auto-sklearn

# -- flaml --
FROM base_python as flaml
RUN pip install --no-cache-dir flaml

# -- gama --
FROM base as gama
RUN pip install --no-cache-dir gama

# -- h2o --
FROM base_python as h2o
RUN pip install --no-cache-dir h2o

# -- tpot --
FROM base_python as tpot
RUN pip install --no-cache-dir tpot

# -- tpot --
FROM python:3.7  as autogluon
# Using 3.7 instead of 3.8 as 3.8 support is experimental
RUN apt-get update && apt-get install python-dev -y && \
	pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir packaging && \
    pip install --no-cache-dir -U setuptools wheel && pip install --no-cache-dir -U "mxnet<2.0.0"
RUN pip install --no-cache-dir autogluon

# ---- RecSys ----
# -- surprise --
FROM base_python as surprise
RUN pip install --no-cache-dir surprise

# -- lenskit --
FROM base as lenskit
# Add specific extra depnendices for specific algorithms to use
RUN pip install --no-cache-dir lenskit[sklearn,hpf,implicit,tf]


# ---- AutoRecSys ----
# -- autosurprise --
FROM base_python as autosurprise
RUN pip install py4j
RUN pip install --no-cache-dir auto-surprise

## -- AutoRec --
# -- AutoRec --
FROM python:3.8 as autorec
# autorec needs python <3.9 for tensorflow < 2.5
RUN pip install --no-cache-dir --upgrade pip && pip install Cython && pip install --no-cache-dir autorec==0.0.2

