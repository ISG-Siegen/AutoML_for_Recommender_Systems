# ---- Base Installations ----
FROM python:3.9.7 as base

RUN apt-get update && apt-get install python-dev -y && \
	apt-get install swig -y && apt install default-jre -y && \
	pip install --no-cache-dir --upgrade pip

# Base requriements to run the benchmark framework (will get newest versions)
RUN pip install --no-cache-dir pandas && pip install --no-cache-dir scikit-learn

# ------ Library Specific Images ------

# ---- Baselines ----
# -- constant_predictors --
FROM base as constant_predictors


# ---- ML ----
# -- scikit-learn --
FROM base as sklearn
RUN pip install --no-cache-dir scikit-learn

# -- xgboost --
FROM base as xgboost
RUN pip install --no-cache-dir xgboost


# ---- AutoML ----
# -- autosklearn --
FROM base as autosklearn
RUN pip install --no-cache-dir auto-sklearn

# -- flaml --
FROM base as flaml
RUN pip install --no-cache-dir flaml

# -- gama --
FROM base as gama
RUN pip uninstall --yes pandas && pip uninstall --yes scikit-learn  && pip install --no-cache-dir gama

# -- h2o --
FROM base as h2o
RUN pip install --no-cache-dir h2o

# -- tpot --
FROM base as tpot
RUN pip install --no-cache-dir tpot


# ---- RecSys ----
# -- surprise --
FROM base as surprise
RUN pip install --no-cache-dir surprise


# ---- AutoRecSys ----
# -- autosurprise --
FROM base as autosurprise
RUN pip install --no-cache-dir auto-surprise







