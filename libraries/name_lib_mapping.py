# Baselines
from libraries.Baselines.constant_predictor import load_constant_predictors_and_all_models

# RecSys
from libraries.RecSys.surprise_handler import load_surprise_and_all_models
from libraries.RecSys.lenskit_handler import load_lenskit_and_all_models
from libraries.RecSys.spotlight_handler import load_spotlight
from libraries.RecSys.autorec_handler import load_autorec_and_all_models

# AutoRecSys
from libraries.AutoRecSys.autosurprise_handler import load_auto_surprise_and_all_models

# ML
from libraries.ML.scikit_models import load_sklearn_and_all_models
from libraries.ML.xgboost_model import load_xgboost_and_all_models
from libraries.ML.ktrain_handler import load_ktrain_and_all_models

# AutoML
from libraries.AutoML.autosklearn_handler import load_auto_sklearn_and_all_models
from libraries.AutoML.flaml_handler import load_flaml_and_all_models
from libraries.AutoML.gama_handler import load_gama_and_all_models
from libraries.AutoML.H2O_handler import load_h2o_and_all_models
from libraries.AutoML.tpot_handler import load_tpot_and_all_models
from libraries.AutoML.autogluon_handler import load_autogluon_and_all_models
from libraries.AutoML.autopytorch_handler import load_auto_pytorch_and_all_models

NAME_LIB_MAP = {
    # Baselines
    "constant_predictors": load_constant_predictors_and_all_models,

    # RecSys
    "surprise": load_surprise_and_all_models,
    "lenskit": load_lenskit_and_all_models,
    "spotlight": load_spotlight,
    "autorec": load_autorec_and_all_models,

    # AutoRecSys
    "autosurprise": load_auto_surprise_and_all_models,

    # ML
    "sklearn": load_sklearn_and_all_models,
    "xgboost": load_xgboost_and_all_models,
    "ktrain": load_ktrain_and_all_models,

    # AutoML
    "autosklearn": load_auto_sklearn_and_all_models,
    "autopytorch": load_auto_pytorch_and_all_models,
    "flaml": load_flaml_and_all_models,
    "h2o": load_h2o_and_all_models,
    "autogluon": load_autogluon_and_all_models,
    "gama": load_gama_and_all_models,
    "tpot": load_tpot_and_all_models,

}


def get_all_lib_names():
    return list(NAME_LIB_MAP.keys())
