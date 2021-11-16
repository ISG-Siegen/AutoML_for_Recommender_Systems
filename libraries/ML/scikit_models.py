from benchmark_framework.model_base import Model
from abc import abstractmethod


def load_sklearn_and_all_models():
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, \
        GradientBoostingRegressor  # , HistGradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
    from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lars, Lasso, LassoLars, \
        OrthogonalMatchingPursuit, ARDRegression, BayesianRidge, HuberRegressor, TheilSenRegressor, PoissonRegressor, \
        GammaRegressor, TweedieRegressor, RANSACRegressor, SGDRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.dummy import DummyRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.tree import DecisionTreeRegressor

    # Default interface for sklearn
    class ScikitModel(Model):

        @abstractmethod
        def __init__(self, name, model):
            super().__init__("SciKit_" + name, model, "ML")

        def train(self, dataset):
            x_train, y_train = dataset.train_data
            self.model_object.fit(x_train, y_train.values.ravel())

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            return self.model_object.predict(x_test)

    # Different Models from the library
    # ------- Linear
    class ScikitLinearRegression(ScikitModel):

        def __init__(self):
            super().__init__("LinearRegressor", LinearRegression())

    class ScikitRidge(ScikitModel):

        def __init__(self):
            super().__init__("Ridge", Ridge())

    class ScikitSGD(ScikitModel):

        def __init__(self):
            super().__init__("StochasticGradientDescentRegressor", SGDRegressor())

    class ScikitElasticNet(ScikitModel):

        def __init__(self):
            super().__init__("ElasticNet", ElasticNet())

    class ScikitLassoLars(ScikitModel):

        def __init__(self):
            super().__init__("LassoLars", LassoLars())

    class ScikitOrthogonalMatchingPursuit(ScikitModel):

        def __init__(self):
            super().__init__("OrthogonalMatchingPursuit", OrthogonalMatchingPursuit())

    class ScikitARDRegression(ScikitModel):

        def __init__(self):
            super().__init__("ARDRegression", ARDRegression())

    class ScikitBayesianRidge(ScikitModel):

        def __init__(self):
            super().__init__("BayesianRidge", BayesianRidge())

    class ScikitLars(ScikitModel):

        def __init__(self):
            super().__init__("Lars", Lars())

    class ScikitLasso(ScikitModel):

        def __init__(self):
            super().__init__("Lasso", Lasso())

    class ScikitHuberRegressor(ScikitModel):

        def __init__(self):
            super().__init__("HuberRegressor", HuberRegressor())

    class ScikitTheilSenRegressor(ScikitModel):

        def __init__(self):
            super().__init__("TheilSenRegressor", TheilSenRegressor())

    class ScikitPoissonRegressor(ScikitModel):

        def __init__(self):
            super().__init__("PoissonRegressor", PoissonRegressor())

    class ScikitGammaRegressor(ScikitModel):

        def __init__(self):
            super().__init__("GammaRegressor", GammaRegressor())

    class ScikitTweedieRegressor(ScikitModel):

        def __init__(self):
            super().__init__("TweedieRegressor", TweedieRegressor())

    class ScikitRANSACRegressor(ScikitModel):

        def __init__(self):
            super().__init__("RANSACRegressor", RANSACRegressor())

    # ------- Ensemble
    class ScikitRF(ScikitModel):

        def __init__(self):
            super().__init__("RandomForestRegressor", RandomForestRegressor())

    class ScikitAdaBoostRegressor(ScikitModel):

        def __init__(self):
            super().__init__("AdaBoostRegressor", AdaBoostRegressor())

    class ScikitBaggingRegressor(ScikitModel):

        def __init__(self):
            super().__init__("BaggingRegressor", BaggingRegressor())

    class ScikitExtraTreesRegressor(ScikitModel):

        def __init__(self):
            super().__init__("ExtraTreesRegressor", ExtraTreesRegressor())

    class ScikitGradientBoostingRegressor(ScikitModel):

        def __init__(self):
            super().__init__("GradientBoostingRegressor", GradientBoostingRegressor())

    class ScikitHistGradientBoostingRegressor(ScikitModel):

        def __init__(self):
            super().__init__("HistGradientBoostingRegressor", HistGradientBoostingRegressor())

    # ------- Other
    class ScikitSVRegressor(ScikitModel):

        def __init__(self):
            super().__init__("SupportVectorRegression", SVR())

    class ScikitKNN(ScikitModel):

        def __init__(self):
            super().__init__("KNeighborsRegressor", KNeighborsRegressor())

    class ScikitRadiusNN(ScikitModel):

        def __init__(self):
            super().__init__("RadiusNeighborsRegressor", RadiusNeighborsRegressor())

    class ScikitKernelRidge(ScikitModel):

        def __init__(self):
            super().__init__("KernelRidgeRegressor", KernelRidge())

    class ScikitDummyRegressor(ScikitModel):

        def __init__(self):
            super().__init__("DummyRegressor", DummyRegressor())

    class ScikitMLPRegressor(ScikitModel):

        def __init__(self):
            super().__init__("MLPRegressor", MLPRegressor())

    class ScikitGaussianProcessRegressor(ScikitModel):

        def __init__(self):
            super().__init__("GaussianProcessRegressor", GaussianProcessRegressor())

    class ScikitDecisionTreeRegressor(ScikitModel):

        def __init__(self):
            super().__init__("DecisionTreeRegressor", DecisionTreeRegressor())

    # ------- Utils
    def load_all_scikit_models():
        linear_models = [ScikitLinearRegression, ScikitRidge, ScikitElasticNet, ScikitLassoLars, ScikitLars,
                         ScikitLasso,
                         ScikitOrthogonalMatchingPursuit, ScikitSGD, ScikitARDRegression, ScikitBayesianRidge,
                         ScikitHuberRegressor, ScikitTheilSenRegressor, ScikitGammaRegressor, ScikitPoissonRegressor,
                         ScikitTweedieRegressor, ScikitRANSACRegressor]

        ensembles = [ScikitRF, ScikitAdaBoostRegressor, ScikitBaggingRegressor, ScikitExtraTreesRegressor,
                     ScikitGradientBoostingRegressor]

        rest = [ScikitKNN, ScikitRadiusNN, ScikitSVRegressor, ScikitDummyRegressor, ScikitMLPRegressor]

        # Informal Docu: Not usable Models
        #   1. ScikitKernelRidge, ScikitGaussianProcessRegressor
        #       -> Blows up RAM with a huge matrix that is impossible even for small datasets
        #   2. ScikitHistGradientBoostingRegressor
        #       -> Could not import for some reason
        #   3. VotingRegressor and StackingRegressor
        #       -> Both require estimators as input
        #   4. ScikitDecisionTreeRegressor
        #       -> already use more sophisticated tree regression

        return rest + ensembles + linear_models

    return load_all_scikit_models()
