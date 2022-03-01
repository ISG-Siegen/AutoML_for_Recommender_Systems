# Adding a new library

See existing code how to re-use model / benchmark framework. Add the code also as a function. Afterwards, add the
name-function-mapping to `name_lib_mapping.py`. Lastly, add the library to the `Dockerfile`.

Make sure to return a list by your load function!

## Supported Libraries

The following details all supported libraries. The version next to it, is the version of the library we have used during
our comparison. By default, the environment will try to install the versions we have used. If you want to use another
version, you will have to adapt the dockerfiles accordingly.

* Surprise, \rurl{github.com/NicolasHug/Surprise}, 1.1.1, BSD 3-Clause;
* LensKit, \rurl{lenskit.org}, 0.13.1, Custom;
* Spotlight, \rurl{github.com/maciejkula/spotlight}, 0.1.6, MIT;
* AutoRec, \rurl{github.com/datamllab/AutoRec}, 0.0.2, None;
* Auto-Surprise, \rurl{github.com/BeelGroup/Auto-Surprise}, 0.1.7, MIT;
* scikit-learn, \rurl{scikit-learn.org}, 1.0.1, BSD 3-Clause;
* XGBoost, \rurl{github.com/dmlc/xgboost}, 1.5.1, Apache-2.0;
* ktrain, \rurl{github.com/amaiya/ktrain}, 0.28.3, Apache-2.0;
* Auto-sklearn, \rurl{automl.github.io/auto-sklearn}, 0.14.2, BSD 3-Clause;
* FLAML, \rurl{github.com/microsoft/FLAML}, 0.9.1, MIT;
* GAMA, \rurl{github.com/openml-labs/gama}, 21.0.1, Apache-2.0;
* H2O, \rurl{h2o.ai/products/h2o-automl}, 3.34.0.3, Apache-2.0;
* TPOT, \rurl{github.com/EpistasisLab/tpot}, 0.11.7, LGPL-3.0;
* AutoGluon, \rurl{auto.gluon.ai/stable/index.html}, 0.3.1, Apache-2.0;
* Auto-PyTorch, \rurl{github.com/automl/Auto-PyTorch}, 0.1.1, Apache-2.0;

### Changing the Version of a Library

The following details how to change the installed version of a library in our dockerfiles. The example will be for
LensKit.

1. Open `docker_env/Dockerfile`
2. Go to the entry for `LensKit`. Find this either by the comment or "FROM base as lenskit" tag.
3. Remove or replace the `==0.13.1` from the `RUN pip install` command.
    * If you remove it, the newest version will be installed by default. This, however, does not guarantee that our code
      will work with it. 