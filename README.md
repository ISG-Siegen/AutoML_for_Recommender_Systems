# The Potential of AutoML for Recommender Systems

TLDR

## Abstract of Related Paper

Automated Machine Learning (AutoML) has successfully made applications in the field of Machine Learning (ML), like image
analysis or machine translation, more accessible. Recommender Systems (RecSys) can be seen as an application of ML, yet
AutoML has found little attention in the RecSys community; nor has RecSys found notable attention in the AutoML
community. Only basic Automated Recommender Systems (AutoRecSys) libraries exist to make RecSys more accessible. The few
existing AutoRecSys libraries are based on student projects, and do not offer the features and thorough development of
current AutoML libraries. We set out to determine how AutoML libraries perform in the scenario of an inexperienced user
who wants to implement a recommender system. We compared the predictive performance of AutoML, ML, RecSys, and
AutoRecSys algorithms by evaluating $60$ algorithms, including a simple baseline, on $14$ explicit feedback RecSys
datasets. To simulate the perspective of an inexperienced user, the algorithms were evaluated with default
hyperparameters. We found that AutoML and AutoRecSys perform almost equally well. AutoML libraries performed best for
six of the $14$ datasets ($43\%$). However, there was not ‘the one’ AutoML library. The single-best library was the
AutoRecSys library Auto-Surprise, which performed best on five datasets ($36\%$). On three datasets ($21\%$), AutoML
libraries performed poorly, and standard RecSys libraries with default parameters performed best. Though, while
obtaining $50\%$ of all placements in the top five per dataset, RecSys algorithms fall behind AutoML on average. ML
algorithms are generally the worst alternative. Surprisingly, the evaluated algorithms perform worse than the baseline
in $26.90\%$ of all evaluations.

## Project Structure

* `benchmark_framework`: basic models / code to keep consistent interfaces
* `data_in_git`: stores data we can/want to share via git - like small result datasets but never input datasets!
* `data_processing`: contains code to preprocess and make data usable for the work
* `evaluation`: contains code related to evaluation and plotting the error
* `libraries`: contains code to build models and execute them on datasets
* `example_config.ini`: example config file that might be needed to map paths to relevant datasets
* `run_comparison.py`: entry point to our scripts, manages container environment and executes the code
* `run_preprocessing.py`: ....
* `docker_env`: contains docker files, readme and requirements.txt to setup the environment similar to ours
*

## How to run

### Data

#### Download Data

* Add links here (what you want to test)
* how to setup directories and folder strucutre
* link to it in config file

* warn about requirements for preprocessing
* by default all preprocssed

### Config File

Edit the config file to fit your system (e.g. path to datasets) and rename it from `example_config.in` to `config.ini`

* how to change config file

### Environment

* how to setup environemtn and point to docker env readme for details

### Running

* how to run preprocessing, comparsion and evaluation

#### Other Notes and Remarks

* Several AutoML Libraries creat cash files, log directory and stuff like that during their execution. If possible, this
  was disabled. However, sometimes this was not possible.
