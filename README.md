# Operation AutoRecSys
A project to evaluate whether AutoRecSys is needed.

## Idea
A comparison is done by looking at the performance of AutoML, ML, AutoRecSys, and RecSys libraries. 
The comparison is done using the predictive accuracy metric RSME on multiple datasets (i.e., offline evaluation + explicit feedback). 

* Why RSME? RSME is a popular metric for predictive accuracy when explicit feedback like ratings is given
* Why offline evaluation? Following the goal of offline evaluation, that is, filtering out inappropriate approaches and tuning for later online usage 

## Project Setup
* `benchmark_framework`: basic models / code to keep consistent interfaces
* `data_in_git`: stores data we can/want to share via git - like small result datasets but never input datasets!
* `data_processing`: contains code to preprocess and make data usable for the work
* `evaluation`: contains code related to evaluation and plotting the error
* `libraries`: contains code to build models and execute them on datasets 
* `example_config.ini`: example config file that might be needed to map paths to relevant datasets 
* `container_env_mgmt.py`: entry point to our scripts, manages container environment and executes the code
* `docker_env`: contains docker files, readme and requirements.txt to setup the environment similar to ours 

## Config File
Edit the config file to fit your system (e.g. path to datasets) and rename it from `example_config.in` to `config.ini` 

# Other Notes and Remarks
* Several AutoML Libraries creat cash files, log directory and stuff like that during their execution. If possible, this was disabled. However, sometimes it was not possible. In these cases, one needs to delete them before running the code again to avoid giving libraries an unfair advantage (i.e. a warm start). 

# Personal notes / Remove later
# General Notes
* Evaluation should be done on 10-fold cross validation 
* More plots and ideas have been already collected and can be implemented.

## Potential Test Setting Setup
* Described here more on the setup and how we compared it - also show example results 
* Ideas
  * Default parameter
  * Fine-tuned example
  * dataset with/without preprocessing allowed
  * Budget? 

## OpenML Test Suite for making data public
* Details will follow soon (maybe)

