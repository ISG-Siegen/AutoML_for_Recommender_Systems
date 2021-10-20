# Opertaion AutoRecSys
A project to evaluate whether AutoRecSys is needed

## Idea
A comparison is done by looking at the performance of AutoML, ML, AutoRecSys, and RecSys libraries. 
The comparison is done using the predictive accuracy metric RSME on multiple datasets (i.e., offline evaluation). #

* Why RSME? RSME is a popular metric for predictive accuracy 
* Why offline evaluation? Following the goal of offline evaluation, that is, filtering out inappropriate approaches and tuning for later online usage 

# Project Setup
* `data_in_git`: stores data we can/want to share via git - like small result datasets but never input datasets!
* `data_processing`: contains code to preprocess and make data usable for the work
* `evaluation`: contains code related to evaluation and plotting the error
* `libraries`: contains code to build models and execute them on datasets 
* `example_config.ini`: example config file that might be needed to map paths to relevant datasets 
* `main.py`: potential main entry point that can be used to start different parts of the projects

## Config File
Edit the config file to fit your system (e.g. path to datasets) and rename it from `example_config.in` to `config.ini` 
  
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


# Outgoing
## Docker Setup
The environment is delivered through docker (as we are using too many libraries, and it is much easier)
* Details will follow soon 
* Windows Bug: if you can not bind the volume as the path keeps chaining, use a path outside the `/Users/` directory 

## OpenML Test Suite for making data public
* Details will follow soon (maybe)