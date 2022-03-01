# The Potential of AutoML for Recommender Systems
A Comparison of AutoML, ML, AutoRecSys and RecSys algorithms on RecSys datasets to determine the potential of AutoML.

## Abstract of Related Paper
Automated Machine Learning (AutoML) has successfully made applications in the field of Machine Learning (ML), like image
analysis or machine translation, more accessible. Recommender Systems (RecSys) can be seen as an application of ML, yet
AutoML has found little attention in the RecSys community; nor has RecSys found notable attention in the AutoML
community. Only basic Automated Recommender Systems (AutoRecSys) libraries exist to make RecSys more accessible. The few
existing AutoRecSys libraries are based on student projects, and do not offer the features and thorough development of
current AutoML libraries. We set out to determine how AutoML libraries perform in the scenario of an inexperienced user
who wants to implement a recommender system. We compared the predictive performance of AutoML, ML, RecSys, and
AutoRecSys algorithms by evaluating $60$ algorithms, including a simple baseline, on 14 explicit feedback RecSys
datasets. To simulate the perspective of an inexperienced user, the algorithms were evaluated with default
hyperparameters. We found that AutoML and AutoRecSys perform almost equally well. AutoML libraries performed best for
six of the 14 datasets (43%). However, there was not ‘the one’ AutoML library. The single-best library was the
AutoRecSys library Auto-Surprise, which performed best on five datasets (36%). On three datasets (21%), AutoML libraries
performed poorly, and standard RecSys libraries with default parameters performed best. Though, while obtaining 50% of
all placements in the top five per dataset, RecSys algorithms fall behind AutoML on average. ML algorithms are generally
the worst alternative. Surprisingly, the evaluated algorithms perform worse than the baseline in 26.90% of all
evaluations.

## Project Structure

* `benchmark_framework`: basic models / code to keep consistent interfaces.
* `data_in_git`: stores data we can/want to share via git - like small result datasets but not input datasets.
* `data_processing`: contains code to preprocess and make data usable for the comparison.
* `docker_env`: contains docker files, readme, and requirements.txt to setup the environment similar to ours (for short
  version, see below).
* `evaluation`: contains code related to evaluation and plotting used in the related paper.
* `libraries`: contains code to build models of libraries and execute them on datasets.
* `general_utils`: contains utility code for different applications
* `example_config.ini`: example config file that needs to be changed (see below).
* `run_comparison.py`: entry point to our scripts, manages container environment and executes the code.
* `run_preprocessing.py`: script to run preprocessing.

# Project Setup

## RecSys Datasets Required for the Comparison

### Download

You first need to download the original datasets you want to use for the evaluation.
See `data_processin/preprocessing/README.md` for information about supported datasets and where to you can download
them.

The code requires the following folder structure to use the original datasets:

1. Set up a directory X which shall contain all datasets and the preprocessed data
2. Create a folder called `preprocessed_data` in this directory. This will be the output directory for preprocessing
3. Assuming you have downloaded the original datasets, they need to be inserted in directory X using the following
   approach:
    * **General**: make sure that the relevant data files are in the directory of the dataset. The data file should not
      be in their own directory within the dataset directory.
    * `Food.com`: Re-named the downloaded archive folder to `food_com_archive`.
        * Should contain at least "RAW_recipes.csv" and "RAW_interactions.csv".
    * `netflix`: Re-named the downloaded archive folder to `neftlix`.
        * Should contain at least "combined_data_1", "combined_data_2", "combined_data_3", "combined_data_4".
    * `yelp`: extract the downloaded datasets into a folder called `yelp`.
        * Should contain at least "yelp_academic_dataset_business", "yelp_academic_dataset_review", and
          "yelp_academic_dataset_user".
    * `Amazon`: Do not extract the downloaded files and meta-files but simply move the ".json.gz" files into a new
      directory called `amazon`.The different category files are assumed to have the following names and end with "
      .json.gz".
        * `amazon-electronics`: "Electronics_5", "meta_Electronics"
        * `amazon-movies-and-tv`: "Movies_and_TV_5", "meta_Movies_and_TV"
        * `amazon-digital-music`: "Digital_Music_5", "meta_Digital_Music"
        * `amazon-toys-and-games`: "Toys_and_Games_5", "meta_Toys_and_Games"
        * `amazon-fashion`: "AMAZON_FASHION_5", "meta_AMAZON_FASHION"
        * `amazon-appliances`: "Appliances_5", "meta_Appliances"
        * `amazon-industrial-and-scientific`: "Industrial_and_Scientific_5", "meta_Industrial_and_Scientific"
        * `amazon-software`: "Software_5", "meta_Software"
    * `Movielens-latest-small`: Extract and re-name the downloaded file to `ml-latest-small`.
        * Should contain at least "movies.csv" and "ratings.csv".
    * `Movielens-1m`: Extract and re-name the downloaded file to `ml-1m`.
        * Should contain at least "movies.dat", "users.dat", and "ratings.dat".
    * `Movielens-100k`: Extract and re-name the downloaded file to `ml-100k`.
        * Should contain at least "u.data", "u.item", and "u.user".

### Preprocessing

_Warning_: The preprocessing of larger datasets like yelp or amazon-electronics requires a lot of memory. If your system
does not have enough memory, use smaller datasets to test this comparison. For a smaller test, we
recommend: `Movielens-100k`, `amazon-software`, `movielens-latest-small`.

Once the files have been downloaded and stored in a directory as describe above, we can preprocess them to be usable in
our comparison. To do so:

1. re-name the `example_config.ini` to `config.ini`
2. In the `config.ini`, change the value for `local_path` to the (absolute) path to the directory X from above, i.e.,
   the path to the directory full of dataset directories.
3. In the `config.ini`, select datasets that shall be preprocessed. In the section "to_preprocess", write "True" for a
   dataset that you have downloaded and want to preprocess and "False" otherwise.
4. Set up a management environment to preprocess the datasets (also part of the environment setup description later)
    * To do so, install the requirements of the `docker_env/mgmt_requirements.txt` in a Python environment.
    * Alternatively, set up the docker environment as described later and use the docker environment.
5. Using the management environment of the last step, run the `run_preprocessing.py` script in the project's root
   directory.
    * This will start the preprocessing for all datasets selected in the config file.
    * By default, this is executed in the local environment. If you have set up the docker environment, you can also let
      the script run in a docker container. This requires changing the parameter in the script file.

## Docker Environment Setup

The following is a short version on how to install the docker environment. Further details on the Docker Setup can be
found in `docker_env/README.md`.

1. Install Docker
    * Windows: You can use and install `Docker for Desktop`.
    * Linux: Install `docker` and `docker-compose`
2. Use Docker to build all images. We have a dockerfile, which contains the installation commands for all libraries that
   are part of this comparison. The docker-compose.yml manages these install commands and creates an image for each
   library. If you want to use only a subset of these libraries, you will need to remove the corresponding entries form
   the dockerfile and docker-compose.yml. Please be aware, that the build process may take some time.
    * By default, the dockerfile will try to install the versions we have used. If you want to use another version, you
      will have to adapt the dockerfiles accordingly. See `libraries/README.md` for more details.
    * In the `docker_env` directory use the following command (in a terminal).
        * Windows: `docker compose build`
        * Linux: `docker-compose build`
3. Once the build process finished successfully, you are done. An image for each library should have been created.

## Run the Comparison

### Config File Setup

To run the comparison, first adapt the `config.ini` file (assuming you have followed the steps above and re-named
the `example_config.ini` already). In this file, you can adapt the search and timeout limit in the section "limits". In
the section "defaults", you can select whether you want to use/catch resource limits and if only new evaluations shall
be executed.

* In the default state, no evaluation would be executed by our script, since all evaluations (for all library+datasets
  combinations) have been run and are presented in the results
  file (`data_in_git/result_data/overall_benchmark_results.csv`). Hence, if you want to test additional evaluations,
  set "only_new_evaluations" to "False". Alternatively, you can remove corresponding entries from the results file.

### Start the Script

To run the comparison, execute the `run_comparison.py` from the project's root directory in a Python environment that
has all management requirements.

0. Set up a management environment to run the comparison (skip if you have already done this for preprocessing)
    * To do so, install the requirements of the `docker_env/mgmt_requirements.txt` in a Python environment.
1. Execute  `run_comparison.py` from the project's root directory in this Python environment

### Other Notes and Remarks

* Several AutoML Libraries creat log directories and auxiliary files during their execution. If possible, this was
  disabled. However, sometimes this was not possible.

## Run the Evaluation Script

To produce the plots and tables which we have used in our paper, you can run our evaluation code.

0. Set up a management environment to run the evaluation (skip if you have already done this before)
    * To do so, install the requirements of the `docker_env/mgmt_requirements.txt` in a Python environment.
1. Move to the `evaluation` directory.
2. Run the `evaler.py` script with the Python environment from above. This will produce and potentially overwrite
   figures and tables in `data_in_git/result_tables` and `data_in_git/images`.
    * We exported and re-sized the images by hand to save them. If you let the script save the figures, they will look
      not the same.
    * The script also prints relevant information. It uses the data
      in `data_in_git/result_data/overall_benchmark_results.csv` to produce its results.
    * **Please note**, the evaluation script uses the `data_in_git/result_data/run_overhead_data.json` file to determine
      which algorithms and datasets are relevant. Specifically, it does not include algorithms or datasets in the
      evaluation if they are not in the `run_overhead_data.json` file. Libraries or Datasets are not in the file if you
      excluded a datasets or library in your latest run of `run_comparison.py` (because it
      updates `run_overhead_data.json` accordingly).
        * If you want to use all data in a results file, toggle the option for it in the evaluation script. Please be
          aware that this can cause crashes or non-representative plots if not all datasets for a library have been
          evaluated, i.e., run by `run_comparison.py`, or the result file contains faulty entries.   