# Docker Container Setup
This implementation of our environment uses a new image (and subsequently container) for each library. 
Hence, each library has its own environment. 

This was done to avoid dependency clashes like Library A needing numba==0.53 
but library B needing numba!=0.53. While this can be fixed by loading older versions, the problem becomes uncontrollable once too many libraries are involved in a single environment. 

To manage such an environment, docker compose is used to build all required images (1 for each library and 1 base image) based on 1 dockerfile that contains the build instructions for all libraries. The management environment is rather simple in terms of dependencies and can be executed by a default Python Environment or also used as part of a container (!not implement yet!). 

See below for more details on running the containers individually or using our build process

## Build all images
Build all images (and prune old once, you can remove the prune part if you wish). Furthermore, feel free to remove libraries from the dockerfiles that you do not want to test.

`docker compose build && docker image prune -f`


## Run by Hand
Start Container + Code by entering the following command. This command is essentially what `container_env_mgmt.py` is doing for an individual library.
The command should be executed from the root directory (otherwise change the paths in the command :).

`docker run -v /path/to/code/workingdir:/opt/project:rw -v /path/to/datasets:/opt/datasets:ro --rm --entrypoint="python" --workdir="/opt/project" <image id | container_name:tag> libraries/main_benchmark.py name_of_library_to_execute`

Alternatively, one could modify main_benchmark such that the library that you want to run is hardcoded (e.g., change `lib_name = str(sys.argv[1])` to `lib_name = "sklearn"`). Then, one can avoid passing the `name_of_library_to_execute` everytime. 

## Setup individual containers as environment to use in PyCharm
The usual setup of using docker as remote interpreter in PyCharm can also be used here for each individual container
(!Do not forget to set volume for the datasets correctly!).
Thus, you could work on a single library in its own interpreter. 
You can execute main_benchmark with such an interpreter and test all relevant code (after hard-coding the library name or editing the run configuration to pass the library name). 
In other words, one can use the container's environment complete independent of our container management script.

If you only want to run / implement / test / build a single library, this approach is highly recommended.

## Get to the bash of an image / container that is mounted to an interpreter by default
Using the idea from https://stackoverflow.com/a/34023343. This allows you to get into a container's bash that has something else as its own default entrypoint. Thus, you could get to the unix environment of our containers to play around and see where / how the files are positioned.  

```
docker create -it --name new-container <image>
docker start new-container
docker exec -it new-container bash
```

# Used Libraries 

## Adding a new library
Add entry to docker file and docker-compose.yml. Use `base` if the library has pandas and sklearn as requirement, else use `base_python`.

Additionally, check using `https://pypi.org/pypi/{}/json` where `{}` is replaced by the library name under `requires_dist`, if the library would have conflicting dependencies with pandas or sklearn (i.e., if the library has both or one of them as a dependency) (See https://stackoverflow.com/a/50424967). IF that is the case, it is generally better to first uninstall both these packages and let it be re-installer automatically. The code used for handling the data by the benchmark framework should generally be backwards compatible enough such that any version of sklearn or pandas suffices to do their job. 
See for example the dockerfile entry for `gama`. 

If not, we might need to-reimplement some stuff. 

## List of Base Libraries for Management (container_env_mgmt.py)
See `mgmt_requirements.txt`.

* This could also be its own docker container
  see https://stackoverflow.com/questions/60054378/i-want-to-use-docker-sdk-inside-a-running-docker-container-and-want-to-build-an
* The sklearn requirement could be removed by replacing the code for metric calculation. For now, it can be part of the
  base requirements. Pandas is generally needed.


## List of Libraries for Tests
### ML
scikit-learn
xgboost

### AutoML
tpot
h2o
gama
flaml
autosklearn

### RecSys
surprise
lenskit

### AutoRecSys
auto_surprise

# Bugs and Other Stuff
## PyCharm Windows Docker Volumes
If you can not bind the volume as the path keeps chaining/reformatting, use a path outside the `/Users/` directory 