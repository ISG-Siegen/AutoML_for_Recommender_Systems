import docker
from utils.lcer import get_logger, get_dataset_local_path, get_base_path, get_dataset_container_path
import signal
import time
from libraries.name_lib_mapping import get_all_lib_names
import logging

# Remove urllib debug messages
logging.getLogger("urllib3").setLevel(logging.WARNING)

# -- Init Setup --
logger = get_logger("EnvManager")
docker_client = docker.from_env()
CONTAINER_PREFIX = "docker_env_"

# Variables for docker
WORKDIR_PATH = "/opt/project"
volumes_dir = {
    get_dataset_local_path(): {"bind": get_dataset_container_path(), "mode": "ro"},
    get_base_path(): {"bind": WORKDIR_PATH, "mode": "rw"}
}


# ---- Code ----
def start_code_in_container_handler(library_to_run, script_to_run="libraries/main_benchmark.py"):
    # Some required variables
    container_name = CONTAINER_PREFIX + library_to_run
    container_cmd = [script_to_run, library_to_run]

    logger.info("Start Container: {} for library {}".format(container_name, library_to_run))
    container = docker_client.containers.run("{}:latest".format(container_name), container_cmd, detach=True,
                                             volumes=volumes_dir, working_dir=WORKDIR_PATH,
                                             entrypoint="python")

    # Handle unexpected exist from manager and kill container in that case
    # This does not work for SIGKILL! e.g. when using pycharm's stop button,
    # for that see: https://youtrack.jetbrains.com/issue/PY-13316#focus=Comments-27-4240420.0-0
    def stop_container(sig=None, frame=None):
        logger.info("!Some Error Occurred, Stopping Container!")
        container.remove(v=True, force=True)

    signal.signal(signal.SIGTERM, stop_container)
    signal.signal(signal.SIGINT, stop_container)

    # Try / catch to avoid endless running containers
    try:
        log_start_len = 0
        # While True
        while True:

            log_msgs = container.logs().decode("utf-8")
            # Get only latest log msgs
            log_msgs_list = log_msgs.split("\n")
            latest_output = "\n".join(log_msgs_list[log_start_len:])
            log_start_len = len(log_msgs_list) - 1
            latest_output = latest_output.rstrip("\n")

            # Print if not empty
            if latest_output:
                print(latest_output)

            # Stop condition (not moved to loop to have a do-while loop)
            container.reload()
            if container.status == "exited":
                break

            # Sleep to avoid pulling too often
            time.sleep(1)

        # Catch if container code failed
        container.reload()
        if container.attrs["State"]["ExitCode"] != 0:
            raise RuntimeError("Container failed to finish correctly! See latest output for more info. " +
                               "Exit Code was: {}".format(container.attrs["State"]["ExitCode"]))

        # Stop if exit successfully
        container.remove(v=True, force=True)

    except Exception as e:  # blanket exception to make sure the container is stopped
        stop_container()
        raise e

    logger.info("Successfully ran the code for the library in the container.")


if __name__ == "__main__":
    # Preprocessing to figure out which libraries have run so far and which not
    # (takes some time as it requires to start each container)
    for lib_name in get_all_lib_names():
        start_code_in_container_handler(lib_name, script_to_run="evaluation/run_overhead_mgmt.py")
    logger.info("Finished preprocessing of libraries")

    # Do benchmarks
    # only runs code for dataset-algorithm combinations that have not been collected so far (change in benchmark main)
    for lib_name in get_all_lib_names():
        start_code_in_container_handler(lib_name, script_to_run="libraries/main_benchmark.py")

    logger.info("Finished all libraries")
