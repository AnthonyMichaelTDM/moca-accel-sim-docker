# moca-accel-sim-docker

## Usage

> **Note**
> For convenience, there is the provided `run-container.sh` script that will:
>
> 1. run `setup.sh` to download necessary context (this script will only download the context if it hasn't been downloaded yet)
> 2. build the docker image if it hasn't been built yet
> 3. run the docker image in interactive mode with the shared volume (`./shared`) mounted.

First, after cloning the repository, run the setup script to download some necessary context:

```bash
./setup.sh
```

Then, build the docker image:

```bash
docker build -t moca-accel-sim .
```

Finally, run the docker image:

```bash
docker run -it moca-accel-sim
```

If you want to run the docker image with a shared directory, you can use the following command:

```bash
docker run -it -v /path/to/shared/directory:/shared moca-accel-sim
```

And if you want to give the container access to your GPU, you can add the `--gpus all` (you must also have the nvidia-container-toolkit installed) flag to the `docker run` command:

```bash
docker run -it --gpus all moca-accel-sim
```

Lastly, if you want to attach VSCode to the container, you do so with the docker extension. (first, run the container in interactive mode in a terminal, THEN you can attach VSCode to the container).

## Once in the container

once in the container (each step can be done in a separate terminal, and 2 can be skipped if you already have traces):

1. install the necessary dependencies

```bash
cd /shared/models
pip install -r requirements.txt
```

2. generate traces

```bash
# modify /accel-sim/accel-sim-framework/util/job_launching/apps/define-moca-apps.yml to configure what workloads you want to trace

# set up the tracer
cd /accel-sim-framework/util/tracer_nvbit
./install_nvbit.sh
make

# generate the traces
./run_hw_trace.py -B moca

# this will output traces in `/accel-sim/accel-sim-framework/hw_run/traces/device-0/12.8/...`, but you may want to move them to the `/shared` directory so you can access them from outside the container
```

3. running in the simulator and collecting stats

```bash
# modify /accel-sim/accel-sim-framework/util/job_launching/apps/define-moca-apps.yml to configure what workloads you want to simulate

# set up environment
cd /accel-sim-framework/
source gpu-simulator/setup_environment.sh
cd util/job_launching

# run the simulator
export SIM_RUN="moca" # or whatever you want to call the run
./run_simulations.py -B moca -C RTX3060-SASS -T /shared/hw_run/traces/device-0/12.8/ -N $SIM_RUN

# monitor the simulation
./monitor_func_test.py -N $SIM_RUN -v

# once it's done, collect the stats
./get_stats.py -K -k -N $SIM_RUN | tee /shared/<path_to_output_file>.csv
```

## Processing stats

once you have a `stats.csv` file, you can process it with some of the scripts in `shared`

```bash
# organize the stats into a table that we can use for plotting
# this will create a new file with the same name but with `organized_` appended to its name
python3 ./organized_sim_results.py <path_to_stats.csv> 

# generate the plots
python3 ./visualize_results.py <path_to_organized_stats.csv> --output_dir <path_to_put_plots>
```

though before running the `visualize_results.py` script you'll want to modify the the `kernel_names` and `kernels_to_keep` variables in the script to match the kernels you want to plot and give them shorter names.
`organized_sim_results.py` will give you the demangled kernel names in its output and will print the list of unique names, so you don't need to worry about that.

the `instructions_count.py` script operates on the raw traces, not the stats, so you can run it before and w/o actually running the simulator.

## Getting a debug build of AccelSim

if you want to get a debug build of AccelSim in the container, you can do so by running the following commands:

```bash
cd /accel-sim-framework/
source gpu-simulator/setup_environment.sh

cmake -S gpu-simulator -B gpu-simulator/build -DCMAKE_BUILD_TYPE=Debug
cmake --build ./gpu-simulator/build
cmake --install ./gpu-simulator/build
```
