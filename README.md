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
