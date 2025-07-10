# Docker

This docker builds all tools required to run and evaluate ZConv.


## Requirements
The docker image uses about 12 GB.


## Build

To build the image, add the following files to this directory.

- blis-yaconv.patch: patch with changes to the BLIS library to add Yaconv
- deeplabv3plus-zconv.patch: patch to deeplabv3plus to enable running it with ZConv
- pytorch-zconv.patch: patch to PyTorch to add a ZConv convolution backend
- VOCtrainval_11-May-2012.tar: dataset utilized by deeplabv3plus
- zero-copy-conv.zip: this repo, archived into a zip file. You can generate it by running `git archive -o zero-copy-conv.zip`.

These files, other than the last one, are provided by the artifact of ZConv, which will be published in the future.

Then, build the docker image with:
```bash
docker build --progress=plain -t zconv .
```

## Run
```bash
# Create the container
docker create -it --name artifact zconv:latest
# Start it
docker start artifact
# Attach to it
docker exec -it artifact bash
# Stop it
docker stop
```

## Perf support

To enable running perf inside the docker container, do the following.

- Run in the host:
```bash
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
```

- Add `--privileged` to the docker create command
