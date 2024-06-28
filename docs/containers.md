# Containers

NVIDIA provides the [JAX
Toolbox](https://github.com/NVIDIA/JAX-Toolbox) containers, which are
bleeding edge containers containing nightly releases of jax and some
models/frameworks. Example usage: `docker run -it --shm-size=1g --gpus
all ghcr.io/nvidia/jax:pax-2024-06-03`. XLA is under `/opt/xla`.

TensorFlow has some caontainers too: [TensorFlow's docker
container](https://www.tensorflow.org/install/docker):
