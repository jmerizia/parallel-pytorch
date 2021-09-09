# Distributed minGPT

This is a parallel implementation of [minGPT](https://github.com/karpathy/minGPT),
which aims to make minimal code changes (~50 more lines of code) to achieve efficient parallelism.

While there exist many parallel implementations of transformers,
their codebases are sprawling and have grown to the order of tens of thousands
of lines of code.
The motivation of this project is to show that parallelizing transformers
across hundreds of CPUs or GPUs can be done simply.

This project relies on the parallel primitives developed in the [DistDL](http://github.com/distdl/distdl)
deep learning framework.
DistDL allows us to parallelize arbitrary networks and is interoperable with PyTorch.

