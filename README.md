<img src="./3d.png" width="400px"></img>

image source:
<a href="https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/">DeepSpeed</a>

## Parallel Pytorch

A (WIP) lightweight implementation of DeepSpeed/Megatron-ML style 3D parallelism,
along with some models and helpful utilities.
It is based around the [DistDL](https://github.com/distdl/distdl) programming model,
but aims to be feature complete for transformers.

## Todo

core
- [X] model parallel primitives
- [X] pipeline scheduler
- [X] data parallel
- [X] checkpointing
- [ ] MoE
- [ ] weight initialization same across all scales
- [ ] tests for correctness across various scales
- [ ] enable sending arbitrary objects in pipeline scheduler
- [ ] CUDA support
- [ ] Dockerfile
- [ ] ability to run larger models on less GPU space with pipeline+offload
- [ ] function to split pipeline layers by # params

models
- [X] minGPT
- [ ] GPT-2
- [ ] ViT
- [ ] GPT-J

future performance improvements
- [ ] benchmark against existing solutions
- [ ] gradient checkpointing for pipeline
- [ ] benchmarking with NVTX
- [ ] ZeRO optimizations
- [ ] contiguous allocator for temporary buffers
- [ ] more meticulous buffer sharing (based on lifetimes)

misc
- [ ] torchtyping

## Citations

```bibtex
@misc{shoeybi2020megatronlm,
      title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism}, 
      author={Mohammad Shoeybi and Mostofa Patwary and Raul Puri and Patrick LeGresley and Jared Casper and Bryan Catanzaro},
      year={2020},
      eprint={1909.08053},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```bibtex
@misc{narayanan2021efficient,
      title={Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM}, 
      author={Deepak Narayanan and Mohammad Shoeybi and Jared Casper and Patrick LeGresley and Mostofa Patwary and Vijay Anand Korthikanti and Dmitri Vainbrand and Prethvi Kashinkunti and Julie Bernauer and Bryan Catanzaro and Amar Phanishayee and Matei Zaharia},
      year={2021},
      eprint={2104.04473},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```bibtex
@misc{rajbhandari2020zero,
      title={ZeRO: Memory Optimizations Toward Training Trillion Parameter Models}, 
      author={Samyam Rajbhandari and Jeff Rasley and Olatunji Ruwase and Yuxiong He},
      year={2020},
      eprint={1910.02054},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```bibtex
@misc{rajbhandari2022deepspeedmoe,
      title={DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale}, 
      author={Samyam Rajbhandari and Conglong Li and Zhewei Yao and Minjia Zhang and Reza Yazdani Aminabadi and Ammar Ahmad Awan and Jeff Rasley and Yuxiong He},
      year={2022},
      eprint={2201.05596},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```bibtex
@misc{rajbhandari2021zeroinfinity,
      title={ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning}, 
      author={Samyam Rajbhandari and Olatunji Ruwase and Jeff Rasley and Shaden Smith and Yuxiong He},
      year={2021},
      eprint={2104.07857},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```

```bibtex
@misc{hewett2020linear,
      title={A Linear Algebraic Approach to Model Parallelism in Deep Learning}, 
      author={Russell J. Hewett and Thomas J. Grady II au2},
      year={2020},
      eprint={2006.03108},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
