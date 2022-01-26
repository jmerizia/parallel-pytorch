<img src="./3d.png" width="500px"></img>

image source:
<a href="https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/">DeepSpeed</a>

## Parallel Pytorch

A (WIP) collection of various PyTorch models implemented with full 3D-parallelism based on DeepSpeed/Megatron-ML, along with some helpful utilities.

## Todo

- [ ] core shared utils
    - [X] model parallel primitives
    - [ ] pipeline scheduler
    - [ ] data parallel
    - [ ] checkpointing
- [ ] models
    - [ ] minGPT
    - [ ] ViT
    - [ ] GPT-J
- [ ] tests for correctness across various scales
- [ ] benchmark against existing solutions

more niche things
- [ ] enable sending arbitrary objects in pipeline scheduler

future performance improvements
- [ ] only send input to the first pipeline stage
- [ ] gradient checkpointing for pipeline

## Citations

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