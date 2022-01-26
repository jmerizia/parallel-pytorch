This is a collection of various PyTorch models implemented with full 3D-parallelism, along with some helpful utilities.

todo list

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