import torch
import torch.nn as nn
import fire
import logging
from parallel_pytorch.data import aggregate_gradients, scatter_batch

from parallel_pytorch.models.minGPT import configure_optimizers, criterion, make_pipelined_GPT
from parallel_pytorch.ops import AllSumReduce, Broadcast
from parallel_pytorch.pipeline import Pipeline
from parallel_pytorch.topology import Topology
from parallel_pytorch.utils import global_rank, set_seed, split_list

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)


def build_model(
    *,
    topo: Topology,
    n_embd: int,
    n_hidden: int,
    n_blocks: int,
    device=None,
):
    topo = topo
    size = topo.model_comm.Get_size()
    assert n_hidden % size == 0
    assert n_blocks == topo.get_num_pipeline_stages()
    stages = [
        nn.Sequential(
            Broadcast(topo.model_comm),
            nn.Linear(n_embd, n_hidden // size, device=device),
            nn.ReLU(),
            nn.Linear(n_hidden // size, n_embd, device=device),
            AllSumReduce(topo.model_comm),
        )
        for _ in range(n_blocks)
    ]
    pipeline = Pipeline(topo=topo, stages=stages)
    return pipeline


def run_test():
    topo = Topology(dp=dp, pp=pp, mp=mp)

    # We set the seed in torch/numpy/random to the current rank to ensure that weight initialization
    # happens differently on all ranks.
    set_seed(seed * topo.data_comm.Get_rank())

    pipeline = build_model(
        topo=topo,
    )

    # Generate some fake data. Technically we only need it on the first and last stages of the pipeline,
    # but text data isn't so expensive to load in on all ranks.
    data = [
        (
            torch.randint(0, vocab_size, [batch_size, block_size], dtype=torch.long),
            torch.randint(0, vocab_size, [batch_size, block_size], dtype=torch.long),
        ) for _ in range(10)
    ]

    # This function also doesn't really change from the original implementation.
    optimizer = configure_optimizers(
        pipeline=pipeline,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )

    running_loss = 0
    for it, (x, y) in enumerate(data):

        # First we want to scatter the batch across all of the data parallel copies.
        x, y = scatter_batch(topo=topo, inputs=x, labels=y)

        optimizer.zero_grad()

        # As usual, forward the model and compute loss.
        # Under the hood, this is passing our input through the entire pipeline.
        logits = pipeline(x)
        loss = criterion(topo, logits, y)

        # Now we do the backwards pass. As mentioned before, since the pipeline is not technically a module,
        # when we do backward on the loss, that populates logits.grad normally,
        # but it doesn't actually propagate the loss down the rest of the pipeline for us.
        # This means we must call `backward()` manually.
        loss.backward()
        pipeline.backward(logits.grad)

        # Now, for each of our parameters, PyTorch has populated a `.grad` member on each of our parameters.
        # Since we are doing data parallelism, we must aggregate these gradients now.
        aggregate_gradients(topo=topo, model=pipeline.stage)

        # Fortunately, PyTorch aggregates the gradients for us if we call `forward()` multiple times
        # (which we do in the pipeline with "micro batches").
        # Thus, we can just step the optimizer as we normally do.
        optimizer.step()

        # This step deserves some explanation. Since we are pipelining the input, we can only use logits/loss
        # if we're at the last stage of the pipeline.
        # Additionally, since there might be multiple model-parallel processes, we must make sure
        # to print on just the root one.
        # Lastly, since there are multiple data parallel copies, we want to only print on the first one.
        if topo.is_last_pipeline_stage() and topo.is_root_model_rank() and topo.get_data_parallel_idx() == 0:
            running_loss += loss.item()
            logger.info(f'batch {it} loss: {running_loss:.3f}')
            running_loss = 0.0


if __name__ == '__main__':
    fire.Fire(main)

