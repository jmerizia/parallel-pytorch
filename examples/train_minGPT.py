# Run this program with `mpirun`. For example:
#   mpirun -np 4 python ./examples/train_minGPT.py --pp=2 --mp=2

import torch
import fire
import logging
from parallel_pytorch.data import aggregate_gradients, scatter_batch
from mpi4py import MPI

from parallel_pytorch.models.minGPT import configure_optimizers, criterion, make_pipelined_GPT
from parallel_pytorch.topology import Topology
from parallel_pytorch.utils import set_seed

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)


def main(
    dp=1,
    pp=1,
    mp=1,
    batch_size=8,
    vocab_size=17,
    block_size=4,
    n_layer=2,
    n_head=4,
    n_embd=4,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
    learning_rate=0.1,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    from_checkpoint=False,
    seed=42,
):
    """ Train a simple minGPT model with 3D parallelism. """

    # We first have to create a "topology" which is a slim class which holds information
    # about the overall shape of the network.
    topo = Topology(dp=dp, pp=pp, mp=mp)

    # We set the seed in torch/numpy/random to the current rank to ensure that weight initialization
    # happens differently on all ranks.
    set_seed(seed * topo.data_comm.Get_rank())

    # Here, we load in our pipelined minGPT. The one caveat to be aware of is that
    # this is not a torch.nn.Module, but rather a "Pipeline" class.
    # It still has forward/backward functions, so we can use it *almost* normally.
    pipeline = make_pipelined_GPT(
        topo=topo,
        block_size=block_size,
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        embd_pdrop=embd_pdrop,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
    )

    if from_checkpoint:
        # Here is how you load a checkpoint. Pipeline and any ParallelModule objects have a `parallel_load_state_dict()`
        # function which can load a state dict. This is similar to torch.nn.Module.load_state_dict(),
        # but it is recursive, which makes it easier to deal with in parallel.
        state_dict = torch.load(f'checkpoint_stage{topo.get_pipeline_stage_idx()}.pt')
        pipeline.parallel_load_state_dict(state_dict)

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

    # Here's how we can save our model. Pipelines and any ParallelModule objects have a `parallel_state_dict()`
    # function which is a recursive replacement to `state_dict()`.
    state_dict = pipeline.parallel_state_dict()
    # Now, the state dict will be different depending on which stage of the pipeline we are at,
    # and it will be collapsed in the model parallel dimension.
    if topo.is_root_model_rank() and topo.get_data_parallel_idx() == 0:
        torch.save(state_dict, f'checkpoint_stage{topo.get_pipeline_stage_idx()}.pt')

    MPI.COMM_WORLD.Barrier()


if __name__ == '__main__':
    fire.Fire(main)
