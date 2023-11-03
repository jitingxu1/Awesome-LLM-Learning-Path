## Data Movement
- [Data Movement is All your Need](https://arxiv.org/pdf/2007.00072.pdf)
## Learnings
- To keep the GPU’s at high utilization, we need to make sure that by the end of a training step on a mini-batch(forward + backward pass), the next mini-batch will be ready to transfer into the GPU’s memory.
    - Preparing the mini-batch for the GPU includes the following steps:
    - Deciding which examples need to be loaded (typically employing shuffling of the datasets)
    - Loading examples from the storage ( IO)
    - Transforming such as pre-processing or augmentation (CPU)
    - Storing them in RAM (CPU)
    - Transferring tensors into the GPU memory (CPU)



