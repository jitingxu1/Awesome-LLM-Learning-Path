## [Multiquey attention](https://blog.fireworks.ai/multi-query-attention-is-all-you-need-db072e758055)

## vLLM papers
    - [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180.pdf)
## concepts
- Continuous Batching [Anyscale blog](https://www.anyscale.com/blog/continuous-batching-llm-inference)[todo]
    - hugging face: text-generation-inference
    - vLLM: 23x inference throuput and reducing p50 latency
- PageAttention - vLLM [todo]
    - In the autoregressive decoding process, all the input tokens to the LLM produce their attention key and value tensors, and these tensors are kept in GPU memory to generate next tokens. These cached key and value tensors are often referred to as KV cache
        - Large: Takes up to 1.7GB for a single sequence in LLaMA-13B
        - Dynamic: Its size depends on the sequence length, which is highly variable and unpredictable. As a result, efficiently managing the KV cache presents a significant challenge. We find that existing systems waste 60% – 80% of memory due to fragmentation and over-reservation.
    - PagedAttention allows storing continuous keys and values in non-contiguous memory space. Specifically, PagedAttention partitions the KV cache of each sequence into blocks, each block containing the keys and values for a fixed number of tokens.
    - During the attention computation, the PagedAttention kernel identifies and fetches these blocks efficiently.
    - Because the blocks do not need to be contiguous in memory, we can manage the keys and values in a more flexible way as in OS’s virtual memory: one can think of blocks as pages, tokens as bytes, and sequences as processes. The contiguous logical blocks of a sequence are mapped to non-contiguous physical blocks via a block table. The physical blocks are allocated on demand as new tokens are generated.
    - In PagedAttention, memory waste only happens in the last block of a sequence. In practice, this results in near-optimal memory usage, with a mere waste of under 4%.