```class LLM
```

    - Designed for offline use, suggest to put all prompts into one list, and it will automatically calculate the batch size bases on memory contraint.
    - gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache. Higher values will increase the KV cache size and thus improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors.
