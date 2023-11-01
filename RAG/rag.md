## Blogs
- [Fix Hallucination in Retrieval Augmented Generation AI Applications Using Schema and Output Parser](https://kelly-kang.medium.com/fix-hallucination-in-retrieval-augmented-generation-ai-applications-using-schema-and-output-parser-d58325daf1da)


## Learnings
- [in Context Learning (ICL)](https://www.hopsworks.ai/dictionary/in-context-learning-icl#:~:text=In%2Dcontext%20learning%20(ICL),the%20need%20for%20fine%2Dtuning.)
    - Add the most relavent context at the beginning or the end of a prompt improve the performance of LLMs, [researchers](https://arxiv.org/abs/2307.03172) have shown that adding relevant context in the middle of the prompt leads to worse performance. 
    - In-context learning benefits from larger context window sizes: the larger the context size, the better the performance is. Need to use LLm with large context size. 
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
    - [github repo](https://github.com/AkariAsai/self-rag/tree/main)
    - Two issues in lots of RAG system: 
        1) irrelevant context is retrieved 
        2) Generated texts is not consistent with the retrieved context
    - Paper contribution
        1) Our framework trains a single arbitrary LM that adaptively retrieves passages on-demand (e.g., can retrieve multiple times during generation, or completely skip retrieval)
            -  Retrieve: Self-RAG first decodes a retrieval token to evaluate the utility of retrieval and control a retrieval component. If retrieval is required, our LM calls an external retrieval module to find top relevant documents, using input query and previous generation.
            - Generate: If retrieval is not required, the model predicts the next output segment, as it does in a standard LM. If retrieval is needed, the model first generates generates critique token evaluating whether retrieved documents are relevant, and then generate continuation conditioned on the retrieved passages.
            - Critique: If retrieval is required, the model further evaluates if passages support generation. Finally, a new critique token evaluates the overall utility of the response.
        2)  generates and reflects on retrieved passages and its own generations using special tokens, called reflection tokens.
    - Steps
        - Self-RAG training consists of three models, a Retriever, a Critic and a Generator.
    - Performance
        - 