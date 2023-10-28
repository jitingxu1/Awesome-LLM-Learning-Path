## RLHF
- [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)
    - pretrain a Language Model
    - gathering data and train a reward model
        - reward model could be another LLM or ranking system could return a scalar as the reward score
        - Need to train reward model with human perferences before the RL 
    - finetune the LM with reinforced learning
        - Maximize the reward (reward + KL divergence penalty) using RL
        - Cannot directly update the LM via backpropagation becuase reward is not differentiable (such as decoder), have to use RL PPO.
- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/pdf/2309.00267.pdf)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
    - Existing methods for fine-tuning language models with human feedback first fit a reward model to a dataset of prompts and human preferences over pairs of responses, and then use RL to find a policy that maximizes the learned reward. In contrast, DPO directly optimizes for the policy best satisfying the preferences with a simple classification objective, without an explicit reward function or RL.
    - key insight is to leverage an analytical mapping from reward functions to optimal policies, which enables us to transform a loss function over reward functions into a loss function over policies.

## Tools
[A modular RL library to fine-tune language models to human preferences](https://github.com/allenai/RL4LMs)
