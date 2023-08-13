# The RWKV Language Model (and my LM tricks)

This is the experimental branch for people want to use fp 16 to train some world serial models.
For easy usage, I just copy data preprocessor from https://github.com/SynthiaDL/TrainChatGalRWKV/


## RWKV: Parallelizable RNN with Transformer-level LLM Performance (pronounced as "RwaKuv", from 4 major params: R W K V)

RWKV is an RNN with Transformer-level LLM performance, which can also be directly trained like a GPT transformer (parallelizable). And it's 100% attention-free. You only need the hidden state at position t to compute the state at position t+1. You can use the "GPT" mode to quickly compute the hidden state for the "RNN" mode.

So it's combining the best of RNN and transformer - **great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding** (using the final hidden state).

The are many other information, please visit: https://github.com/BlinkDL/RWKV-LM