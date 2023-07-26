########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
#
# This file is used for wrapping the retnet official model implementation.
########################################################################################################

# from retnet
from .config import RetNetConfig
from .retnet import RetNetDecoder
# DEFAULT_MAX_TARGET_POSITIONS = 1024
from .configurate_retnet import *

import torch.nn as nn
import torch
import gc
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# copy from src/model.py
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

class Wrapper_RetNetConfig(RetNetConfig):
    def override_and_update(self, rwkv_args):
        '''
            Just override all variables from rwkv trainer, 
            TODO: compare variables with different names but same concept.
        '''
        for hp in vars(rwkv_args):
            setattr(self, hp, getattr(rwkv_args, hp))

class Wrapper_RetNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Initial Embedding with RWKV style, instead of using fairseq.
        embedding = nn.Embedding(args.vocab_size, args.decoder_embed_dim)
        output_projection = nn.Linear(args.decoder_embed_dim, args.vocab_size, bias=False)
        self.retnet = RetNetDecoder(args, embedding, output_projection)

    def forward(self, src_tokens, **kwargs):
        x, _ =  self.retnet.forward(src_tokens, **kwargs)
        return x
    
    def max_positions(self):
        #NOTE: seems not useful, not sure
        return self.args.max_target_positions

    def generate_init_weight(self):
        #NOTE: just quickly initialize to make code works.
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape
            if "layer_norm" in n or "layernorm" in n:
                m[n] = p
            else:
                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty(shape, device="cuda")
                else:
                    m[n] = torch.empty(shape)
                nn.init.uniform_(m[n])
        gc.collect()
        torch.cuda.empty_cache()
        return m

    def training_step(self, batch, batch_idx):
        args = self.args
        if args.my_qa_mask != 1:
            #NOTE: skip qa mask.
            idx, targets = batch
            logits = self(idx)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def configure_optimizers(self):
        args = self.args
        #NOTE: just quickly initialize to make code works.
        param_dict = {n: p for n, p in self.named_parameters()}
        optim_groups = [ {"params": [param_dict[n] for n in param_dict], 
                        "weight_decay": 0.0, "my_lr_scale": 1.0},]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, 
                        betas=self.args.betas, eps=self.args.adam_eps, 
                        bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, 
                        betas=self.args.betas, eps=self.args.adam_eps, 
                        bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)

def get_retnet_model(args):
    actual_configuration = None
    if args.retnet_official_name == "retnet_base":
        actual_configuration = retnet_base_architecture
    elif args.retnet_official_name == "retnet_medium":
        actual_configuration = retnet_medium
    elif args.retnet_official_name == "retnet_xl":
        actual_configuration = retnet_xl
    elif args.retnet_official_name == "retnet_3b":
        actual_configuration = retnet_3b
    elif args.retnet_official_name == "retnet_7b":
        actual_configuration = retnet_7b
    elif args.retnet_official_name == "retnet_13b":
        actual_configuration = retnet_13b
    elif args.retnet_official_name == "retnet_65b":
        actual_configuration = retnet_65b
    else:
        NotImplementedError
    
    actual_configuration(args)
    retnet_config = Wrapper_RetNetConfig()
    retnet_config.override_and_update(args)
    model = Wrapper_RetNet(retnet_config)
    return model