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

        embedding = nn.Embedding(args.vocab_size, args.decoder_embed_dim)
        nn.init.normal_(embedding.weight, mean=0, std=args.decoder_embed_dim ** -0.5)
        # TODO: not sure how padding_idx works in fairseq framework, check later.
        # nn.init.constant_(embedding.weight[padding_idx], 0)
        
        output_projection = nn.Linear(args.decoder_embed_dim, args.vocab_size, bias=False)
        torch.nn.init.normal_(
            output_projection.weight, mean=0, std=args.decoder_embed_dim**-0.5
        )
        
        
        self.retnet = RetNetDecoder(args, embedding, output_projection)

    def forward(self, src_tokens, **kwargs):
        x, _ =  self.retnet.forward(src_tokens, **kwargs)
        return x
    
    def max_positions(self):
        #NOTE: seems not useful, not sure
        return self.args.max_target_positions

    def generate_init_weight(self):
        '''
            Accroding to https://arxiv.org/pdf/2203.00555.pdf and https://arxiv.org/pdf/2307.08621.pdf,
            retnet use xavier_normal_ for all most of layers. most of layers in torchscale has included this.
            TODO: need to make sure what to do with embedding.

            embedding initialization is from fairseq repo:

            def Embedding(num_embeddings, embedding_dim, padding_idx):
                m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
                nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
                nn.init.constant_(m.weight[padding_idx], 0)
                return m

            o_proj is from torchscale repo.
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.decoder_embed_dim**-0.5
            )

            layers like 'ffn', 'v_proj', 'out_proj' 'q_proj', 'k_proj' use nn.init.xavier_normal_ when 
            initializeing MultiScaleRetention/FeedForwardNetwork with reset_parameters function
            
            Seems no special initialzation in ffn/final layer norm, means torch default when initialized:
            ones(weight); zeros(bias)
            
            The only remain layer is RetNetRelPos, was initialized with some hard coded function, based on config parameters:
            angle = 1.0 / (10000 ** torch.linspace(0, 1, args.decoder_embed_dim // args.decoder_retention_heads // 2))
            angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
            decay = torch.log(1 - 2 ** (-5 - torch.arange(args.decoder_retention_heads, dtype=torch.float)))
        '''
        # Nothing need to be initialize here, so just copy the state dict for saving.
        m = {}
        for n in self.state_dict():
            if not self.args.deepnorm:
                NotImplementedError
            p = self.state_dict()[n]
            shape = p.shape
            m[n] = p
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
        '''
            Taken from fairseq/fairseq/optim/adam.py
            @register_optimizer("adam", dataclass=FairseqAdamConfig)
            class FairseqAdam(FairseqOptimizer):
                """Adam optimizer for fairseq.

                Important note: this optimizer corresponds to the "AdamW" variant of
                Adam in its weight decay behavior. As such, it is most closely
                analogous to torch.optim.AdamW from PyTorch.
                """
        '''
        args = self.args
        #NOTE: just quickly initialize to make code works.
        param_dict = {n: p for n, p in self.named_parameters()}
        optim_groups = [ {"params": [param_dict[n] for n in param_dict], "weight_decay":0.0, "my_lr_scale":1.0},]
        # RETNET PAPER USED weight_decay = 0.01
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, 
                        betas=self.args.betas, eps=self.args.adam_eps, 
                        bias_correction=True, adamw_mode=False, weight_decay=args.weight_decay, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, 
                        betas=self.args.betas, eps=self.args.adam_eps, 
                        bias_correction=True, adam_w_mode=False, weight_decay=args.weight_decay, amsgrad=False)

def retnet_rwkv_config_amap(args):
    #NOTE: I am not confident about my understanding.
    setattr(args, "decoder_embed_dim", args.n_embd)
    setattr(args, "decoder_ffn_embed_dim", args.dim_ffn)
    setattr(args, "decoder_attention_heads", args.head_qk)
    setattr(args, "decoder_layers", args.n_layer)
    
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
    elif args.retnet_official_name == "retnet_rwkvconf":
        actual_configuration = retnet_rwkv_config_amap
    else:
        NotImplementedError
    
    # use deepnorm to initialize
    setattr(args, "deepnorm", True)
    # this is how fairseq called ctxlen
    setattr(args, "tokens_per_sample", args.ctx_len)
    
    
    actual_configuration(args)
    retnet_config = Wrapper_RetNetConfig()
    retnet_config.override_and_update(args)
    model = Wrapper_RetNet(retnet_config)
    return model