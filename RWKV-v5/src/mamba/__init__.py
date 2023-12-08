__all__=["mamba_model"]

from .mixer_seq_simple import MambaLMHeadModel
import torch
from torch.nn import functional as F
import os, importlib
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    
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

class WrappedMambaLHHeadModel(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        backbone_kwargs = {
            "ssm_cfg": {},
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            
        }
        
        if args.mamba_size == "130m":
            args.n_layer = 24
            args.n_embd = 768
            
        self.mamba_model = MambaLMHeadModel(
            d_model=args.n_embd,
            n_layer=args.n_layer,
            vocab_size=args.vocab_size,
            initializer_cfg=None,
            pad_vocab_size_multiple=8,
            device="cuda",
            dtype=torch.float,
            **backbone_kwargs,
        )
            
    def configure_optimizers(self):
        args = self.args
        
        lr_set = set()
        for n, p in self.named_parameters():
            lr_set.add(n)
            
        lr_set = sorted(list(lr_set))
        param_dict = {n: p for n, p in self.named_parameters()}
        optim_groups = [{"params": [param_dict[n] for n in lr_set], "weight_decay": 0.1, "my_lr_scale": 1.0}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
        else:
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)

    def forward(self, input_ids):
        return self.mamba_model.forward(input_ids, grad_cp=self.args.grad_cp)
    
    def training_step(self, batch, batch_idx):
        args = self.args
        if args.my_qa_mask != 1:
            idx, targets = batch
            logits = self(idx)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        else:
            idx, targets, mask = batch
            mask = mask.view(-1)
            sum_mask = torch.sum(mask).item()
            # if sum_mask == 0:
            #     return torch.tensor([0.0], requires_grad=True)

            logits = self(idx)
            if sum_mask == mask.shape[0]:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                # print('rank', self.global_rank, 'loss', loss.item())
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                # loss_raw = loss
                loss = torch.sum(loss * mask) / sum_mask

        return L2Wrap.apply(loss, logits)
    
    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
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
    
    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        for n in self.state_dict():
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                self.state_dict()[n] = self.state_dict()[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                self.state_dict()[n] = self.state_dict()[n].bfloat16()

        return self.state_dict()
        