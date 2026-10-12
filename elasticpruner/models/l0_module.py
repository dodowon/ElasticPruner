# Based on princeton-nlp/LLM-Shearing (MIT License), Copyright (c) 2023 Princeton Natural Language Processing
# Modified by dodowon, 2026. See THIRD_PARTY_NOTICES.md for details.

import math
import sys

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from argparse import Namespace as NS
from typing import Any, List

from composer.core.time import Time

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6 ## epsilon 1e-6

class Mask(nn.Module):
    def __init__(self, 
                 name: str,
                 mask_shape: List, 
                 num_params_per_mask: int, 
                 mask_output_shape: List, 
                 target_sparsity: float,
                 target_mask_size: int,
                 device: str,
                 eval_target_model: bool=True) -> None:
        super().__init__()
        self.name = name
        self.num_params_per_mask = num_params_per_mask
        self.mask_output_shape = mask_output_shape
        self.target_sparsity=target_sparsity

        self.droprate_init = 0.5 # 0.5
        self.temperature = 2./3. 
        # self.temperature = 1.0
        self.magical_number = 0.8
        # self.magical_number = 1.0
        self.device = device
        
        self.z_loga = self.initialize_mask(mask_shape) 
        self.mask_size = self.z_loga.shape[-1] # the full size of each unit
        self.target_mask_size = target_mask_size
        self.eval_target_model = eval_target_model
        
        
    def param_init_fn(self, module):
        """ Initialize the parameters for masking variables. """
        mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        # mean = 5
        # mean = 1
        if isinstance(module, nn.Parameter):
            module.data.normal_(mean, 1e-2)
        else:
            for tensor in module.parameters():
                tensor.data.normal_(mean, 1e-2)
        
    def initialize_mask(self, mask_shape: List):
        """ Initialize the parameters for masking variables. """
        z_loga = nn.Parameter(torch.ones(*mask_shape, device=self.device))
        self.param_init_fn(z_loga)
        return z_loga

    def cdf_qz(self, z_loga: torch.Tensor = None):
        """Implements the CDF of the 'stretched' concrete distribution"""
        if z_loga is None:
            z_loga = self.z_loga
        xn = (0 - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - z_loga).clamp(min=epsilon, max=1 - epsilon)
    
    def get_eps(self, size: List):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps) # is it a must?
        return eps
    
    def quantile_concrete(self, eps: torch.Tensor):
        y = torch.sigmoid((torch.log(eps) - torch.log(1 - eps) + self.z_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    # def quantile_concrete(self, eps: torch.Tensor):
    #     alpha = torch.exp(self.z_loga)
    #     safe_alpha = torch.clamp(alpha, min=epsilon)
    #     y = torch.sigmoid((torch.log(eps) - torch.log(1 - eps) + self.z_loga - torch.log(1-safe_alpha)) / self.temperature)
    #     return y * (limit_b - limit_a) + limit_a
    
    def sample_z(self):
        eps = self.get_eps(torch.FloatTensor(*self.z_loga.shape)).to(self.z_loga.device)
        z = self.quantile_concrete(eps)
        z = F.hardtanh(z, min_val=0, max_val=1).reshape(*self.mask_output_shape)
        return z
    
    def _deterministic_z(self, z_loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        # print("## 00099 : ", self.mask_size, ", ", self.target_mask_size)
        # print("##009 ", self.target_mask_size, ", ", self.eval_target_model)
        if self.target_mask_size is None or not self.eval_target_model:
            expected_score = 1 - self.cdf_qz(z_loga)
            expected_num_nonzeros = expected_score.sum()
            expected_num_zeros = z_loga.nelement() - expected_num_nonzeros.item()
        else:
            expected_num_zeros = self.mask_size - self.target_mask_size 
        try:
            num_zeros = round(expected_num_zeros)
        except:
            print("num of zeros is nan....")
            sys.exit()
        soft_mask = torch.sigmoid(z_loga / self.temperature * self.magical_number)
        # print("num_zeros : ", num_zeros)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(self.z_loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask
    
    def deterministic_z(self):
        if self.z_loga.ndim == 1:
            z = self._deterministic_z(self.z_loga).reshape(*self.mask_output_shape)
        else:
            z_loga = self.z_loga.reshape(-1, self.z_loga.shape[-1])
            z = []
            for i in range(z_loga.shape[0]):
                z_ = self._deterministic_z(z_loga[i])
                z.append(z_)
            z = torch.stack(z).reshape(*self.mask_output_shape)
        # print(f"##04 {z.shape} , {(z==0).sum().item()}")
        # print(f"##05 {z.shape} , {(z==1.0).sum().item()}")
        return z
    
    def forward(self):
        print(self.training)
        func = self.sample_z if self.training else self.deterministic_z
        z = func(self.z_loga).reshape(self.mask_output_shape)
        return z            
        
    def constrain_parameters(self):
        self.z_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def calculate_expected_score_sparsity(self):
        score = 1 - self.cdf_qz()
        sparsity = 1 - score.sum(-1) / self.mask_size
        return score, sparsity
 
class L0Module(nn.Module):
    def __init__(self, cfg, device):
        super(L0Module, self).__init__()

        # base and target model info
        n_matrix_mlp = 2 if "pythia" in cfg.name else 3
        self.base_model_info = self.set_model_info(cfg, n_matrix_mlp=n_matrix_mlp) 
        l0_module_cfg = cfg.l0_module
        self.l0_module_cfg = cfg.l0_module
        self.target_model_info = None
        target_model_cfg = getattr(l0_module_cfg, "target_model", None)
        if target_model_cfg is not None:
            self.target_model_info = self.set_model_info(target_model_cfg, n_matrix_mlp=n_matrix_mlp)
        
        # l0 config
        self.pruning_modules = l0_module_cfg.pruning_modules        
        self.start_sparsity = l0_module_cfg.start_sparsity 
        self.lagrangian_warmup_steps = Time.from_timestring(l0_module_cfg.lagrangian_warmup_steps).value
        self.device = device
        self.eval_target_model = l0_module_cfg.get("eval_target_model", True)
        print("eval_target_model : ", self.eval_target_model)
        # l0 params
        self.lambdas = {}
        self.lambdas["lambda_1"] = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.lambdas["lambda_2"] = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.masks = {}
        for pruning_module in self.pruning_modules:
            self.initialize_one_module(pruning_module)
        self.masks = torch.nn.ModuleDict(self.masks)
        self.lambdas = torch.nn.ParameterDict(self.lambdas)
    
        # config after initialization
        self.prunable_model_size = self.calculate_prunable_model_size(self.base_model_info)
        if target_model_cfg is not None:
            self.prunable_target_model_size = self.calculate_prunable_model_size(self.target_model_info)
            self.target_sparsity = 1 - self.prunable_target_model_size / self.prunable_model_size
        else:
            self.target_sparsity = l0_module_cfg.target_sparsity

        print("********** Initializing L0 Module **********") 
        for pruning_module in self.pruning_modules:
            print(f"***** {pruning_module} *****")
            print(f"z.shape", self.masks[pruning_module].z_loga.shape)
            print(f"size", self.masks[pruning_module].mask_size)
        print(f"prunable model size: {self.prunable_model_size}")
        
    
    def set_model_info(self, cfg, n_matrix_mlp):
        ns = NS() 
        ns.hidden_size = cfg.d_model
        ns.intermediate_size = cfg.intermediate_size
        ns.num_attention_heads = cfg.n_heads
        ns.mlp_num_per_layer = 1
        ns.dim_per_head = ns.hidden_size // ns.num_attention_heads 
        ns.num_layers = cfg.n_layers
        ns.vocab_size = cfg.vocab_size

        ns.params_per_head_layer = ns.hidden_size * ns.hidden_size * 4
        ns.params_per_head =  ns.params_per_head_layer // ns.num_attention_heads
        ns.params_per_mlp_layer = ns.hidden_size * ns.intermediate_size * n_matrix_mlp
        ns.params_per_intermediate_dim = ns.params_per_mlp_layer // ns.intermediate_size

        ns.full_model_size = (ns.params_per_head_layer + ns.params_per_mlp_layer) * ns.num_layers
        return ns
        
    def calculate_prunable_model_size(self, ns: NS):
        prunable_mlp_size = ns.params_per_mlp_layer * ns.num_layers
        prunable_head_layer_size = ns.params_per_head_layer * ns.num_layers
        prunable_model_size = 0
        if "hidden" in self.pruning_modules:
            return prunable_mlp_size + prunable_head_layer_size
        if "head_layer" in self.pruning_modules or "head" in self.pruning_modules:
            prunable_model_size += prunable_head_layer_size
        if "mlp" in self.pruning_modules or "intermediate" in self.pruning_modules:
            prunable_model_size += prunable_mlp_size
        return prunable_model_size
        
    def initialize_one_module(self, module_name: str):
        func_name = f"initialize_{module_name}"
        try:
            method = getattr(self, func_name)
        except AttributeError:
            raise NotImplementedError("Instance `{}` does not implement `{}`".format(self, func_name))
        method()
            
    def initialize_hidden(self):
        mask_shape = [self.base_model_info.hidden_size]
        num_params_per_mask=self.base_model_info.hidden_size * 4 + self.base_model_info.hidden_size * 4 * 2
        
        target_hidden_sparsity = None; pd=None; target_mask_size=None; 
        if self.target_model_info is not None:
            target_hidden_sparsity = 1 - self.target_model_info.hidden_size / self.base_model_info.hidden_size
            target_mask_size = self.target_model_info.hidden_size
            # pd = {"lambda_1_hidden": torch.nn.Parameter(torch.tensor(0.01, device=self.device)),
            #       "lambda_2_hidden": torch.nn.Parameter(torch.tensor(0.05, device=self.device))}
            # self.lambdas.update(pd)
        
        hidden_mask = Mask(name="hidden",
                           mask_shape=mask_shape,
                           num_params_per_mask=num_params_per_mask,
                           mask_output_shape=[self.base_model_info.hidden_size],
                           target_sparsity=target_hidden_sparsity,
                           target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["hidden"] = hidden_mask

    def initialize_head(self):
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_heads]
        num_params_per_mask = self.base_model_info.params_per_head
        mask_output_shape = [self.base_model_info.num_layers, 1, self.base_model_info.num_attention_heads, 1] 
        
        target_head_sparsity = None; pd = {} ; target_mask_size=None; 
        if self.target_model_info is not None:
            target_head_sparsity = 1 - self.target_model_info.num_attention_heads / self.base_model_info.num_attention_heads
            target_mask_size = self.target_model_info.num_attention_heads
            # pd = {"lambda_1_head": torch.nn.Parameter(torch.tensor(0.01, device=self.device)),
                #   "lambda_2_head": torch.nn.Parameter(torch.tensor(0.05, device=self.device))}
            # self.lambdas.update(pd)
        head_mask = Mask(name="head",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_head_sparsity,
                         target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["head"] = head_mask 

    def initialize_qk_head_dim(self): # only campatible when target model info is available
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_heads, self.base_model_info.dim_per_head]
        num_params_per_mask = 2 * self.base_model_info.hidden_size
        mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.hidden_size] 
        
        target_qk_head_dim_sparsity = None; pd = {} 
        if self.target_model_info is not None:
            target_qk_head_dim_sparsity = 1 - self.target_model_info.hidden_size / self.base_model_info.hidden_size
            # pd = {"lambda_1_qk_head_dim": torch.nn.Parameter(torch.tensor(0.01, device=self.device)),
            #       "lambda_2_qk_head_dim": torch.nn.Parameter(torch.tensor(0.05, device=self.device))}
            # self.lambdas.update(pd)
        qk_head_dim = Mask(name="qk_head_dim",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_qk_head_dim_sparsity,
                         target_mask_size=self.target_model_info.hidden_size,
                         device=self.device)
        self.masks["qk_head_dim"] = qk_head_dim 
          
          
    def initialize_vo_head_dim(self): # only campatible when target model info is available
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_heads, self.base_model_info.dim_per_head]
        num_params_per_mask = 2 * self.base_model_info.hidden_size
        mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.hidden_size] 
        
        target_vo_head_dim_sparsity = None; pd = {} 
        if self.target_model_info is not None:
            target_vo_head_dim_sparsity = 1 - self.target_model_info.hidden_size / self.base_model_info.hidden_size
            # pd = {"lambda_1_vo_head_dim": torch.nn.Parameter(torch.tensor(0.01, device=self.device)),
            #       "lambda_2_vo_head_dim": torch.nn.Parameter(torch.tensor(0.05, device=self.device))}
            # self.lambdas.update(pd)
        vo_head_dim = Mask(name="vo_head_dim",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_vo_head_dim_sparsity,
                         device=self.device)
        self.masks["vo_head_dim"] = vo_head_dim 
        
    def initialize_head_layer(self):
        mask_shape = [self.base_model_info.num_layers]
        num_params_per_mask=self.base_model_info.params_per_head *  self.base_model_info.num_attention_heads
        mask_output_shape = [self.base_model_info.num_layers] 
        
        target_head_layer_sparsity = None; pd = {}; target_mask_size=None; 
        if self.target_model_info is not None:
            target_head_layer_sparsity = 1 - self.target_model_info.num_layers / self.base_model_info.num_layers
            target_mask_size = self.target_model_info.num_layers
            # pd = {"lambda_1_head_layer": torch.nn.Parameter(torch.tensor(0.01, device=self.device)),
            #       "lambda_2_head_layer": torch.nn.Parameter(torch.tensor(0.05, device=self.device))}
            # self.lambdas.update(pd)
        
        head_layer_mask = Mask(name="head_layer",
                              mask_shape=mask_shape,
                               num_params_per_mask=num_params_per_mask,
                               mask_output_shape=mask_output_shape,
                               target_sparsity=target_head_layer_sparsity,
                               target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["head_layer"] = head_layer_mask
        
    def initialize_intermediate(self):
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.intermediate_size]
        num_params_per_mask=self.base_model_info.params_per_intermediate_dim
        mask_output_shape = [self.base_model_info.num_layers, 1, 1, self.base_model_info.intermediate_size] 
        
        target_int_sparsity = None; pd = {}; target_mask_size=None; 
        if self.target_model_info is not None:
            target_int_sparsity = 1 - self.target_model_info.intermediate_size / self.base_model_info.intermediate_size
            target_mask_size = self.target_model_info.intermediate_size
            # pd = {"lambda_1_intermediate": torch.nn.Parameter(torch.tensor(0.01, device=self.device)),
            #       "lambda_2_intermediate": torch.nn.Parameter(torch.tensor(0.05, device=self.device))}
            # self.lambdas.update(pd)
        
        int_mask = Mask(name="intermediate",
                        mask_shape=mask_shape,
                        num_params_per_mask=num_params_per_mask,
                        mask_output_shape=mask_output_shape,
                        target_sparsity=target_int_sparsity,
                        target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["intermediate"] = int_mask
       

    def initialize_mlp(self):
        mask_shape = [self.base_model_info.num_layers]
        num_params_per_mask=self.base_model_info.params_per_mlp_layer
        mask_output_shape = [self.base_model_info.num_layers] 
        
        target_mlp_sparsity = None; pd = {}; target_mask_size=None; 
        if self.target_model_info is not None:
            target_mlp_sparsity = 1 - self.target_model_info.num_layers / self.base_model_info.num_layers
            target_mask_size = self.target_model_info.num_layers
            # pd = {"lambda_1_mlp": torch.nn.Parameter(torch.tensor(0.01, device=self.device)),
            #       "lambda_2_mlp": torch.nn.Parameter(torch.tensor(0.05, device=self.device))}
            # self.lambdas.update(pd)
        
        mlp_mask = Mask(name="mlp",
                        mask_shape=mask_shape,
                        num_params_per_mask=num_params_per_mask,
                        mask_output_shape=mask_output_shape,
                        target_sparsity=target_mlp_sparsity,
                        target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["mlp"] = mlp_mask 

    def initialize_layer(self):
        mask_shape = [self.base_model_info.num_layers]
        num_params_per_mask=self.base_model_info.params_per_head * self.base_model_info.num_attention_heads + self.base_model_info.params_per_mlp_layer
        mask_output_shape = [self.base_model_info.num_layers] 
        
        target_layer_sparsity = None; target_mask_size=None;  pd = {}
        if self.target_model_info is not None:
            target_layer_sparsity = 1 - self.target_model_info.num_layers / self.base_model_info.num_layers
            target_mask_size = self.target_model_info.num_layers
            # pd = {"lambda_1_layer": torch.nn.Parameter(torch.tensor(0.01, device=self.device)),
            #       "lambda_2_layer": torch.nn.Parameter(torch.tensor(0.05, device=self.device))}
            # self.lambdas.update(pd)
        
        layer_mask = Mask(name="layer",
                         mask_shape=mask_shape,
                          num_params_per_mask=num_params_per_mask,
                          mask_output_shape=mask_output_shape,
                          target_sparsity=target_layer_sparsity,
                          target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model) 
        self.masks["layer"] = layer_mask 
    
    def constrain_parameters(self):
        for key in self.masks:
            self.masks[key].constrain_parameters()

    def calculate_expected_score_sparsity(self):
        expected_scores = {}
        expected_sparsitys = {}
        for key in self.masks:
            score, sparsity = self.masks[key].calculate_expected_score_sparsity()
            expected_scores[key] = score
            expected_sparsitys[key] = sparsity
        return expected_scores, expected_sparsitys
    
    def transform_scores_for_head(self, expected_scores: dict):
        head_score = expected_scores["head"] # 12 * 12

        head_layer_score = None
        if "head_layer" in expected_scores:
            head_layer_score = expected_scores["head_layer"]
        elif "layer" in expected_scores:
            head_layer_score = expected_scores["layer"] # 12
        if head_layer_score is not None:
            head_layer_score = head_layer_score.view(-1, 1) # 12 * 1
       
        return head_layer_score, head_score

    def transform_scores_for_mlp(self, expected_scores: dict):
        mlp_score = None
        if "mlp" in expected_scores:
            mlp_score = expected_scores["mlp"] # 12
        elif "layer" in expected_scores:
            mlp_score = expected_scores["layer"] # 12
        if mlp_score is not None:
            mlp_score = mlp_score.unsqueeze(-1)
        
        intermediate_score = expected_scores["intermediate"] # 12 * 3072
        return mlp_score, intermediate_score

    def get_expected_num_params_new(self, expected_scores: dict):
        # All operations are performed only at the scalar/vector/small-matrix level
        num_parameters = expected_scores['hidden'].new_tensor(0.) \
                        if 'hidden' in expected_scores else \
                        next(iter(expected_scores.values())).new_tensor(0.)

        head_layer_score, head_score = self.transform_scores_for_head(expected_scores)  # [L], [L]
        mlp_score, int_score = self.transform_scores_for_mlp(expected_scores)           # [L], [L*I] or [L,I]

        # For now, reduce the head/int scores to per-layer weighted sums
        head_score = head_layer_score * head_score        # [L]
        int_score  = mlp_score * int_score               # [L, I] (same meaning as [L*I])

        L = head_score.shape[0]
        # dim_per_head, hidden_size, etc. are constant scalars
        H = self.base_model_info.hidden_size
        d_h = self.base_model_info.dim_per_head

        if "qk_head_dim" in expected_scores:
            # Treat qk/vo as [L, H] and hidden as [H]
            qk = expected_scores["qk_head_dim"].view(L, -1)   # [L, H]
            vo = expected_scores["vo_head_dim"].view(L, -1)   # [L, H]

            # Instead of repeating head_score along head_dim, reduce it to a sum
            # Original: sum_l,i,j hidden[i]* (head_score[l]*qk[l,j])  == (hidden.sum()) * sum_l (head_score[l]*sum_j qk[l,j])
            # However, qk/vo are meaningful per head_dim → the "per-layer inner product" sum with hidden is more accurate
            hidden = expected_scores["hidden"]                # [H]
            # Sum of per-layer inner products: sum_l dot(hidden, head_score[l]*qk[l])
            qk_contrib = torch.sum((qk * head_score.unsqueeze(1)) @ hidden)   # scalar
            vo_contrib = torch.sum((vo * head_score.unsqueeze(1)) @ hidden)   # scalar
            num_parameters = num_parameters + 2 * qk_contrib + 2 * vo_contrib

            # int_score: originally a 12×768×3072 broadcast matmul → can be reduced to a product of sums
            # sum_{l,i,j} hidden[i] * int_score[l,j]  ==  hidden.sum() * int_score.sum()
            int_total = int_score.sum()
            num_parameters = num_parameters + 3 * hidden.sum() * int_total

        elif "hidden" in expected_scores:
            hidden = expected_scores["hidden"]                # [H]

            # total sum of outer(hidden, head_score.reshape(-1)) = hidden.sum() * head_score.sum()
            num_parameters = num_parameters + (
                hidden.sum() * head_score.sum()
            ) * (self.masks.head.num_params_per_mask / H)

            # total sum of outer(hidden, int_score.reshape(-1)) = hidden.sum() * int_score.sum()
            num_parameters = num_parameters + (
                hidden.sum() * int_score.sum()
            ) * (self.masks.intermediate.num_params_per_mask / H)

        else:
            num_parameters = num_parameters + head_score.sum() * self.masks.head.num_params_per_mask
            num_parameters = num_parameters + int_score.sum()  * self.masks.intermediate.num_params_per_mask

        return num_parameters

    def get_expected_num_params(self, expected_scores: dict): #! calculate the current parsity
        num_parameters = 0
       
        # 12 * 1 
        # 12 * 12
        head_layer_score, head_score = self.transform_scores_for_head(expected_scores)
        mlp_score, int_score = self.transform_scores_for_mlp(expected_scores)
        
        head_score = (head_layer_score * head_score) # 12 * 12
        int_score = (mlp_score * int_score) # 12 * 3072

        qk_score = None
        if "qk_head_dim" in expected_scores:
            qk_head_dim_score = expected_scores["qk_head_dim"] # num_layer * hidden_size
            vo_head_dim_score = expected_scores["vo_head_dim"] # num_layer * hidden_size
            qk_head_dim_score = qk_head_dim_score.view(qk_head_dim_score.shape[0], -1) # 12 * 768
            vo_head_dim_score = vo_head_dim_score.view(vo_head_dim_score.shape[0], -1) # 12 * 768
            head_score = torch.repeat_interleave(head_score, self.base_model_info.dim_per_head, dim=1) # 12 * 768

            qk_score = head_score * qk_head_dim_score # 12 * 768
            vo_score = head_score * vo_head_dim_score # 12 * 768
                
        if "hidden" in expected_scores:
            hidden_score = expected_scores["hidden"] # 768 
            
            if qk_score is None:
                num_parameters += torch.outer(hidden_score, head_score.reshape(-1)).sum() * self.masks.head.num_params_per_mask / self.base_model_info.hidden_size # 768 * 144
                num_parameters += torch.outer(hidden_score, int_score.reshape(-1)).sum() * self.masks.intermediate.num_params_per_mask / self.base_model_info.hidden_size # 768 * 36864
            else:
                num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), qk_score.unsqueeze(1))) * 2 # 12 * 768 * 768
                num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), vo_score.unsqueeze(1))) * 2 # 12 * 768 * 768
                num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), int_score.unsqueeze(1))) * 3 # 12 * 768 * 3072
        else:
            num_parameters += torch.sum(head_score) * self.masks.head.num_params_per_mask
            num_parameters += torch.sum(int_score) * self.masks.intermediate.num_params_per_mask
        return num_parameters
    
    def get_target_sparsity(self, pruned_steps: int, full_sparsity: float = None):
        target_sparsity = full_sparsity
        if getattr(self, "lagrangian_warmup_steps", 0) > 0:
            target_sparsity = (target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup_steps) + self.start_sparsity
        return target_sparsity


    def lagrangian_regularization(self, pruned_steps: int):
        # def _lag_loss(expected_sparsity: torch.tensor, target_sparsity: float, lambda_1: torch.tensor, lambda_2: torch.tensor): # previous
        #     lagrangian_loss = lambda_1 * (expected_sparsity - target_sparsity) + lambda_2 * (expected_sparsity - target_sparsity) ** 2 
        #     lagrangian_loss = lagrangian_loss.mean()
        #     return lagrangian_loss
        def _lag_loss(expected_sparsity: torch.tensor, target_sparsity: float, sub_module_sparsitys,lambda_1: torch.tensor, lambda_2):
            term = ((expected_sparsity / target_sparsity) - 1)
            term = torch.sqrt(torch.pow(term, 2))
            lagrangian_loss = lambda_1 * term
            hidden_sparsity = sub_module_sparsitys['hidden']
            layer_sparsity = sub_module_sparsitys['layer']
            # print("hidden: ", hidden_sparsity, "layer: ", layer_sparsity)
            layer_hidden_term = hidden_sparsity * 4096 / (layer_sparsity * 32 ) - 4096 / 32
            layer_hidden_loss = torch.sqrt(torch.pow(  layer_hidden_term, 2 )) * lambda_2
            total_loss = lagrangian_loss + layer_hidden_loss
            # total_loss = lagrangian_loss
            # lagrangian_loss = lambda_1 * (expected_sparsity - target_sparsity) 
            # print(f"calc loss {lagrangian_loss.mean()}")
            # print("layer_hidden_term: ", layer_hidden_loss.mean())
            print("lambda1: ", lambda_1, " lambda2: ", lambda_2)
            return total_loss.mean()

        # target_sparsity = self.get_target_sparsity(pruned_steps, self.target_sparsity) # previous
        target_sparsity = self.get_target_sparsity(pruned_steps, self.target_sparsity) 
        # target_sparsity = self.target_sparsity          
        target_sparsity = 1 - target_sparsity
        target_sparsity = target_sparsity * self.prunable_model_size
        expected_scores, expected_sparsitys = self.calculate_expected_score_sparsity()
        # expected_size = self.get_expected_num_params(expected_scores) #! calculate \bar s
        expected_size = self.get_expected_num_params_new(expected_scores)
        # expected_sparsity = 1 - expected_size / self.prunable_model_size # previous
        # expected_sparsity = (1 - expected_size / self.prunable_model_size) * self.prunable_model_size
        expected_sparsity =   expected_size / self.prunable_model_size * self.prunable_model_size
        print("sparsity: ", target_sparsity / self.prunable_model_size, ", ", expected_size / self.prunable_model_size)
        return_v = {}
        if self.target_model_info is None:
            # lagrangian_loss = _lag_loss(expected_sparsity, target_sparsity, self.lambdas["lambda_1"], self.lambdas["lambda_2"]) # previous
            lagrangian_loss = _lag_loss(expected_sparsity, target_sparsity, self.lambdas["lambda_1"])
            return_v = {"expected_sparsity": expected_sparsity.item(), "target_sparsity": target_sparsity}
            for key in expected_sparsitys:
                return_v[f"expected_{key}_sparsity"] = expected_sparsitys[key].mean().item()
        else:
            lagrangian_loss = 0
            return_v = {}
            # for pruning_module in self.pruning_modules: # previous
            #     ts = self.get_target_sparsity(pruned_steps, self.masks[pruning_module].target_sparsity)
            #     expected_ts = expected_sparsitys[pruning_module] 
            #     lagrangian_loss += _lag_loss(expected_ts, ts, self.lambdas[f"lambda_1_{pruning_module}"], self.lambdas[f"lambda_2_{pruning_module}"])
            #     expected_ts = expected_ts.mean().item()
            #     return_v.update({"expected_{}_sparsity".format(pruning_module): expected_ts, "target_{}_sparsity".format(pruning_module): ts})
            lagrangian_loss = _lag_loss(expected_sparsity, target_sparsity, expected_sparsitys, self.lambdas["lambda_1"], self.lambdas['lambda_2'])
            expected_sparsity = expected_sparsity / self.prunable_model_size
            target_sparsity = target_sparsity / self.prunable_model_size
            return_v["expected_sparsity"] = expected_sparsity.item()
            return_v["target_sparsity"] = target_sparsity
            for key in expected_sparsitys:
                return_v[f"expected_{key}_sparsity"] = expected_sparsitys[key].mean().item()


        # return_v might not matter
        return lagrangian_loss, return_v
 
    def forward(self, calculate_lagrangian: bool = False, pruned_steps: int = 0, deter=False):
        self.constrain_parameters()
        if calculate_lagrangian:
            return self.lagrangian_regularization(pruned_steps)
        
        zs = {f"{pruning_module}_z": [] for pruning_module in self.pruning_modules}
        
        ##XXX
        #for key in list(zs.keys()):
        #    if key[:-2] not in self.l0_module_cfg.get("target_modules"):
        #        zs.pop(key)

        ##

        if "layer" in self.pruning_modules:
            zs.pop("layer_z")
            zs["mlp_z"] = []
            zs["head_layer_z"] = []
        
        if self.training:
            for pruning_module in self.pruning_modules:
                mask = self.masks[pruning_module]
                z = mask.sample_z()
                zs[f"{pruning_module}_z"] = z
            # torch.save(zs, "zs_tensors.pt")
        else: # removed layerwise! 
            with torch.no_grad():
                for pruning_module in self.pruning_modules:


                    mask = self.masks[pruning_module]
                    z = mask.deterministic_z()

                    zs[f"{pruning_module}_z"] = z
        if "layer_z" in zs:
            zs["mlp_z"] = zs.pop("layer_z")
            zs["head_layer_z"] = zs["mlp_z"]
        ##XXX
        #a_thre = 0.5
        #for key, mask in zs.items():
        #    print(type(mask))
        #    count = (mask > a_thre).sum().item()
        #    print(f"{key}: {count}")
        ##

        return zs 


def test_l0_module():
    from omegaconf import OmegaConf as om 
    cfg = om.load("/path/to/ElasticPruner/elasticpruner/configs/llama/7b.yaml")
    cfg = om.load("/path/to/ElasticPruner/elasticpruner/configs/pythia/410m.yaml")
    cfg.model.l0_module.pruning_modules = ["layer", "head", "intermediate", "hidden"]
    
    l0_module = L0Module(cfg.model, "cpu")
    
    # test run_through
    print("\n***************************** \n run forward pass during training")
    l0_module.train()
    zs = l0_module.forward(calculate_lagrangian=False)
    for key in zs:
        print(key, zs[key].shape)

    print("\n***************************** \n run forward pass during eval")
    l0_module.eval()
    zs = l0_module.forward(calculate_lagrangian=False)
    for key in zs:
        print(key, zs[key].shape)

    print("\n***************************** \n run forward pass during lagrangian")
    l0_module.train()
    loss, v = l0_module(calculate_lagrangian=True, pruned_steps=320)
    print("loss", loss.item())
    for key in v:
        if torch.is_tensor(v[key]): vv = v[key].item()
        else: vv = v[key]
        print(key, vv)
    
    print("\n***************************** \n Test target sparsity") 
    # test target_sparsity
    target_sparsity = l0_module.get_target_sparsity(50, l0_module.target_sparsity)
    print("target sparsity at step 50: ", target_sparsity)
     
    target_sparsity = l0_module.get_target_sparsity(100, l0_module.target_sparsity)
    print("target sparsity at step 100: ", target_sparsity)
    
    target_sparsity = l0_module.get_target_sparsity(200, l0_module.target_sparsity)
    print("target sparsity at step 200: ", target_sparsity)
    import pdb; pdb.set_trace()


  
if __name__ == "__main__":
    test_l0_module()
    
