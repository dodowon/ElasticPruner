# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import math
import os
import time
from typing import Any, Dict, List

import torch
from composer import Callback, Logger, State
from composer.loggers import Logger
from composer.utils import dist
from torch.nn import functional as F
# from composer.algorithms import AMP
# from llmfoundry.utils.builders import build_algorithm
from elasticpruner.utils.greedy_selection import GreedyArchitectureSelector
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from types import SimpleNamespace
import copy

class AirPruningCallback(Callback):
    """
        The interplay of pruning and the main training process is implemented fully based on the callback mechanism.
    """
    def __init__(self, lag_warmup_batch) -> None:
        self.lag_warmup_batch = lag_warmup_batch

             
    def before_train_batch(self, state: State, logger: Logger) -> None:
        step = state.timestamp.batch.value
        l0_module = state.model.model.l0_module
        lambda_param = l0_module.lambdas['lambda_1']
        lambda_param_2 = l0_module.lambdas['lambda_2']
        if step < self.lag_warmup_batch:
            lambda_param.requires_grad = False
            lambda_param_2.requires_grad = False
        else:
            lambda_param.requires_grad = True
            lambda_param_2.requires_grad = True


class DropCallback(Callback):
    def __init__(self) :
        self.eval_losses = []

    def set_next_min_to_zero(self, tensor, excluded_indices):
        # Create a mask of the same shape as the tensor and set True at the positions to exclude
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        for idx in excluded_indices:
            mask[tuple(idx)] = True

        # Find the minimum among the values that are not excluded
        masked_tensor = tensor.masked_fill(mask, float('inf'))
        min_val = masked_tensor.min()
        min_index = (masked_tensor == min_val).nonzero(as_tuple=False)[0]

        # Change those positions to 0
        tensor[tuple(min_index)] = 0

        # Return the indices
        return min_index.tolist()
    def eval_end(self, state,  logger):
        print("test 1100")
        print(f"{dir(state)}")
        print(f"{state.model.model.l0_module}")
        import seaborn
        from copy import deepcopy
        l0_module = state.model.model.l0_module
        step = state.timestamp.batch.value
        self.eval_losses.append(state.eval_metrics["eval"][f"LanguageCrossEntropy"].compute().item())
        print("losses :", self.eval_losses)
        for param in l0_module.parameters():
            param.requires_grad = False
        if getattr(l0_module, "origin_masks", None) is None:
            l0_module.origin_masks = deepcopy(l0_module.masks)
        if getattr(l0_module, "excluded_indices", None) is None:
            l0_module.excluded_indices = {k:[] for k in l0_module.pruning_modules}
        for pruning_module in l0_module.pruning_modules:
            if pruning_module not in l0_module.l0_module_cfg.target_modules:
                continue
            mask = l0_module.masks[pruning_module]
            origin_mask = l0_module.origin_masks[pruning_module]
            if self.check_convergence(self.eval_losses) == 1:
                l0_module.excluded_indices[pruning_module].extend(self.set_next_min_to_zero(mask.z_loga, l0_module.excluded_indices[pruning_module]))
            print("###0011 :", l0_module.excluded_indices)
            print("###0012 :", l0_module.masks[pruning_module].z_loga.min())
            #print(f"callback##001 {mask.z_loga}, {mask}")
            soft_mask = torch.sigmoid(mask.z_loga / mask.temperature * mask.magical_number)
            seaborn.histplot(soft_mask.flatten().cpu().numpy(), bins=100, kde=True)
            plt.title(f"Distribution of {pruning_module} {step}step mask ")
            plt.xlabel("sigmoid value")
            plt.ylabel("Count")
            try:
                outpath = os.path.join("/path/to/output/images", f"sigmoid_module_{pruning_module}_{step}step.png")
            except:
                outpath = f"sigmoid_module_{pruning_module}_{step}step.png"
            plt.savefig(outpath)
            plt.close()
            #print(f"callback##001 {pruning_module}, {(mask.z_loga==0).sum().item()}")
            #print(f"callback##003 {pruning_module} , {(mask==0).sum().item()}")
        try:
            loss_save_path = os.path.join("/path/to/output/images", f"loss_graph.png")
        except:
            loss_save_path = "loss_graph.png"
        self.save_loss_plot(loss_save_path)

    def save_loss_plot(self, save_path="eval_loss_plot.png"):
        plt.figure(figsize=(10, 5))
        plt.plot(self.eval_losses, marker='o')
        plt.title("Evaluation Loss over Time")
        plt.xlabel("Evaluation Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()  # release resources

    def check_convergence(self, losses, slope_threshold=0.01, loss_threshold=0.05, window=5):
        """
        Check whether the loss decreases and converges to 0.
        - losses: list or array of losses
        - slope_threshold: slope flattening criterion (smaller is stricter)
        - loss_threshold: the recent loss must be below this value to count as converged
        - window: how many recent values to average over
        """
        if len(losses) < 2:
            return "Cannot determine due to insufficient data"
    
        losses = np.array(losses)
        epochs = np.arange(len(losses))
        slopes = np.diff(losses) / np.diff(epochs)
    
        # Slopes of the most recent `window` values
        recent_slopes = slopes[-window:] if len(slopes) >= window else slopes
        avg_slope = np.mean(recent_slopes)
    
        is_decreasing = avg_slope < 0  # negative: decreasing
        is_flattening = abs(avg_slope) < slope_threshold  # small slope
        is_near_zero = losses[-1] < loss_threshold  # recent loss is small
    
        if is_decreasing and is_flattening and is_near_zero:
            return 1
        elif not is_decreasing:
            return 0
        else:
            return -1




# (replace with this class)
class DistillationCallback(Callback):
    def __init__(self, 
                 teacher_model_name: str, 
                 distillation_weight: float = 0.5, 
                 temperature: float = 2.0):
        
        self.teacher_model_name = teacher_model_name
        self.distillation_weight = distillation_weight
        self.temperature = temperature
        
        # [Modified] Do not load the model here
        self.teacher_model = None
        self.stream_cuda3 = None
        # self.beta = 7/70000
        self.beta = 7/14000
        
        print(f"DistillationCallback initialized (lazy). Will load {self.teacher_model_name} on first call.", flush=True)

    def _lazy_init_teacher(self):
        """On the first call, load the teacher model and move it to cuda:3"""
        print(f"[DEBUG] LAZY INIT: Loading teacher model '{self.teacher_model_name}'...", flush=True)
        
        try:
            # (Refer to the internal logic of your 'get_hf_model_tokenizer' function for this part)
            # (Hugging Face login, trust_remote_code=True, etc. may be required)
            model = AutoModelForCausalLM.from_pretrained(
                self.teacher_model_name,
                torch_dtype=torch.bfloat16, # (example: use bfloat16 since it is an A6000)
                trust_remote_code=True
            )
            
            print(f"[DEBUG] LAZY INIT: Moving teacher model to cuda:1...", flush=True)
            self.teacher_model = model.to('cuda:1')
            # self.stream_cuda3 = torch.cuda.Stream(device=1)
            print(f"[DEBUG] LAZY INIT: Teacher model ready on cuda:1.", flush=True)
        
        except Exception as e:
            print(f"[FATAL ERROR] Failed to lazy-load teacher model: {e}", flush=True)
            raise e

    def _compute_distill_loss(self, student_logits, teacher_logits):
        # (this function is unchanged)
        student_logits_scaled = student_logits / self.temperature
        teacher_logits_scaled = teacher_logits / self.temperature
        soft_student = F.log_softmax(student_logits_scaled, dim=-1)
        soft_teacher = F.softmax(teacher_logits_scaled, dim=-1)
        loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        # return (self.temperature ** 2) * loss
        return loss

    def after_loss(self, state: State, logger: Logger):
        
        # [Modified] Load the teacher model on the first run
        if self.teacher_model is None:
            self._lazy_init_teacher()
        
        # print(f"[DEBUG] STEP {state.timestamp.batch} / MICRO-BATCH: Entering before_loss...", flush=True)
        
        try:
            # print(state.outputs.keys())
            student_logits = state.outputs['logits']
            original_loss = state.loss
            self.teacher_model.eval()

            # (the logic below is the same as the 'separate stream' version)
            # with torch.no_grad(), torch.cuda.stream(self.stream_cuda3):
            with torch.no_grad():
                inputs_cuda1 = {}
                # print("[DEBUG] ... moving data to cuda:1 (non-blocking)", flush=True)
                for key, val in state.batch.items():
                    if isinstance(val, torch.Tensor):
                        inputs_cuda1[key] = val.to('cuda:1', non_blocking=True)
                inputs_cuda1.pop('labels', None)
                
                # print("[DEBUG] ... data moved. Running teacher model on cuda:1 (on stream)...", flush=True)
                teacher_outputs = self.teacher_model(**inputs_cuda1)
                
                # print("[DEBUG] ... teacher model ran. Moving logits to CPU (non-blocking)...", flush=True)
                teacher_logits_on_cpu = teacher_outputs.logits.to('cpu', non_blocking=True)
            
            # print("[DEBUG] ... explicit sync on cuda:3 stream...", flush=True)
            # self.stream_cuda3.synchronize()
            # print("[DEBUG] ... sync complete. Moving logits to cuda:0...", flush=True)

            teacher_logits = teacher_logits_on_cpu.to('cuda:0')
            
            # print("[DEBUG] ... logits on cuda:0. Calculating loss...", flush=True)

            # ... (the loss computation logic below is unchanged) ...
            if student_logits.shape != teacher_logits.shape:
                # ...
                return
            
            distill_loss = self._compute_distill_loss(student_logits, teacher_logits)
            # print(f"deice: {original_loss.device}, {distill_loss.device}")
            # print(original_loss)
            # combined_loss = (1.0 - self.distillation_weight) * original_loss.to("cuda:0") + \
            #                 (self.distillation_weight) * distill_loss
            # print(f"losses: {state.loss}, {distill_loss}")
            
            state.loss['total'] += self.beta * distill_loss
            # print(state.loss)
            logger.log_metrics({
                # ...
            })
            
            # print("[DEBUG] ... loss calculated. Exiting before_loss.", flush=True)

        except Exception as e:
            print(f"\n\n[FATAL ERROR] in DistillationCallback: {e}\n\n", flush=True)
            raise e
        
class AirAutoresumeCallback(Callback):
    def __init__(self, clean_optimizer_copy: list = None):
        # 2. The constructor receives the deep-copied, clean algorithm list.
        # self.clean_optimizer_copy = clean_optimizer_copy
        self.is_first = True

    def before_backward(self, state, logger):
        model = state.model
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            
  

class AirDebugCallback(Callback):
    # def before_loss(self, state, logger):
    #     for param in state.model.parameters():
    #         # print(param)
    #         if torch.isinf(param.data).any():
    #             print("before !!! INF occurred in parameters !!!")
    #             break
    #         if torch.isnan(param.data).any():
    #             print("before !!! NAN occurred in parameters !!!")
    #             break

    #     for name, param in state.model.model.l0_module.named_parameters():
    #         if param.grad is not None:
    #             print(f"before loss : {name} - Grad Max: {param.grad.max()}, Min: {param.grad.min()}")
    #             # If Grad is in the tens to hundreds, Lambda should be reduced

    def after_loss(self, state, logger):
        pass
        # for param in state.model.parameters():
        #     if torch.isinf(param.grad).any():
        #         print("after !!! INF occurred in parameters !!!")
        #         break
        #     if torch.isnan(param.grad).any():
        #         print("after !!! NAN occurred in parameters !!!")
        #         break


            
        # for name, param in state.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"after loss : {name} - Grad Max: {param.grad.max()}, Min: {param.grad.min()}")
        #         # If Grad is in the tens to hundreds, Lambda should be reduced

    def after_backward(self, state, logger):
        for param in state.model.parameters():
            if param.grad is None:
                continue
            if torch.isinf(param.data).any():
                print("after backward !!! INF occurred in parameters !!!")
                break
            if torch.isnan(param.data).any():
                print("after backward !!! NAN occurred in parameters !!!")
                break
        print("=== Attention Layer Status Check ===")
        found = False
        l0_module = state.model.model.l0_module
        for param in l0_module.parameters():
            if param.grad is None:
                continue
            if torch.isinf(param.grad).any():
                print("after !!! INF occurred in parameters !!!")
                found = True
                break
            if torch.isnan(param.grad).any():
                print("after !!! NAN occurred in parameters !!!")
                found = True
                break
        if found:
            l0_module = state.model.model.l0_module
            head_mask = l0_module.masks['head'].sample_z()
            hidden_mask = l0_module.masks['hidden'].sample_z()
            intermediate_mask = l0_module.masks['intermediate'].sample_z()
            layer_mask = l0_module.masks['layer'].sample_z()
            for layer_idx in range(head_mask.shape[0]):
                dead_head = (head_mask[layer_idx] == 0).sum().item()
                print(f"dead heads: {dead_head}, total: {head_mask[layer_idx].shape}")
                dead_intermediate = (intermediate_mask[layer_idx] == 0).sum().item()
                print(f"dead intermediate: {dead_intermediate}, intermediate: {intermediate_mask[layer_idx]}")
            dead_layer = (layer_mask == 0).sum().item()
            print(f"dead layer: {dead_layer}")
            dead_hidden = (hidden_mask == 0).sum().item()
            print(f"dead hidden: {dead_hidden}")

        

        
        # for name, param in state.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"after bacward : {name} - Grad Max: {param.grad.max()}, Min: {param.grad.min()}")
        #         # If Grad is in the tens to hundreds, Lambda should be reduced


class AirGreedySelectCallback(Callback):
    def __init__(self):
        self.is_first=True
        self.greedy_selector = None

    def change_target_mask_size(self, l0_module, config):
        print("change mask")
        for mask in l0_module.masks.keys():
            if mask == "head":
                l0_module.masks[mask].target_mask_size = config.n_heads
            elif mask == "intermediate":
                l0_module.masks[mask].target_mask_size = config.intermediate_size
            elif mask == "layer":
                l0_module.masks[mask].target_mask_size = config.n_layers
            elif mask == "hidden":
                l0_module.masks[mask].target_mask_size = config.d_model

    def eval_start(self, state, logger):
        l0_module = state.model.model.l0_module
        self.greedy_selector = GreedyArchitectureSelector(l0_module.to("cpu"))
        L, h, i = self.greedy_selector.greedy_select(1.3e9, 12, 1024, 3000, 128, 128)
        last_config = {"n_layers": L, "d_model": h, "n_heads": h//128, "intermediate_size":i, 'vocab_size':32000}
        last_config = SimpleNamespace(**last_config)
        target_module_cfg = l0_module.set_model_info(last_config, 3)
        # origin_target_model_cfg = deepcopy(self.model.model.l0_module.target_model_info)
        l0_module.target_model_info = target_module_cfg
        self.change_target_mask_size(l0_module, last_config)

    # def after_loss(self, state: State, logger: Logger):
    #     # print(dir(state.optimizers))
    #     # print(state.optimizers)
    #     # print(state.optimizers[0])
    #     print("Checking optimizer state for NaNs...")
    #     optimizer = state.optimizers[0]
    #     has_nan_inf = False
    #     try:
    #         # Access the state via state_dict()
    #         state_dict = optimizer.state_dict()
            
    #         # The actual tensors are inside the 'state' key.
    #         for param_id, state_info in state_dict['state'].items():
    #             if 'exp_avg' in state_info:
    #                 exp_avg = state_info['exp_avg']
    #                 if torch.isnan(exp_avg).any() or torch.isinf(exp_avg).any():
    #                     print(f"!!! NaN/Inf found: param_id={param_id}, buffer='exp_avg'")
    #                     has_nan_inf = True
                
    #             if 'exp_avg_sq' in state_info:
    #                 exp_avg_sq = state_info['exp_avg_sq']
    #                 if torch.isnan(exp_avg_sq).any() or torch.isinf(exp_avg_sq).any():
    #                     print(f"!!! NaN/Inf found: param_id={param_id}, buffer='exp_avg_sq'")
    #                     has_nan_inf = True

    #         if not has_nan_inf:
    #             print(">>> Optimizer state: OK (no NaN/Inf)")
    #         # else:
    #         #     # Stop training if NaN is found
    #         #     raise RuntimeError("Optimizer state corrupted with NaN/Inf.")

    #     except Exception as e:
    #         print(f"Error while checking optimizer state: {e}")
    #         print("Note: the state_dict structure may differ. Check the state_dict['state'] key.")

class ModelFreezeCallback(Callback):
    def __init__(self, freeze_until: int = 0, unfreeze_at: int = None):
        self.freeze_until = freeze_until  # freeze until this step
        self.unfreeze_at = unfreeze_at    # unfreeze at this step
        self.frozen = False
    def fit_start(self, state, logger):
        if self.freeze_until > 0:
            self._freeze_model(state)
    def after_train_batch(self, state, logger):
        step = state.timestamp.batch.value
        if self.frozen and self.unfreeze_at and step >= self.unfreeze_at:
            self._unfreeze_model(state)
    def _freeze_model(self, state):
        for name, param in state.model.named_parameters():
            if 'l0_module' not in name:
                print(f"[ModelFreezeCallback] Freezing {name}")
                param.requires_grad = False
        self.frozen = True
        print(f"[ModelFreezeCallback] Model frozen (l0_module only training)")
    def _unfreeze_model(self, state):
        for name, param in state.model.named_parameters():
            param.requires_grad = True
        self.frozen = False
        print(f"[ModelFreezeCallback] Model unfrozen (all params training)")


class AdaptiveLayerSparsityCallback(Callback):
    """
    At every eval_end:
        1) If there was a previous attempt, compare PPL → keep/rollback
        2) Increase sparsity for the next training interval
    → The mask is adjusted to the new sparsity during training
    → Judged again at the next eval_end
    
    Termination conditions:
        - Stop changing sparsity once max_step is exceeded
        - If consecutive rollbacks reach max_rollback or more, early stop (stop changing)
    """
    def __init__(self, 
                 start_step: int = 500,
                 max_step: int = 2000,
                 sparsity_delta: float = 0.05,
                 max_layer_sparsity: float = 0.5,
                 prune_k: int = 2,
                 max_rollback: int = 3):
        self.start_step = start_step
        self.max_step = max_step
        self.sparsity_delta = sparsity_delta
        self.max_layer_sparsity = max_layer_sparsity
        self.max_rollback = max_rollback
        self.prune_k = prune_k
        self.consecutive_prune_k = 0

        self.prev_ppl = None
        self.prev_target_sparsity = None
        self.prev_mask_state = None
        self.trial_active = False
        self.consecutive_rollbacks = 0
        self.stopped = False  # True on early stop or when max_step is reached

    def eval_end(self, state, logger):
        step = state.timestamp.batch.value
        l0_module = state.model.model.l0_module
        if l0_module is None or step < self.start_step:
            return

        if 'layer' not in l0_module.masks:
            return
        layer_mask = l0_module.masks['layer']
        if layer_mask.target_sparsity is None:
            return

        # ── 1) Compute PPL ──
        current_ppl = self._get_ppl(state)
        if current_ppl is None:
            print("[AdaptiveLayerSparsity] Could not obtain PPL")
            return

        # ── 2) If there was a previous attempt, judge it ──
        if self.trial_active:
            if self.prev_ppl is not None and current_ppl <= self.prev_ppl:
                # ✅ Improved → keep
                print(f"[AdaptiveLayerSparsity] step={step}: "
                      f"PPL {self.prev_ppl:.4f} → {current_ppl:.4f} ✅ keep "
                      f"(layer_sparsity={layer_mask.target_sparsity:.4f})")
                self.prev_mask_state = None
                self.consecutive_rollbacks = 0  # reset the consecutive rollback count
            else:
                # ❌ Worsened → rollback
                self.consecutive_rollbacks += 1
                print(f"[AdaptiveLayerSparsity] step={step}: "
                      f"PPL {self.prev_ppl} → {current_ppl:.4f} ❌ rollback "
                      f"(consecutive {self.consecutive_rollbacks}/{self.max_rollback} times)")
                self._rollback(l0_module)
                self.consecutive_prune_k -= self.prune_k
                
                # early stop check
                if self.consecutive_rollbacks >= self.max_rollback:
                    self.stopped = True
                    print(f"[AdaptiveLayerSparsity] step={step}: "
                          f"{self.max_rollback} consecutive rollbacks → early stop! "
                          f"final layer_sparsity={layer_mask.target_sparsity:.4f}")
                    self.prev_ppl = current_ppl
                    self.trial_active = False
                    return

        self.prev_ppl = current_ppl
        self.trial_active = False

        # ── 3) Check termination conditions ──
        if self.stopped:
            return
        if step >= self.max_step:
            self.stopped = True
            print(f"[AdaptiveLayerSparsity] step={step}: "
                  f"max_step({self.max_step}) reached → stop changing. "
                  f"final layer_sparsity={layer_mask.target_sparsity:.4f}")
            return

        # ── 4) Next attempt: increase sparsity ──
        current_target = layer_mask.target_sparsity
        new_target = min(current_target + self.sparsity_delta, self.max_layer_sparsity)
        # l0_module.lambdas['lambda_1_layer'].data.fill_(0.01)  # immediate pressure
        # l0_module.lambdas['lambda_2_layer'].data.fill_(0.01)
        self.consecutive_prune_k += self.prune_k
        self._prune_layer(l0_module, self.consecutive_prune_k)
        if new_target == current_target:
            self.stopped = True
            print(f"[AdaptiveLayerSparsity] step={step}: already at maximum, stopping attempts")
            return

        # Back up the current state
        self.prev_target_sparsity = current_target
        self.prev_mask_state = {
            'z_loga': layer_mask.z_loga.data.clone(),
            'target_mask_size': layer_mask.target_mask_size,
        }

        # Increase sparsity
        layer_mask.target_sparsity = new_target
        layer_mask.target_mask_size = int(layer_mask.mask_size * (1 - new_target))
        
        self.trial_active = True
        print(f"[AdaptiveLayerSparsity] step={step}: "
              f"layer_sparsity {current_target:.4f} → {new_target:.4f} (trying for the next training interval)")

    def _get_ppl(self, state):
        """Extract perplexity from the eval metrics"""
        for evaluator_name, metrics in state.eval_metrics.items():
            for metric_name, metric in metrics.items():
                if 'loss' in metric_name.lower():
                    val = metric.compute()
                    return torch.exp(val).item()
                elif 'perplexity' in metric_name.lower():
                    return metric.compute().item()
        return None

    def _rollback(self, l0_module):
        layer_mask = l0_module.masks['layer']
        if self.prev_target_sparsity is not None:
            layer_mask.target_sparsity = self.prev_target_sparsity
        if self.prev_mask_state is not None:
            layer_mask.z_loga.data.copy_(self.prev_mask_state['z_loga'])
            layer_mask.target_mask_size = self.prev_mask_state['target_mask_size']
        self.prev_mask_state = None
        self.prev_target_sparsity = None
        layer_mask.z_loga.requires_grad = True
        print(f"[AdaptiveLayerSparsity] Rollback complete: "
              f"layer_sparsity={layer_mask.target_sparsity:.4f}")

    def _prune_layer(self, l0_module, prune_k):
        # Force z_loga down starting from the weakest layer
        layer_mask = l0_module.masks['layer']
        with torch.no_grad():
            # Lower slightly overall
            layer_mask.z_loga.data -= 1.0
            # Or set the lowest k directly to -10 (effectively removing them)
            _, indices = torch.topk(layer_mask.z_loga.data, k=prune_k, largest=False)
            layer_mask.z_loga.data[indices] = -10.0
            layer_mask.z_loga.requires_grad = False


class AirNanDiagnosisCallback(Callback):
    """Diagnostic callback for tracking where NaN/Inf first appears.

    Logged to metric/train/* at every batch (after_backward):
        - lambda values and absolute grad values (tracks blow-up of lambdas/lambda_2, lambda_3)
        - each mask's z_loga: max/min/mean/std, ±4.6 saturation ratio, grad max/norm
        - model weights: grad max abs (excluding l0_module)
        - NaN/Inf flags: where it blows up *first*
          * z_loga (data, grad) per mask
          * lambda (data, grad)
          * model weight (data, grad)
          * loss
        - AMP grad scaler (when available)

    How to verify: in wandb, check which nan/* flag becomes 1 at the step where NaN occurs.
        - lambda_grad/data becomes 1 first → λ accumulation blow-up → mixed precision overflow
        - z_loga_grad becomes 1 first → lag loss gradient overflow
        - model_grad/weight becomes 1 first → the model itself breaks on the CE loss path
    """

    CLAMP_MAX = math.log(1e2)        # +4.605
    CLAMP_MIN = math.log(1e-2)       # -4.605
    SAT_TOL = 1e-3                   # tolerance for saturation detection

    def __init__(self, log_interval: int = 1):
        self.log_interval = max(1, int(log_interval))

    def _should_log(self, state: State) -> bool:
        if self.log_interval <= 1:
            return True
        return state.timestamp.batch.value % self.log_interval == 0

    @staticmethod
    def _has_nan_inf(t):
        if t is None:
            return False, False
        with torch.no_grad():
            return (
                bool(torch.isnan(t).any().item()),
                bool(torch.isinf(t).any().item()),
            )

    def _log_lambdas(self, l0_module, metrics: Dict[str, float]) -> None:
        any_data_nan = any_data_inf = False
        any_grad_nan = any_grad_inf = False
        max_abs_val = 0.0
        max_abs_grad = 0.0
        for name, param in l0_module.lambdas.items():
            with torch.no_grad():
                val = param.detach().float().mean().item()
            metrics[f"metric/train/lambda/{name}"] = val
            metrics[f"metric/train/lambda/{name}_abs"] = abs(val)
            if abs(val) > max_abs_val:
                max_abs_val = abs(val)

            d_nan, d_inf = self._has_nan_inf(param.data)
            any_data_nan |= d_nan
            any_data_inf |= d_inf

            if param.grad is not None:
                with torch.no_grad():
                    g_abs_max = param.grad.detach().float().abs().max().item()
                metrics[f"metric/train/lambda/{name}_grad_abs"] = g_abs_max
                if g_abs_max > max_abs_grad:
                    max_abs_grad = g_abs_max
                g_nan, g_inf = self._has_nan_inf(param.grad)
                any_grad_nan |= g_nan
                any_grad_inf |= g_inf

        metrics["metric/train/lambda/_max_abs"] = max_abs_val
        metrics["metric/train/lambda/_max_abs_grad"] = max_abs_grad
        metrics["metric/train/nan/lambda_data"] = float(any_data_nan)
        metrics["metric/train/nan/lambda_grad"] = float(any_grad_nan)
        metrics["metric/train/inf/lambda_data"] = float(any_data_inf)
        metrics["metric/train/inf/lambda_grad"] = float(any_grad_inf)

    def _log_z_loga(self, l0_module, metrics: Dict[str, float]) -> None:
        any_z_nan = any_z_inf = False
        any_zg_nan = any_zg_inf = False
        for mask_name, mask in l0_module.masks.items():
            z = mask.z_loga.data
            with torch.no_grad():
                z_f = z.detach().float()
                metrics[f"metric/train/z_loga/{mask_name}/max"] = z_f.max().item()
                metrics[f"metric/train/z_loga/{mask_name}/min"] = z_f.min().item()
                metrics[f"metric/train/z_loga/{mask_name}/mean"] = z_f.mean().item()
                metrics[f"metric/train/z_loga/{mask_name}/std"] = z_f.std().item()
                metrics[f"metric/train/z_loga/{mask_name}/pos_sat_ratio"] = (
                    (z_f >= self.CLAMP_MAX - self.SAT_TOL).float().mean().item()
                )
                metrics[f"metric/train/z_loga/{mask_name}/neg_sat_ratio"] = (
                    (z_f <= self.CLAMP_MIN + self.SAT_TOL).float().mean().item()
                )

            d_nan, d_inf = self._has_nan_inf(z)
            metrics[f"metric/train/nan/z_loga_{mask_name}_data"] = float(d_nan)
            metrics[f"metric/train/inf/z_loga_{mask_name}_data"] = float(d_inf)
            any_z_nan |= d_nan
            any_z_inf |= d_inf

            grad = mask.z_loga.grad
            if grad is not None:
                with torch.no_grad():
                    g_f = grad.detach().float()
                    metrics[f"metric/train/z_loga/{mask_name}/grad_norm"] = g_f.norm().item()
                    metrics[f"metric/train/z_loga/{mask_name}/grad_max_abs"] = g_f.abs().max().item()
                g_nan, g_inf = self._has_nan_inf(grad)
                metrics[f"metric/train/nan/z_loga_{mask_name}_grad"] = float(g_nan)
                metrics[f"metric/train/inf/z_loga_{mask_name}_grad"] = float(g_inf)
                any_zg_nan |= g_nan
                any_zg_inf |= g_inf

        metrics["metric/train/nan/z_loga_data_any"] = float(any_z_nan)
        metrics["metric/train/inf/z_loga_data_any"] = float(any_z_inf)
        metrics["metric/train/nan/z_loga_grad_any"] = float(any_zg_nan)
        metrics["metric/train/inf/z_loga_grad_any"] = float(any_zg_inf)

    def _log_model(self, state: State, metrics: Dict[str, float]) -> None:
        m_w_nan = m_w_inf = False
        m_g_nan = m_g_inf = False
        max_abs_grad = 0.0
        with torch.no_grad():
            for name, param in state.model.model.named_parameters():
                if "l0_module" in name:
                    continue
                if not (m_w_nan and m_w_inf):
                    w_nan, w_inf = self._has_nan_inf(param.data)
                    m_w_nan |= w_nan
                    m_w_inf |= w_inf
                if param.grad is None:
                    continue
                if not (m_g_nan and m_g_inf):
                    g_nan, g_inf = self._has_nan_inf(param.grad)
                    m_g_nan |= g_nan
                    m_g_inf |= g_inf
                if not (m_g_nan or m_g_inf):
                    v = param.grad.detach().float().abs().max().item()
                    if v > max_abs_grad:
                        max_abs_grad = v
        metrics["metric/train/model/grad_max_abs"] = max_abs_grad
        metrics["metric/train/nan/model_weight"] = float(m_w_nan)
        metrics["metric/train/inf/model_weight"] = float(m_w_inf)
        metrics["metric/train/nan/model_grad"] = float(m_g_nan)
        metrics["metric/train/inf/model_grad"] = float(m_g_inf)

    def _log_loss(self, state: State, metrics: Dict[str, float]) -> None:
        loss = getattr(state, "loss", None)
        if loss is None:
            return
        if isinstance(loss, dict):
            for k, v in loss.items():
                if not torch.is_tensor(v):
                    continue
                n, i = self._has_nan_inf(v)
                metrics[f"metric/train/nan/loss_{k}"] = float(n)
                metrics[f"metric/train/inf/loss_{k}"] = float(i)
                with torch.no_grad():
                    metrics[f"metric/train/loss/{k}"] = v.detach().float().mean().item()
        elif torch.is_tensor(loss):
            n, i = self._has_nan_inf(loss)
            metrics["metric/train/nan/loss_total"] = float(n)
            metrics["metric/train/inf/loss_total"] = float(i)
            with torch.no_grad():
                metrics["metric/train/loss/total"] = loss.detach().float().mean().item()

    def _log_grad_scaler(self, state: State, metrics: Dict[str, float]) -> None:
        scaler = getattr(state, "scaler", None)
        if scaler is None:
            return
        try:
            metrics["metric/train/grad_scaler/scale"] = float(scaler.get_scale())
        except Exception:
            pass

    def after_backward(self, state: State, logger: Logger) -> None:
        if not self._should_log(state):
            return
        l0_module = getattr(state.model.model, "l0_module", None)
        if l0_module is None:
            return

        metrics: Dict[str, float] = {}
        try:
            self._log_lambdas(l0_module, metrics)
            self._log_z_loga(l0_module, metrics)
            self._log_model(state, metrics)
            self._log_loss(state, metrics)
            self._log_grad_scaler(state, metrics)
        except Exception as e:
            print(f"[AirNanDiagnosisCallback] logging error: {e}", flush=True)
            return

        logger.log_metrics(metrics)
