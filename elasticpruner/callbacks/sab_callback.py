# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import torch
import numpy as np
import torch.distributed as torch_dist
from composer import Callback, Logger, State
from typing import List, Dict, Any
from composer.loggers import Logger

class SAB2DynamicCallback(Callback):
    """
    SAB2-inspired Dynamic Loading Callback with Uniform Probing.
    
    Process at Eval Start:
    1. Set Dataset proportions to Uniform (1/N) to ensure all domains are sampled.
    2. Sample 'num_check_batches' to measure current PPL (Uncertainty).
    3. Calculate new weights based on PPL and Improvement history (SAB2 Logic).
    4. Update Dataset with the new calculated proportions for the next epoch.
    """
    def __init__(self, 
                 set_names: List[str],
                 num_check_batches: int = 50,
                 patience: int = 3,
                 decay_factor: float = 0.8,
                 epsilon: float = 0.01
                 ) -> None:
        self.set_names = set_names
        self.n_domains = len(set_names)
        self.num_check_batches = num_check_batches
        
        # SAB2 Parameters
        self.patience = patience
        self.decay_factor = decay_factor
        self.epsilon = epsilon
        
        # State tracking
        self.ppl_history = {domain: [] for domain in set_names}
        self.improvement_factors = {domain: 1.0 for domain in set_names}
        self.stagnation_counter = {domain: 0 for domain in set_names}
        self.used_domain_ids = [[] for _ in range(self.n_domains)]

    def _compute_train_ppl_subset(self, state: State):
        """
        Samples a subset from the training dataloader to compute current PPL per domain.
        Assumes the dataset is currently in 'Uniform Mode'.
        """
        model = state.model
        original_training_mode = model.training
        model.eval()
        
        # Track per-domain loss sums and counts separately (avoids the all_gather size issue)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        domain_loss_sum = torch.zeros(self.n_domains, device=device)
        domain_loss_count = torch.zeros(self.n_domains, device=device, dtype=torch.long)
        
        # Create a FRESH iterator to ensure it picks up the Uniform proportion update
        iterator = iter(state.train_dataloader)
        
        with torch.no_grad():
            for _ in range(self.num_check_batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                
                # Device handling
                batch = state.device.batch_to_device(batch)
                
                # Forward pass
                outputs = model(batch)
                
                if 'set' in batch:
                    domain_ids = batch['set'].view(-1)  # keep as GPU tensor

                    # Compute per-sample loss manually for accuracy
                    logits = outputs['logits']
                    labels = batch['labels']
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    token_losses = token_losses.view(shift_labels.size())
                    sample_losses = token_losses.mean(dim=1).float()  # [batch_size], GPU tensor
                    
                    # Sum per domain (processed on GPU)
                    for d_id in range(self.n_domains):
                        mask = (domain_ids == d_id)
                        if mask.any():
                            domain_loss_sum[d_id] += sample_losses[mask].sum()
                            domain_loss_count[d_id] += mask.sum()
                            
        # Sum across all ranks in distributed training (safe since it is a fixed-size tensor)
        if torch_dist.is_initialized():
            torch_dist.all_reduce(domain_loss_sum, op=torch_dist.ReduceOp.SUM)
            torch_dist.all_reduce(domain_loss_count, op=torch_dist.ReduceOp.SUM)
        
        # Compute PPL
        avg_ppls = {}
        for d_id in range(self.n_domains):
            count = domain_loss_count[d_id].item()
            if count > 0:
                avg_loss = (domain_loss_sum[d_id] / count).item()
                avg_ppls[self.set_names[d_id]] = np.exp(avg_loss)
            else:
                # If the domain was not sampled, use the previous value or a penalty
                last_ppl = self.ppl_history[self.set_names[d_id]][-1] if self.ppl_history[self.set_names[d_id]] else 10000.0
                avg_ppls[self.set_names[d_id]] = last_ppl
                
        model.train(original_training_mode)
        return avg_ppls

    def _calculate_sab_weights(self, current_ppls: Dict[str, float]):
        """ Calculates new weights based on SAB2 logic (Uncertainty * Decay) """
        raw_scores = []
        
        for domain in self.set_names:
            ppl = current_ppls[domain]
            history = self.ppl_history[domain]
            
            # --- 1. Improvement Check (Pruning) ---
            if len(history) > 0:
                prev_ppl = history[-1]
                improvement = prev_ppl - ppl
                
                if improvement <= 0: # Stagnant or Worsened
                    self.stagnation_counter[domain] += 1
                else:
                    self.stagnation_counter[domain] = 0
                
                # Apply Decay if stagnant too long
                if self.stagnation_counter[domain] > self.patience:
                    self.improvement_factors[domain] *= self.decay_factor
            
            # --- 2. Score Calculation ---
            # Score = Uncertainty (PPL) * Efficiency (Improvement Factor)
            score = ppl * self.improvement_factors[domain]
            raw_scores.append(score)
            
            self.ppl_history[domain].append(ppl)
            
        # --- 3. Normalization ---
        total_score = sum(raw_scores)
        if total_score == 0:
            return [1.0 / self.n_domains] * self.n_domains
            
        new_weights = np.array([score / total_score for score in raw_scores])
        
        # --- 4. Safety Floor (Gini/Epsilon) ---
        new_weights = np.maximum(new_weights, self.epsilon)
        new_weights = new_weights / new_weights.sum() # Renormalize
        
        return new_weights.tolist()

    def after_train_batch(self, state: State, logger: Logger) -> None:
        """ Track used_domain_ids (checkpoint-saving compatibility) + log weights """
        idx = state.batch["idx"]
        sets = state.batch["set"]
        
        local_rank = torch_dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        idx = idx.to(device)
        sets = sets.to(device)
        
        world_size = torch_dist.get_world_size()
        gathered_idx = [torch.zeros_like(idx) for _ in range(world_size)]
        gathered_sets = [torch.zeros_like(sets) for _ in range(world_size)]
        torch_dist.all_gather(gathered_idx, idx)
        torch_dist.all_gather(gathered_sets, sets)
        torch_dist.barrier()
        
        all_idx = torch.cat(gathered_idx)
        all_sets = torch.cat(gathered_sets)
        
        for i in range(self.n_domains):
            mask = all_sets == i
            domain_idx = all_idx[mask]
            self.used_domain_ids[i].extend(domain_idx.cpu().tolist())

        # Log weights
        prop = state.train_dataloader.dataset.proportion
        for domain in self.set_names:
            logger.log_metrics({f'metrics/train/{domain}_weight': round(prop[self.set_names.index(domain)], 4)})

    def eval_start(self, state: State, logger: Logger) -> None:
        """
        Orchestrates the Dynamic Weight Update:
        1. Backup current weights (optional, implied we overwrite anyway).
        2. Set Uniform Weights -> Ensures we can sample every domain to check PPL.
        3. Measure PPL.
        4. Calculate & Set New SAB2 Weights.
        """
        dataset = state.train_dataloader.dataset
        if not hasattr(dataset, "update_proportion"):
            print("[SAB2] Warning: Dataset does not have 'update_proportion'. Callback skipped.")
            return

        print("\n[SAB2] 1. Setting Uniform Proportions for PPL Probing...")
        uniform_props = [1.0 / self.n_domains] * self.n_domains
        dataset.update_proportion(uniform_props)
        
        print(f"[SAB2] 2. Measuring PPL on {self.num_check_batches} batches...")
        current_ppls = self._compute_train_ppl_subset(state)
        print(f"[SAB2] Measured PPLs: {current_ppls}")
        
        print("[SAB2] 3. Calculating New Domain Weights...")
        new_props = self._calculate_sab_weights(current_ppls)
        
        print(f"[SAB2] 4. Updating Dataset with New Weights: {dict(zip(self.set_names, np.round(new_props, 4)))}")
        dataset.update_proportion(new_props)
        
        # Logging
        for i, domain in enumerate(self.set_names):
            logger.log_metrics({
                f'sab2/weight/{domain}': new_props[i],
                f'sab2/ppl/{domain}': current_ppls[domain],
                f'sab2/decay_factor/{domain}': self.improvement_factors[domain]
            })

    def state_dict(self) -> Dict[str, Any]:
        return {
            "used_domain_ids": self.used_domain_ids,
            "ppl_history": self.ppl_history,
            "improvement_factors": self.improvement_factors,
            "stagnation_counter": self.stagnation_counter
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.used_domain_ids = state_dict.get("used_domain_ids", self.used_domain_ids)
        self.ppl_history = state_dict.get("ppl_history", self.ppl_history)
        self.improvement_factors = state_dict.get("improvement_factors", self.improvement_factors)
        self.stagnation_counter = state_dict.get("stagnation_counter", self.stagnation_counter)


class SAB2EvalDynamicCallback(Callback):
    """
    SAB2 Dynamic Loading Callback (eval-based).
    
    Instead of creating a separate iterator from the train dataloader,
    it reuses the eval_metrics that Composer has already computed at eval_end
    to measure per-domain PPL and update weights with the SAB2 logic.
    """
    def __init__(self, 
                 set_names: List[str],
                 patience: int = 3,
                 decay_factor: float = 0.8,
                 epsilon: float = 0.01
                 ) -> None:
        self.set_names = set_names
        self.n_domains = len(set_names)
        
        # SAB2 Parameters
        self.patience = patience
        self.decay_factor = decay_factor
        self.epsilon = epsilon
        
        # State tracking
        self.ppl_history = {domain: [] for domain in set_names}
        self.improvement_factors = {domain: 1.0 for domain in set_names}
        self.stagnation_counter = {domain: 0 for domain in set_names}
        self.used_domain_ids = [[] for _ in range(self.n_domains)]
        
        print("====called SAB2EvalDynamicCallback====")

    def _calculate_sab_weights(self, current_ppls: Dict[str, float]):
        """ Calculates new weights based on SAB2 logic (Uncertainty * Decay) """
        raw_scores = []
        
        for domain in self.set_names:
            ppl = current_ppls[domain]
            history = self.ppl_history[domain]
            
            # --- 1. Improvement Check (Pruning) ---
            if len(history) > 0:
                prev_ppl = history[-1]
                improvement = prev_ppl - ppl
                
                if improvement <= 0: # Stagnant or Worsened
                    self.stagnation_counter[domain] += 1
                else:
                    self.stagnation_counter[domain] = 0
                
                # Apply Decay if stagnant too long
                if self.stagnation_counter[domain] > self.patience:
                    self.improvement_factors[domain] *= self.decay_factor
            
            # --- 2. Score Calculation ---
            # Score = Uncertainty (PPL) * Efficiency (Improvement Factor)
            score = ppl * self.improvement_factors[domain]
            raw_scores.append(score)
            
            self.ppl_history[domain].append(ppl)
            
        # --- 3. Normalization ---
        total_score = sum(raw_scores)
        if total_score == 0:
            return [1.0 / self.n_domains] * self.n_domains
            
        new_weights = np.array([score / total_score for score in raw_scores])
        
        # --- 4. Safety Floor (Gini/Epsilon) ---
        new_weights = np.maximum(new_weights, self.epsilon)
        new_weights = new_weights / new_weights.sum() # Renormalize
        
        return new_weights.tolist()

    def after_train_batch(self, state: State, logger: Logger) -> None:
        """ Track used_domain_ids (checkpoint-saving compatibility) + log weights """
        idx = state.batch["idx"]
        sets = state.batch["set"]
        
        local_rank = torch_dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        idx = idx.to(device)
        sets = sets.to(device)
        
        world_size = torch_dist.get_world_size()
        gathered_idx = [torch.zeros_like(idx) for _ in range(world_size)]
        gathered_sets = [torch.zeros_like(sets) for _ in range(world_size)]
        torch_dist.all_gather(gathered_idx, idx)
        torch_dist.all_gather(gathered_sets, sets)
        torch_dist.barrier()
        
        all_idx = torch.cat(gathered_idx)
        all_sets = torch.cat(gathered_sets)
        
        for i in range(self.n_domains):
            mask = all_sets == i
            domain_idx = all_idx[mask]
            self.used_domain_ids[i].extend(domain_idx.cpu().tolist())

        # Log weights
        prop = state.train_dataloader.dataset.proportion
        for domain in self.set_names:
            logger.log_metrics({f'metrics/train/{domain}_weight': round(prop[self.set_names.index(domain)], 4)})

    def eval_end(self, state: State, logger: Logger) -> None:
        """
        Fetch per-domain CE Loss from the eval_metrics that Composer computed at eval_end,
        convert it to PPL, and update the SAB2 weights.
        """
        dataset = state.train_dataloader.dataset
        if not hasattr(dataset, "update_proportion"):
            print("[SAB2-Eval] Warning: Dataset does not have 'update_proportion'. Callback skipped.")
            return

        # 1. Fetch per-domain Loss from eval_metrics
        current_ppls = {}
        print("[SAB2-Eval] loss keys: ", state.eval_metrics['eval'].keys())
        for domain in self.set_names:
            metric_key = f"{domain}_LanguageCrossEntropy"
            if metric_key in state.eval_metrics['eval']:
                loss_val = state.eval_metrics['eval'][metric_key].compute().item()
                current_ppls[domain] = np.exp(loss_val)
            else:
                # If there is no metric, use the previous value
                last_ppl = self.ppl_history[domain][-1] if self.ppl_history[domain] else 10000.0
                current_ppls[domain] = last_ppl
                print(f"[SAB2-Eval] Warning: metric '{metric_key}' not found, using fallback PPL={current_ppls[domain]:.2f}")
        
        print(f"[SAB2-Eval] Measured PPLs: {current_ppls}")
        
        # 2. Compute SAB2 weights
        new_props = self._calculate_sab_weights(current_ppls)
        
        # 3. Update the Dataset
        print(f"[SAB2-Eval] Updating Dataset with New Weights: {dict(zip(self.set_names, np.round(new_props, 4)))}")
        dataset.update_proportion(new_props)
        
        # 4. Logging
        for i, domain in enumerate(self.set_names):
            logger.log_metrics({
                f'sab2/weight/{domain}': new_props[i],
                f'sab2/ppl/{domain}': current_ppls[domain],
                f'sab2/decay_factor/{domain}': self.improvement_factors[domain]
            })

    def state_dict(self) -> Dict[str, Any]:
        return {
            "used_domain_ids": self.used_domain_ids,
            "ppl_history": self.ppl_history,
            "improvement_factors": self.improvement_factors,
            "stagnation_counter": self.stagnation_counter
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.used_domain_ids = state_dict.get("used_domain_ids", self.used_domain_ids)
        self.ppl_history = state_dict.get("ppl_history", self.ppl_history)
        self.improvement_factors = state_dict.get("improvement_factors", self.improvement_factors)
        self.stagnation_counter = state_dict.get("stagnation_counter", self.stagnation_counter)