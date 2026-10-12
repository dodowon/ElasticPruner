# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import numpy as np
from collections import deque
from typing import List, Dict, Any
import torch
import torch.distributed as torch_dist
from composer.core import Callback, State
from composer.loggers import Logger

class EmaZScoreDynamicCallback(Callback):
    """
    Dynamic domain allocation callback using PPL EMA and the standard normal distribution (Z-score).
    Following the plan, it uses the most recent k EMA values to extract only the domains where training is actually effective (not noise) and allocates ratios to them.
    """
    def __init__(self, 
                 set_names: List[str],
                 k: int = 300,             # number of most recent values to keep in the EMA list
                 ema_alpha: float = 0.2,   # weight of the current value in the EMA computation
                 z_threshold: float = -1.0,# Z-score threshold (negative recommended to find domains whose EMA is decreasing)
                 z_condition: str = 'le',  # choose among 'le' (<=), 'ge' (>=), 'abs_ge' (|z| >=)
                 ) -> None:
        self.set_names = set_names
        self.n_domains = len(set_names)
        
        # Algorithm parameters
        self.k = k
        self.ema_alpha = ema_alpha
        self.z_threshold = z_threshold
        self.z_condition = z_condition
        
        # State tracking
        # Setting maxlen to k with a deque automatically discards old data
        self.ema_history = {domain: deque(maxlen=k) for domain in set_names}
        self.used_domain_ids = [[] for _ in range(self.n_domains)]
        
        print(f"====called EmaZScoreDynamicCallback (k={k}, z_threshold={z_threshold})====")

    # ---------------------------------------------------------
    # Detailed feature breakdown section
    # ---------------------------------------------------------
    def _compute_single_ema(self, domain: str, current_ppl: float) -> float:
        """Computes the new EMA value for a single domain."""
        history = self.ema_history[domain]
        if len(history) == 0:
            return current_ppl # use the first value as-is
        
        last_ema = history[-1]
        new_ema = (self.ema_alpha * current_ppl) + ((1 - self.ema_alpha) * last_ema)
        return new_ema

    def _update_ema_history(self, current_ppls: Dict[str, float]) -> None:
        """Takes each domain's current PPL, computes the EMA, and appends it to the list."""
        for domain in self.set_names:
            new_ema = self._compute_single_ema(domain, current_ppls[domain])
            self.ema_history[domain].append(new_ema)

    def _calculate_z_scores(self) -> Dict[str, float]:
        """Computes the mean and standard deviation of each domain's EMA history, and the Z-score of the latest value."""
        z_scores = {}
        for domain in self.set_names:
            history = list(self.ema_history[domain])
            if len(history) < 2:
                # Standard deviation cannot be computed with fewer than 2 data points
                z_scores[domain] = 0.0
                continue
            
            mean = np.mean(history)
            std = np.std(history)
            latest_ema = history[-1]
            
            if std > 1e-6: # prevent division by zero
                z_scores[domain] = (latest_ema - mean) / std
            else:
                z_scores[domain] = 0.0 # treat as 0 when there is almost no oscillation (regarded as noise)
                
        return z_scores

    def _select_domains(self, z_scores: Dict[str, float]) -> List[str]:
        """Extracts the domains to train on that satisfy the Z-score criterion."""
        selected = []
        for domain, z in z_scores.items():
            if self.z_condition == 'le' and z <= self.z_threshold:
                selected.append(domain)
            elif self.z_condition == 'ge' and z >= self.z_threshold:
                selected.append(domain)
            elif self.z_condition == 'abs_ge' and abs(z) >= abs(self.z_threshold):
                selected.append(domain)
        return selected

    def _allocate_weights(self, selected_domains: List[str]) -> Dict[str, float]:
        """Allocates ratios to the selected domains."""
        new_weights = {domain: 0.0 for domain in self.set_names}
        n_selected = len(selected_domains)
        
        if n_selected > 0:
            # 1. uniform allocation (divide 1/n among only the selected domains)
            uniform_ratio = 1.0 / n_selected
            for domain in selected_domains:
                new_weights[domain] = uniform_ratio
        else:
            # Exception handling: if it is judged that no domain should be trained, apply uniform to all
            print("[EmaZScore] Warning: No domains selected. Applying global uniform fallback.")
            global_ratio = 1.0 / self.n_domains
            for domain in self.set_names:
                new_weights[domain] = global_ratio
                
        return new_weights

    # ---------------------------------------------------------
    # Composer Callback Hooks
    # ---------------------------------------------------------
    def after_train_batch(self, state: State, logger: Logger) -> None:
        """ Track used_domain_ids and log train batches (existing code kept) """
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
        """ Measures PPL at eval_end and calls the separated internal methods in sequence to update. """
        dataset = state.train_dataloader.dataset
        if not hasattr(dataset, "update_proportion"):
            print("[EmaZScore] Warning: Dataset missing 'update_proportion'.")
            return

        # 1. Extract per-domain PPL from eval_metrics
        current_ppls = {}
        for domain in self.set_names:
            metric_key = f"{domain}_LanguageCrossEntropy"
            if metric_key in state.eval_metrics['eval']:
                loss_val = state.eval_metrics['eval'][metric_key].compute().item()
                current_ppls[domain] = np.exp(loss_val)
            else:
                last_ppl = self.ema_history[domain][-1] if self.ema_history[domain] else 10000.0
                current_ppls[domain] = last_ppl
        
        # 2. Update the EMA and compute the Z-score
        self._update_ema_history(current_ppls)
        z_scores = self._calculate_z_scores()
        
        # 3. Filter domains and allocate weights
        selected_domains = self._select_domains(z_scores)
        new_weights_dict = self._allocate_weights(selected_domains)
        
        # Convert to lists for the Dataset update
        new_props = [new_weights_dict[domain] for domain in self.set_names]
        
        # 4. Update the Dataset and log
        print(f"[EmaZScore] PPLs: {current_ppls}")
        print(f"[EmaZScore] Z-Scores: {z_scores}")
        print(f"[EmaZScore] Selected Domains: {selected_domains}")
        print(f"[EmaZScore] Updating Weights: {new_weights_dict}")
        
        dataset.update_proportion(new_props)
        
        for i, domain in enumerate(self.set_names):
            logger.log_metrics({
                f'ema_zscore/weight/{domain}': new_props[i],
                f'ema_zscore/ppl/{domain}': current_ppls[domain],
                f'ema_zscore/z_score/{domain}': z_scores.get(domain, 0.0),
                f'ema_zscore/ema_latest/{domain}': self.ema_history[domain][-1] if self.ema_history[domain] else 0.0
            })

    def state_dict(self) -> Dict[str, Any]:
        return {
            "used_domain_ids": self.used_domain_ids,
            "ema_history": {d: list(q) for d, q in self.ema_history.items()} # convert deque to list for saving
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.used_domain_ids = state_dict.get("used_domain_ids", self.used_domain_ids)
        if "ema_history" in state_dict:
            for d, lst in state_dict["ema_history"].items():
                self.ema_history[d] = deque(lst, maxlen=self.k)


class TrainPplEmaZScoreCallback(Callback):
    """
    Dynamic domain allocation callback using the train data's PPL EMA and the standard normal distribution (Z-score).
    Accumulates the loss on every Train batch, computes the Train PPL at eval_end, and adjusts the ratios.
    """
    def __init__(self, 
                 set_names: List[str],
                 target_loss: List[float] = None, 
                 proportion: List[float] = None,
                 update_type: str ="doremi",
                 k: int = 300,             
                 ema_alpha: float = 0.2,   
                 z_threshold: float = -0.5,
                 z_condition: str = 'le',  
                 ) -> None:
        self.set_names = set_names
        self.n_domains = len(set_names)
        
        # Algorithm parameters
        self.k = k
        self.ema_alpha = ema_alpha
        self.z_threshold = z_threshold
        self.z_condition = z_condition
        
        # State tracking
        self.ema_history = {domain: deque(maxlen=k) for domain in set_names}
        self.used_domain_ids = [[] for _ in range(self.n_domains)]
        
        # Variables for accumulating Train Loss
        self.train_loss_sum = {domain: [] for domain in set_names}
        self.train_batch_count = {domain: 0 for domain in set_names}
        
        self.dynamic_loading_module = DynamicLoadingModule(
            target_loss=target_loss,
            proportion=proportion,
            set_names=set_names,
            update_type=update_type,
        )
        print(f"====called TrainPplEmaZScoreCallback (k={k}, z_threshold={z_threshold})====")

    # ---------------------------------------------------------
    # Detailed feature breakdown section (same as before)
    # ---------------------------------------------------------
    def _compute_single_ema(self, domain: str, current_ppl: float) -> float:
        history = self.ema_history[domain]
        if len(history) == 0:
            return current_ppl
        
        last_ema = history[-1]
        new_ema = (self.ema_alpha * current_ppl) + ((1 - self.ema_alpha) * last_ema)
        return new_ema

    def _update_ema_history(self, current_ppls: Dict[str, float]) -> None:
        for domain in self.set_names:
            new_ema = self._compute_single_ema(domain, current_ppls[domain])
            self.ema_history[domain].append(new_ema)

    def _calculate_z_scores(self) -> Dict[str, float]:
        z_scores = {}
        for domain in self.set_names:
            history = list(self.ema_history[domain])
            if len(history) < 2:
                z_scores[domain] = 0.0
                continue
            
            mean = np.mean(history)
            std = np.std(history)
            latest_ema = history[-1]
            
            if std > 1e-6: 
                z_scores[domain] = (latest_ema - mean) / std
            else:
                z_scores[domain] = 0.0 
                
        return z_scores

    def _select_domains(self, z_scores: Dict[str, float]) -> List[str]:
        selected = []
        for domain, z in z_scores.items():
            if self.z_condition == 'le' and z <= self.z_threshold:
                selected.append(domain)
            elif self.z_condition == 'ge' and z >= self.z_threshold:
                selected.append(domain)
            elif self.z_condition == 'abs_ge' and abs(z) >= abs(self.z_threshold):
                selected.append(domain)
        return selected

    def _allocate_weights(self, selected_domains: List[str]) -> Dict[str, float]:
        new_weights = {domain: 0.0 for domain in self.set_names}
        n_selected = len(selected_domains)
        
        # Exception handling: what to do if it is judged that no domain should be trained? -> apply uniform ratios
        if n_selected == 0:
            print("[TrainPplEma] Warning: No domains selected. Applying global uniform fallback.")
            global_ratio = 1.0 / self.n_domains
            for domain in self.set_names:
                new_weights[domain] = global_ratio
        else:
            uniform_ratio = 1.0 / n_selected
            for domain in selected_domains:
                new_weights[domain] = uniform_ratio
                
        return new_weights

    def _allocate_dynamic_weights(self, selected_domains: List[str]) -> Dict[str, float]:
        new_weights = {domain: 0.0 for domain in self.set_names}
        n_selected = len(selected_domains)
        
        # Exception handling: what to do if it is judged that no domain should be trained? -> apply uniform ratios
        if n_selected == 0:
            print("[TrainPplEma] Warning: No domains selected. Applying global uniform fallback.")
            global_ratio = 1.0 
            for domain in self.set_names:
                new_weights[domain] = global_ratio
        else:
            uniform_ratio = 1.0
            for domain in selected_domains:
                new_weights[domain] = uniform_ratio
                
        return new_weights

    # ---------------------------------------------------------
    # Composer Callback Hooks
    # ---------------------------------------------------------
    def after_loss(self, state: State, logger: Logger) -> None:
        """ Called per microbatch; collects the loss at the point where state.batch and state.loss are consistent. """
        sets = state.batch["set"]
        
        # 1. Get the per-sample loss vector from the dictionary returned by the model
        if isinstance(state.loss, dict) and "ce_loss_vector" in state.loss:
            sample_losses = state.loss["ce_loss_vector"].detach()
        else:
            print("[TrainPplEma] Warning: Loss vector not found. Using fallback PPL.")
            loss_val = state.loss.item() if isinstance(state.loss, torch.Tensor) else state.loss.get("total", 0.0).item()
            sample_losses = torch.full((sets.shape[0],), loss_val, dtype=torch.float32, device=sets.device)

        sets_flat = sets.view(-1)
        
        # 🚨 [VERY IMPORTANT] Defensive code that checks the counts match, for debugging
        if sets_flat.shape[0] != sample_losses.shape[0]:
            print(f"[TrainPplEma] ERROR: Shape mismatch! sets: {sets_flat.shape[0]}, losses: {sample_losses.shape[0]}")

        # 2. Compute the per-domain mean Loss within the microbatch and accumulate only one value
        sets_list = sets.view(-1).cpu().tolist()
        batch_domain_losses = {d: [] for d in self.set_names}
        for domain_id, loss_val in zip(sets_list, sample_losses):
            if domain_id < self.n_domains:
                domain_name = self.set_names[domain_id]
                batch_domain_losses[domain_name].append(loss_val.item())
        
        for domain in self.set_names:
            if batch_domain_losses[domain]:
                avg_loss = np.mean(batch_domain_losses[domain])
                self.train_loss_sum[domain].append(avg_loss)
                self.train_batch_count[domain] += 1

    def after_train_batch(self, state: State, logger: Logger) -> None:
        """ Track used_domain_ids and log weights at the full-batch level"""
        idx = state.batch["idx"]
        sets = state.batch["set"]
        
        # Existing used_domain_ids tracking logic
        local_rank = torch_dist.get_rank() if torch_dist.is_initialized() else 0
        device = torch.device(f"cuda:{local_rank}")
        idx = idx.to(device)
        sets = sets.to(device)
        
        if torch_dist.is_initialized():
            world_size = torch_dist.get_world_size()
            gathered_idx = [torch.zeros_like(idx) for _ in range(world_size)]
            gathered_sets = [torch.zeros_like(sets) for _ in range(world_size)]
            torch_dist.all_gather(gathered_idx, idx)
            torch_dist.all_gather(gathered_sets, sets)
            all_idx = torch.cat(gathered_idx)
            all_sets = torch.cat(gathered_sets)
        else:
            all_idx = idx
            all_sets = sets
        
        for i in range(self.n_domains):
            mask = all_sets == i
            domain_idx = all_idx[mask]
            self.used_domain_ids[i].extend(domain_idx.cpu().tolist())

        # Log weights
        prop = state.train_dataloader.dataset.proportion
        for domain in self.set_names:
            logger.log_metrics({f'metrics/train/{domain}_weight': round(prop[self.set_names.index(domain)], 4)})

    def eval_end(self, state: State, logger: Logger) -> None:
        """ Synchronizes the accumulated Train Loss across GPUs, computes the PPL, and updates the ratios. """
        dataset = state.train_dataloader.dataset
        if not hasattr(dataset, "update_proportion"):
            print("[TrainPplEma] Warning: Dataset missing 'update_proportion'.")
            return

        local_rank = torch_dist.get_rank() if torch_dist.is_initialized() else 0
        device = torch.device(f"cuda:{local_rank}")

        # 1. Synchronize in a multi-GPU environment (collect the accumulated Loss lists of all GPUs)
        if torch_dist.is_initialized():
            world_size = torch_dist.get_world_size()
            gathered_losses = [None for _ in range(world_size)]
            # Use all_gather_object to merge the list objects
            torch_dist.all_gather_object(gathered_losses, self.train_loss_sum)
            
            # Merge the data of all GPUs into a single list
            merged_losses = {domain: [] for domain in self.set_names}
            for gpu_losses in gathered_losses:
                for domain in self.set_names:
                    merged_losses[domain].extend(gpu_losses[domain])
        else:
            merged_losses = self.train_loss_sum

        # 2. Sequentially update the per-domain EMA based on the collected individual Losses
        current_ppls = {}
        for domain in self.set_names:
            domain_losses = merged_losses[domain]
            if len(domain_losses) > 0:
                # Compute PPL individually for each data (training sample) point and update the EMA
                for loss_val in domain_losses:
                    ppl_val = np.exp(loss_val)
                    new_ema = self._compute_single_ema(domain, ppl_val)
                    self.ema_history[domain].append(new_ema)
                
                # PPL value for logging purposes (simple mean for reference)
                current_ppls[domain] = np.exp(np.mean(domain_losses))
            else:
                current_ppls[domain] = self.ema_history[domain][-1] if self.ema_history[domain] else 10000.0
                
            # Reset the accumulation list and count for the next interval 
            self.train_loss_sum[domain] = []
            self.train_batch_count[domain] = 0
        
        # 3. Compute the Z-score (EMA was accumulated per individual point, so compute it from the current history)
        z_scores = self._calculate_z_scores()
        
        # 4. Filter domains and allocate weights
        selected_domains = self._select_domains(z_scores)
        new_weights_dict = self._allocate_dynamic_weights(selected_domains)
        
        ppl_weights = [new_weights_dict[domain] for domain in self.set_names]
        
        # 5. Update the Dataset and log
        if local_rank == 0:
            print(f"\n[TrainPplEma] Multi-GPU Synced Train PPLs: {{k: round(v, 4) for k, v in current_ppls.items()}}")
            print(f"[TrainPplEma] Z-Scores: {{k: round(v, 4) for k, v in z_scores.items()}}")
            print(f"[TrainPplEma] Selected Domains: {selected_domains}")
        
        self.dynamic_loading_module.eval_end(state, logger, ppl_weights)
        logger.log_metrics({
            'metrics/eval/train_ema_zscore/selected_domains': selected_domains
        })
        for i, domain in enumerate(self.set_names):
            logger.log_metrics({
                f'metrics/eval/train_ema_zscore/weight/{domain}': ppl_weights[i],
                f'metrics/eval/train_ema_zscore/train_ppl/{domain}': current_ppls[domain],
                f'metrics/eval/train_ema_zscore/z_score/{domain}': z_scores.get(domain, 0.0),
                f'metrics/eval/train_ema_zscore/ema_latest/{domain}': self.ema_history[domain][-1] if self.ema_history[domain] else 0.0
            })

    def state_dict(self) -> Dict[str, Any]:
        return {
            "used_domain_ids": self.used_domain_ids,
            "ema_history": {d: list(q) for d, q in self.ema_history.items()},
            "base_proportion": self.dynamic_loading_module.base_proportion
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.used_domain_ids = state_dict.get("used_domain_ids", self.used_domain_ids)
        if "ema_history" in state_dict:
            for d, lst in state_dict["ema_history"].items():
                self.ema_history[d] = deque(lst, maxlen=self.k)


class DynamicLoadingModule():
    """ 
        Callback for dynamic loading of data from different domains. The key components include 1) calculate the new proportion after each evaluation step; 2) update proportion in the dataset objective; 3) save the used domain ids after each epoch for resuming training from a previous checkpoint to make sure that used samples are not used again.
    """
    def __init__(self, 
                 target_loss: List[float] = None, 
                 proportion: List[float] = None,
                 set_names: List[str] = None,
                 update_type: str ="doremi", 
                 ) -> None:
        self.set_names = set_names
        self.n_domains = len(set_names)
        self.update_type = update_type 
        self.target_loss = target_loss
        self.proportion = proportion
        self.count = -1
        self.used_domain_ids = [[] for _ in range(self.n_domains)]
        self.base_proportion = None
        print("Target loss:", self.target_loss)
            
    def update_proportion(self, current_prop, losses):
        """ Update the proportion of each domain """
        diff = torch.tensor(losses) - torch.tensor(self.target_loss)
        eta = 1.
        c = 1e-4 # following Doremi (Xie et al., 2023)
        if self.update_type == "doremi": # update with exponential descent
            updated_alpha = torch.log(torch.tensor(current_prop)) + eta * diff 
            updated_alpha = torch.nn.functional.softmax(updated_alpha, dim=0)
            updated_domain_weights = (1-c) * updated_alpha + c / self.n_domains
        elif self.update_type == "bandit": 
            updated_alpha = torch.tensor(current_prop) + eta * diff 
            updated_alpha = torch.nn.functional.softmax(updated_alpha, dim=0)
            updated_domain_weights = (1-c) * updated_alpha + c / self.n_domains
        elif self.update_type == "constant": # constant proportion
            updated_domain_weights = torch.tensor(current_prop)
            

        updated_domain_weights = updated_domain_weights.numpy().astype('float64')
        updated_domain_weights = updated_domain_weights / updated_domain_weights.sum()
        if np.isnan(updated_domain_weights).any():
            return current_prop
        
        return updated_domain_weights.tolist()
    
    def after_train_batch(self, state: State, logger: Logger) -> None:
        """ Print out the number of used samples in each domain after each training batch, and log the updated proportion of each domain """
    
        idx = state.batch["idx"]
        sets = state.batch["set"]
        ### XXX
        local_rank = torch.distributed.get_rank()
        # local_rank = dist.get_rank() if dist.is_initialized() else 0
        device = torch.device(f"cuda:{local_rank}")
        idx = idx.to(device)
        sets = sets.to(device)
        ###
        all_idx = torch.cat(dist.all_gather(idx))
        all_sets = torch.cat(dist.all_gather(sets))
        dist.barrier() 
        
        for i in range(self.n_domains):
            mask = all_sets == i
            domain_idx = all_idx[mask]
            self.used_domain_ids[i].extend(domain_idx.cpu().tolist())
            # for debugging
            # print(f"domain {i} used {mask.sum().item()} new samples")

        prop = state.train_dataloader.dataset.proportion
        for domain in self.set_names:
            logger.log_metrics({f'metrics/train/{domain}_weight': round(prop[self.set_names.index(domain)], 4)})
        
    def eval_end(self, state: State, logger: Logger, ppl_weights: List[float]) -> None:
        """ Update the proportion of each domain after each evaluation and update the dataset """
        if self.base_proportion is None:
            self.base_proportion = state.train_dataloader.dataset.proportion
        current_prop = self.base_proportion
        losses = []
        print("loss keys: ", state.eval_metrics['eval'].keys())
        for domain in self.set_names:
            losses.append(state.eval_metrics["eval"][f"{domain}_LanguageCrossEntropy"].compute().item())
        new_proportion = self.update_proportion(current_prop, losses)
        print("new_proportion: ", new_proportion, "ppl_proportion: ", ppl_weights)

        self.base_proportion = new_proportion

        new_proportion_with_ppl = [x * y for x, y in zip(new_proportion, ppl_weights)]
        proportion_sum = sum(new_proportion_with_ppl)
        new_proportion_with_ppl = [x / proportion_sum for x in new_proportion_with_ppl] 
        print("new_proportion_with_ppl: ", new_proportion_with_ppl)
        state.train_dataloader.dataset.update_proportion(new_proportion_with_ppl)
    
    def state_dict(self) -> Dict[str, Any]:
        """ Save the used domain ids after each epoch, for resuming training from a previous checkpoint to make sure that used samples are not used again """
        return {"used_domain_ids": self.used_domain_ids}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """ Load the used domain ids """
        self.used_domain_ids = state_dict["used_domain_ids"]