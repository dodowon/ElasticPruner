# Based on princeton-nlp/LLM-Shearing (MIT License), Copyright (c) 2023 Princeton Natural Language Processing
# Modified by dodowon, 2026. See THIRD_PARTY_NOTICES.md for details.

import os
import time
from typing import Any, Dict, List

import torch
from composer import Callback, Logger, State
from composer.loggers import Logger
from composer.utils import dist
from torch.nn import functional as F
import numpy as np

class DynamicLoadingCallbackTrackerOnly(Callback):
    """
    Lightweight version of DynamicLoadingCallback.
    Only tracks used_domain_ids, without updating weights.
    This callback is required because state.py references used_domain_ids when saving checkpoints.
    """
    def __init__(self, 
                 target_loss: List[float] = None, 
                 proportion: List[float] = None,
                 set_names: List[str] = None,
                 update_type: str = "doremi",
                 ) -> None:
        self.set_names = set_names
        self.n_domains = len(set_names)
        self.used_domain_ids = [[] for _ in range(self.n_domains)]

    def after_train_batch(self, state: State, logger: Logger) -> None:
        idx = state.batch["idx"]
        sets = state.batch["set"]
        
        local_rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        idx = idx.to(device)
        sets = sets.to(device)
        
        all_idx = torch.cat(dist.all_gather(idx))
        all_sets = torch.cat(dist.all_gather(sets))
        dist.barrier()
        
        for i in range(self.n_domains):
            mask = all_sets == i
            domain_idx = all_idx[mask]
            self.used_domain_ids[i].extend(domain_idx.cpu().tolist())

    def state_dict(self) -> Dict[str, Any]:
        return {"used_domain_ids": self.used_domain_ids}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.used_domain_ids = state_dict["used_domain_ids"]

class DynamicLoadingCallbackAirEMA(Callback):
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
        self.loss_history = [[] for _ in range(self.n_domains)]
        self.ema = []
        print("====called AirDynamicLoadingCallback====")

    def calc_ema(self, losses):
        """ Update the proportion of each domain """

        ema = []
        n = 5
        tau = 1
        alpha = 2 / (n + 1)
        diff = torch.tensor(losses) - torch.tensor(self.target_loss)
        if self.ema == []:
            self.ema = diff
        else:
            ema = diff * alpha + self.ema * (1-alpha)
            self.ema = ema
        result = self.ema / tau
        result = torch.nn.functional.softmax(result, dim=0)
        result = result.numpy().astype('float64')
        result = result / result.sum()
        return result.tolist()
    def update_proportion(self, current_prop, losses):
        """ Update the proportion of each domain """
        #diff = torch.tensor(losses) - torch.tensor(self.target_loss)
        diff = torch.tensor(losses)
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
        # if np.isnan(updated_domain_weights).any():
        #     return current_prop
        
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
        
    def eval_end(self, state: State, logger: Logger) -> None:
        """ Update the proportion of each domain after each evaluation and update the dataset """
        current_prop = state.train_dataloader.dataset.proportion
        losses = []
        print("loss keys: ", state.eval_metrics['eval'].keys())
        for domain in self.set_names:
            losses.append(state.eval_metrics["eval"][f"{domain}_LanguageCrossEntropy"].compute().item())
        ema_diff = self.calc_ema(losses)
        new_proportion = self.update_proportion(current_prop, ema_diff)
        state.train_dataloader.dataset.update_proportion(new_proportion)
    
    def state_dict(self) -> Dict[str, Any]:
        """ Save the used domain ids after each epoch, for resuming training from a previous checkpoint to make sure that used samples are not used again """
        return {"used_domain_ids": self.used_domain_ids}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """ Load the used domain ids """
        self.used_domain_ids = state_dict["used_domain_ids"]

class DynamicLoadingCallback(Callback):
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
        
    def eval_end(self, state: State, logger: Logger) -> None:
        """ Update the proportion of each domain after each evaluation and update the dataset """
        current_prop = state.train_dataloader.dataset.proportion
        losses = []
        print("loss keys: ", state.eval_metrics['eval'].keys())
        for domain in self.set_names:
            losses.append(state.eval_metrics["eval"][f"{domain}_LanguageCrossEntropy"].compute().item())
        new_proportion = self.update_proportion(current_prop, losses)
        state.train_dataloader.dataset.update_proportion(new_proportion)
    
    def state_dict(self) -> Dict[str, Any]:
        """ Save the used domain ids after each epoch, for resuming training from a previous checkpoint to make sure that used samples are not used again """
        return {"used_domain_ids": self.used_domain_ids}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """ Load the used domain ids """
        self.used_domain_ids = state_dict["used_domain_ids"]
     
