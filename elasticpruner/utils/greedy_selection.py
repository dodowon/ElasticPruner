# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import math
import sys

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import copy

class GreedyArchitectureSelector:
    """
    Algorithm 1: Optimized Greedy Architecture Selection with Cumulative Sums
    (PyTorch Tensor version)

    This class is initialized with an l0_module (mask module).
    l0_module must provide the following two interfaces:
    
    1. get_importance_scores(): 
       Returns a (z_L, z_h, z_i) tuple.
       Each is a *Torch tensor* holding the sorted (descending)
       importance scores of the layer, hidden dimension, and intermediate dimension.
       
    2. calculate_params(L, h, i):
       Returns the total parameter count (int) for the given architecture dimensions (L, h, i).
    """
    
    def __init__(self, l0_module):
        """
        Initializes the selector with an l0_module.
        
        Args:
            l0_module (object): An object that has the get_importance_scores() and calculate_params(L, h, i)
                                methods.
        """
        print("Initializing GreedyArchitectureSelector (Torch version)...")
        self.l0_module = l0_module
        
        # 1. Get the sorted importance scores (Torch tensors) from l0_module.
        try:
            mask = self.get_importance_scores()
            self.z_L = mask['layer'].z_loga
            self.z_i = mask['intermediate'].z_loga
            self.z_h = mask['hidden'].z_loga
            # self.z_L, self.z_h, self.z_i = self.l0_module.get_importance_scores()
        except AttributeError:
            raise AttributeError("l0_module needs a 'get_importance_scores' method.")
        except Exception as e:
            print(f"Error while calling l0_module.get_importance_scores(): {e}")
            raise
        
        if self.z_i.dim() == 2:
            print(f"  Info: The z_i tensor is 2D (Shape: {self.z_i.shape}). Summing over dim=0 (layer).")
            self.z_i = torch.sum(self.z_i, dim=0)
            print(f"  Info: The z_i tensor was reduced to 1D (Shape: {self.z_i.shape}).")

        # Check the tensor's device and dtype (GPU/CPU compatibility)
        self.device = self.z_L.device
        self.dtype = self.z_L.dtype
        print(f"Input tensor info: Device={self.device}, Dtype={self.dtype}")
        print(f"shapes Layer {self.z_L.shape}, inter {self.z_i.shape}, hidden {self.z_h.shape} ")

        # 2. Pre-compute the cumulative sums arrays (O(N) optimization)
        # Use 0-based indexing and pad the 0th element with 0.
        zero_pad_L = torch.zeros(1, dtype=self.dtype, device=self.device)
        zero_pad_h = torch.zeros(1, dtype=self.z_h.dtype, device=self.z_h.device)
        zero_pad_i = torch.zeros(1, dtype=self.z_i.dtype, device=self.z_i.device)
        
        self.cumsum_zL = torch.cat((zero_pad_L, torch.cumsum(self.z_L, dim=0)))
        self.cumsum_zh = torch.cat((zero_pad_h, torch.cumsum(self.z_h, dim=0)))
        self.cumsum_zi = torch.cat((zero_pad_i, torch.cumsum(self.z_i, dim=0)))

        # Get the parameter calculation function of l0_module.
        try:
            self.CalculateParams = self.calculate_params
        except AttributeError:
            raise AttributeError("l0_module needs a 'calculate_params' method.")

        # Store the maximum dimensions
        self.max_L = len(self.z_L)
        self.max_h = len(self.z_h)
        self.max_i = len(self.z_i)
        
        print(f"Maximum dimensions set: L={self.max_L}, h={self.max_h}, i={self.max_i}")

    def get_importance_scores(self):
        mask = copy.deepcopy(self.l0_module.masks.cpu())
        return mask

    def calculate_params(self, L, h, i):
        """
        Computes the parameter count using the Llama example formula from the algorithm image.
        (This function returns a scalar int, so no modification is needed)
        """
        if L == 0 or h == 0:
            return 0
        self.V = 32000
        # P = L * (4*h^2 + 3*h*i + 2*h) + V*h + h (formula from the image)
        params = L * (4 * h**2 + 3 * h * i + 2 * h) + self.V * h + h
        
        return int(params)

    def greedy_select(self, P_target, L_base, h_base, i_base, c_h, c_i):
        """
        Runs the `GREEDYSELECT` procedure of Algorithm 1.

        Args:
            P_target (int): Target parameter count.
            L_base (int): Base number of layers.
            h_base (int): Base hidden dimension.
            i_base (int): Base intermediate dimension.
            c_h (int): Chunk size for increasing the hidden dimension.
            c_i (int): Chunk size for increasing the intermediate dimension.

        Returns:
            tuple: (L, h, i) - the selected optimal architecture dimensions.
        """
        
        # 3. Initialize the architecture and parameter count
        L, h, i = L_base, h_base, i_base
        P_current = self.CalculateParams(L, h, i)
        
        print(f"Search started: Base (L={L}, h={h}, i={i}), P_current={P_current}, P_target={P_target}")
        
        # 4. Greedily select components until the target parameter count is reached
        while P_current < P_target:
            # Initialize the value (Value-per-parameter) of each option (L, h, i)
            V_L, V_h, V_i = float('-inf'), float('-inf'), float('-inf')
            
            # --- a. Compute the layer (L) addition option ---
            if L < self.max_L:
                # Importance score of the next layer (L) (0-based indexing)
                S_L = self.z_L[L] # 0-dim tensor
                
                # Compute the parameter change
                P_next_L = self.CalculateParams(L + 1, h, i)
                delta_P_L = P_next_L - P_current
                
                if delta_P_L > 0:
                    V_L = S_L / delta_P_L # tensor / scalar -> tensor

            # --- b. Compute the hidden dimension (h) increase option ---
            if h + c_h <= self.max_h:
                # Sum of importance scores of the next chunk (h ~ h+c_h-1)
                S_h = self.cumsum_zh[h + c_h] - self.cumsum_zh[h] # 0-dim tensor
                
                # Compute the parameter change
                P_next_h = self.CalculateParams(L, h + c_h, i)
                delta_P_h = P_next_h - P_current
                
                if delta_P_h > 0:
                    V_h = S_h / delta_P_h
                    
            # --- c. Compute the intermediate dimension (i) increase option ---
            if i + c_i <= self.max_i:
                # Sum of importance scores of the next chunk (i ~ i+c_i-1)
                S_i = self.cumsum_zi[i + c_i] - self.cumsum_zi[i] # 0-dim tensor

                # Compute the parameter change
                P_next_i = self.CalculateParams(L, h, i + c_i)
                delta_P_i = P_next_i - P_current
                
                if delta_P_i > 0:
                    V_i = S_i / delta_P_i

            # --- Select the best option ---
            # V_L, V_h, V_i are 0-dim tensors or float('-inf')
            options = {'L': V_L, 'h': V_h, 'i': V_i}
            
            # Filter only the valid (non -inf) options
            # (Comparing a tensor and a float works fine in Python)
            valid_options = {k: v for k, v in options.items() if v > float('-inf')}
            
            if not valid_options:
                # 5.2. Stop if no more components can be added
                print("The value of every option is invalid (Delta P <= 0 or maximum dimension reached). Stopping the search.")
                break

            # Select the option with the highest value (max performs the tensor comparison)
            best_option = max(valid_options, key=valid_options.get)
            print(f"L, h, i: {S_L}, {S_h}, {S_i}")
            print(f"{best_option}, {valid_options}")
            # 4.2. Update the architecture
            if best_option == 'L':
                L += 1
                # Use .item() to extract the scalar value from the tensor for printing
                print(f"  -> Option selected: add layer (New L={L}), Value={V_L.item():.2e}")
            elif best_option == 'h':
                h += c_h
                print(f"  -> Option selected: increase hidden dimension (New h={h}), Value={V_h.item():.2e}")
            elif best_option == 'i':
                i += c_i
                print(f"  -> Option selected: increase intermediate dimension (New i={i}), Value={V_i.item():.2e}")

            # 5.3. Update the current parameter count
            P_current = self.CalculateParams(L, h, i)
            print(f"     New P_current: {P_current} / {P_target}")
            
        # 5.6. Return the final architecture after the loop ends
        print(f"\nSearch finished. Final Architecture (L, h, i): ({L}, {h}, {i}), Final Params: {P_current}")
        return L, h, i