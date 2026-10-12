# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

### Find the function set to replace layers with a genetic algorithm

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from datetime import datetime
from copy import deepcopy
from types import SimpleNamespace
import math
from elasticpruner.utils.genetic_math import top_k_sim_logits, topk_cosine_sim_logits, top_k_rwem
random.seed(time.time())

def change_target_mask_size(l0_module, config):
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

def kl_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor):
    """
    KL(P || Q) where P, Q are distributions from logits
    logits_p, logits_q: shape (..., vocab_size)
    """
    # Convert to a probability distribution
    logits_q = logits_q.to(logits_p.device)
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)

    # Using log_softmax allows stable computation
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)

    # KL(P||Q) = sum( p * (log_p - log_q) )
    kl = torch.sum(p * (log_p - log_q), dim=-1)  # over the last dimension
    kl_mean = kl.mean()
    similarity = torch.exp(-kl_mean)
    return similarity.item()



def formatted_current_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")  # e.g., 2025-01-23 15:30:45

def add_median_to_tensor(x):
    median_value = torch.median(x)

    # Add the median value to every element of the tensor
    result_tensor = x + median_value

    return result_tensor

def subtract_median_to_tensor(x):
    median_value = torch.median(x)

    # Add the median value to every element of the tensor
    result_tensor = x - median_value

    return result_tensor

def add_gaussian_noise(x, mean=0.0, std=0.1):
    noise = torch.randn_like(x) * std + mean
    return x + noise

def add_uniform_noise(x, low=-0.1, high=0.1):
    noise = (high - low) * torch.rand_like(x) + low
    return x + noise

def standardize_tensor(x):
    return (x - torch.mean(x)) / torch.std(x)

def add_point_5(x):
    return x + 0.5

def sub_point_5(x):
    return x - 0.5

def mul_2(x):
    return x * 2

def mul_3(x):
    return x * 3

def add_1(x):
    return x + 1.0

def solve_d_model(total_params, n_layers, intermediate_size, vocab_size):
    """
    total = n_layers*(4*d^2 + 3*d*inter + 2*d) + 2*vocab*d + d
    => 4*n_layers*d^2 + (3*n_layers*inter + 2*n_layers + 2*vocab + 1)*d - total = 0
    """
    a = n_layers * 4
    b = n_layers * 3 * intermediate_size + n_layers * 2 + 2 * vocab_size + 1
    c = -total_params

    import math
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No solution exists.")
    return (-b + math.sqrt(discriminant)) / (2 * a)


def solve_n_layers(total_params, d_model, intermediate_size, vocab_size):
    """
    total = n_layers * per_layer + 2*vocab*d + d
    => n_layers = (total - 2*vocab*d - d) / per_layer
    """
    per_layer = 4 * d_model ** 2 + 3 * d_model * intermediate_size + 2 * d_model
    embedding_part = 2 * vocab_size * d_model + d_model
    return (total_params - embedding_part) // per_layer

def solve_intermediate_size(total_params, d_model, n_layers, vocab_size):
    """
    total = n_layers*(4*d^2 + 3*d*inter + 2*d) + 2*vocab*d + d
    => inter = (total - 2*vocab*d - d - n_layers*(4*d^2 + 2*d)) / (3*n_layers*d)
    """
    embedding_part = 2 * vocab_size * d_model + d_model
    non_ffn_per_layer = 4 * d_model ** 2 + 2 * d_model
    numerator = total_params - embedding_part - n_layers * non_ffn_per_layer
    return numerator // (n_layers * 3 * d_model)

# (Optional) to align to a multiple of the number of heads:
def solve_d_model_multiple_of_heads(total_params, n_layers, intermediate_size, vocab_size, n_heads):
    d_cont = solve_d_model(total_params, n_layers, intermediate_size, vocab_size)
    # Snap to the nearest multiple of n_heads
    d_rounded = max(n_heads, int(round(d_cont / n_heads)) * n_heads)
    return d_rounded

def pick_n_heads_by_cap(d_model: int, max_heads: int) -> int:
    """
    Returns the largest divisor of d_model that does not exceed max_heads.
    Always guarantees at least 1 (1 is a divisor of every integer).
    """
    if d_model <= 0 or max_heads < 1:
        raise ValueError("d_model must be positive and max_heads must be at least 1.")
    
    best = 1  # 1 is always a divisor, so it is the minimum safe value
    # Divisor search: O(sqrt(d_model))
    i = 1
    while i * i <= d_model:
        if d_model % i == 0:
            a, b = i, d_model // i
            if a <= max_heads and a > best:
                best = a
            if b <= max_heads and b > best:
                best = b
        i += 1
    return best



class AirGeneticSparsity:
    MODULE_LIST = ['d_model', 'intermediate_size', 'n_layers']
    def __init__(self, model,teacher_model,target_params=int(1.3e9),sim_func="cosine"):
        self.model = model
        self.teacher = teacher_model
        self.criterion = nn.CrossEntropyLoss()
        self.population = []
        self.target_params = target_params
        self.target_output = None
        self.init_taget_module_cfg = None
        self.origin_module_cfg = None
        self.range_ratio = 0.01
        sim_func_name = sim_func['func_name']
        topk_rate = sim_func.get("topk_rate")
        if sim_func_name == "cosine":
            self.sim_func = self.calculate_cosine_similarity
        elif sim_func_name == "kl" :
            self.sim_func = kl_from_logits
        elif sim_func_name == "topk_exact":
            self.sim_func = lambda x, y: top_k_sim_logits(x, y, k=topk_rate)
        elif sim_func_name == "topk_cos":
            self.sim_func = lambda x, y: topk_cosine_sim_logits(x, y, k=topk_rate)
        elif sim_func_name == "rwem":
            self.sim_func = lambda x, y: top_k_rwem(x, y, k=topk_rate)
        else:
            raise Exception(f"{sim_func} is not support")

    def infer_missing_llama_config(self,
        target_params: int,
        d_model: int = None,
        n_heads: int = None,
        n_layers: int = None,
        intermediate_size: int = None,
        vocab_size: int = 32000,
        total_params=0
    ):
        """
        LLaMA-style config computation: given 3 of the 4, computes the remaining 1
        and adjusts so that the total parameter count matches target_params.
        """
        # First define the parameter count function
        def compute_params(d_model, n_heads, n_layers, intermediate_size, vocab_size):
            attn = 4 * d_model * d_model
            ffn = 2 * d_model * intermediate_size + intermediate_size * d_model
            ln = 2 * d_model
            per_layer = attn + ffn + ln
            total = n_layers * per_layer
            embedding = vocab_size * d_model
            return total + embedding * 2 + d_model
        total = target_params
        # Check that only 3 were given
        given = [d_model, n_layers, intermediate_size]
        if given.count(None) != 1:
            print(f"[WARNING] Exactly one of d_model, n_layers, intermediate_size must be None: {given}")
            # raise ValueError("Exactly one of d_model, n_layers, intermediate_size must be None.", given)
            return None

        # if n_heads is not None and d_model is not None and d_model % n_heads != 0:
        #     n_heads = nh = pick_n_heads_by_cap(d_model=d_model, max_heads=32)

        # Change to a multiple of 2
        if isinstance(d_model, int):
            # if d_model > self.origin_module_cfg['d_model'] * 0.7:
            #     d_model = round(self.origin_module_cfg['d_model'] * 0.7)
            # d_model += d_model % 2
            if d_model > 4096: d_model=4096
            d_model = d_model // 128 * 128
            n_heads = d_model // 128
        if isinstance(n_layers, int):
            n_layers += n_layers % 2
            if n_layers > 32: n_layers=32
        if isinstance(intermediate_size, int):
            intermediate_size += intermediate_size % 2
            if intermediate_size > 11008: intermediate_size = 11008

        # Estimate the missing item
        if d_model is None:
            # for _ in range(64):
                # dm = random.randint(256, 4096)
            # for dm in range(256, 4096, 64):  # increase in steps of 8
                # total = compute_params(dm, n_heads, n_layers, intermediate_size, vocab_size)
                dm = solve_d_model(total, n_layers=n_layers, intermediate_size=intermediate_size, vocab_size=vocab_size)
                dm += dm % 2
                lower = int(dm // 128) * 128
                upper = lower + 128
                # if dm > self.origin_module_cfg['d_model'] * 0.7:
                #     dm = int(round(self.origin_module_cfg['d_model'] * 0.7))
                #     dm += dm % 2
                # dm = int(dm)
                # n_heads = pick_n_heads_by_cap(dm, 32)
                p_lower = compute_params(lower, None, n_layers, intermediate_size, vocab_size)
                p_upper = compute_params(upper, None, n_layers, intermediate_size, vocab_size)
                if abs(p_lower - total_params) <= abs(p_upper - total_params):
                    # return lower, p_lower
                    dm = lower
                    new_total = p_lower
                else:
                    dm = upper
                    new_total = p_upper
                new_total = compute_params(dm, n_heads, n_layers, intermediate_size, vocab_size)
                n_heads = dm // 128
                indiv = dict(
                            d_model=dm,
                            n_heads=n_heads,
                            n_layers=n_layers,
                            intermediate_size=intermediate_size,
                            total_params=new_total,
                            vocab_size=vocab_size
                        )
                if abs(new_total - target_params) < target_params * self.range_ratio and dm > 0 and self.check_indiv_constraint(indiv):  # error range ±5M
                    if dm % n_heads == 0:  # check that it is divisible by the number of heads
                        return indiv
        # elif n_heads is None:
        #     # for nh in range(64, 1, -2):
        #         # nh = random.randint(1,32)
        #         # total = compute_params(d_model, nh, n_layers, intermediate_size, vocab_size)
        #         nh = pick_n_heads_by_cap(d_model=d_model, max_heads=32)
        #         nh += nh % 2
        #         new_total = compute_params(d_model, nh, n_layers, intermediate_size, vocab_size)
        #         if abs(new_total - target_params) < 5e7 and d_model % nh ==0 :
        #             return dict(
        #                 d_model=d_model,
        #                 n_heads=nh,
        #                 n_layers=n_layers,
        #                 intermediate_size=intermediate_size,
        #                 total_params=new_total,
        #                 vocab_size=vocab_size
        #             )

        elif n_layers is None:
            # for nl in range(1, 32):
            # for _ in range(64):
                # nl = random.randint(1,32)
                # total = compute_params(d_model, n_heads, nl, intermediate_size, vocab_size)
                try:
                    # nl = round((total - vocab_size*d_model) / (4*d_model**2 + 3*d_model*intermediate_size + 4*d_model))
                    nl = solve_n_layers(total, d_model, intermediate_size, vocab_size)
                except:
                    return None
                nl += nl % 2
                new_total = compute_params(d_model, n_heads, nl, intermediate_size, vocab_size)
                # print(abs(total - target_params))
                indiv = dict(
                        d_model=d_model,
                        n_heads=n_heads,
                        n_layers=nl,
                        intermediate_size=intermediate_size,
                        total_params=new_total,
                        vocab_size=vocab_size
                    )
                if abs(new_total - target_params) < target_params * self.range_ratio and nl > 0 and self.check_indiv_constraint(indiv):
                    return indiv

        elif intermediate_size is None:
            # for i in range(256, 11008, 64):  # approximate FFN range
            # for _ in range(64):
                # i = random.randint(256, 11008)
                # total = compute_params(d_model, n_heads, n_layers, i, vocab_size)
                try:
                    # i = round(((total - vocab_size*d_model) / n_layers - 4*d_model**2 - 4*d_model) / (3*d_model))
                    i = solve_intermediate_size(total, d_model, n_layers, vocab_size)
                except:
                    return None
                i += i % 2
                new_total = compute_params(d_model, n_heads, n_layers, i, vocab_size)
                indiv = dict(
                        d_model=d_model,
                        n_heads=n_heads,
                        n_layers=n_layers,
                        intermediate_size=i,
                        total_params=new_total,
                        vocab_size=vocab_size
                    )
                if abs(new_total - target_params) < target_params * self.range_ratio and i > 0 and self.check_indiv_constraint(indiv):
                    return indiv

        return None

    def randomize_bits(self,bit_length=8):
        """
        num:        integer to convert
        bit_length: bit length (number of bits to keep)
        """
        # Randomly generate 0/1 for the bit length

        bits = [str(random.randint(0, 1)) for _ in range(bit_length)]
        if bits[0] == '0' and bits[1] == '0':
            bits[1] = '1'
        # list → string → integer conversion
        bit_str = ''.join(bits)
        new_num = int(bit_str, 2)
        return new_num, bit_str

    def check_indiv_constraint(self,indiv):
        for key in self.MODULE_LIST:
            if indiv[key] > self.origin_module_cfg[key]:
                return False
        return True

    def calculate_cosine_similarity(self,tensor1, tensor2, dim=2):
        """
        Calculates the cosine similarity between two tensors along the specified dimension.
    
        Args:
            tensor1 (torch.Tensor): The first tensor.
            tensor2 (torch.Tensor): The second tensor.
            dim (int): The dimension along which to compute cosine similarity.
        
        Returns:
            torch.Tensor: Cosine similarity values.
        """
        tensor1 = tensor1.to(tensor2.device)
        return F.cosine_similarity(tensor1, tensor2, dim=dim).mean().item()

    def intialize_population(self,population_size, target_modules=[], mutant=False):
        """
        target_modules : List[Dict]
            * a dict that includes initial values, such as {"head" : 16}
        """
        if target_modules and self.init_taget_module_cfg is None:
            self.init_taget_module_cfg = target_modules
        if mutant:
            self.population = self.population[:-population_size]
            target_modules = self.init_taget_module_cfg
        else:
            self.population = []
        for _ in range(population_size):
            now_target_modules = deepcopy(target_modules)
            remove_key = random.choice(self.MODULE_LIST)
            now_target_modules.pop(remove_key)
            candidate = {}
            for key in now_target_modules.keys():
                bit_int = format(target_modules[key], 'b')
                _, new_bit = self.randomize_bits(len(bit_int))
                candidate[key] = new_bit
            for key in self.MODULE_LIST:
                if key not in now_target_modules.keys():
                    candidate[key] = None
            self.population.append(candidate)
        # for _ in range(population_size):
        #     candidate = [self.FUNCTION_LIST[random.randint(0, len(self.FUNCTION_LIST)-1)] for _ in range(function_length)]
        #     population.append(candidate)
        # if not special_functions:
        #     for i in range(len(special_functions)):
        #         population[i] = special_functions[i]
        # return population

    def get_random_population(self,population_size):
        """
        target_modules : List[Dict]
            * a dict that includes initial values, such as {"head" : 16}
        """
        population = []
        
        for _ in range(population_size):
            now_target_modules = deepcopy(self.init_taget_module_cfg)
            remove_key = random.choice(list(now_target_modules.keys()))
            now_target_modules.pop(remove_key)
            candidate = {}
            for key in now_target_modules.keys():
                bit_int = format(self.init_taget_module_cfg[key], 'b')
                _, new_bit = self.randomize_bits(len(bit_int))
                candidate[key] = new_bit
            for key in self.MODULE_LIST:
                if key not in now_target_modules.keys():
                    candidate[key] = None
            population.append(candidate)

        return population

    def evolution(self, x, population_size):
        import matplotlib.pyplot as plt
        iter_index = []
        score_index = []
        history = {"d_model": [], "intermediate_size": [], "n_layers": [], "n_heads": []}
        best_indiv = None
        best_score = -30000
        iter_num = 10
        patience = 0
        before_best = 0
        top_logits = None
        elite_score_list = []
        for i in range(1, iter_num):
            score_list = []
            new_population = []
            for indiv in self.population:
                sim, new_indv, logits = self.apply_config(indiv, x)
                print(sim)
                #sim = self.calculate_cosine_similarity(applied_x, y).mean().item()
                score_list.append(sim)
                new_population.append(new_indv)
                if sim > best_score:
                    best_score = sim
                    best_indiv = new_indv
                    top_logits = logits
            # FSDP: broadcast the score from rank 0 so that all ranks perform the same selection
            # (prevents the score from differing per rank due to tiny differences in the FSDP forward)
            if torch.distributed.is_initialized():
                score_tensor = torch.tensor(score_list, dtype=torch.float64, device="cuda")
                torch.distributed.broadcast(score_tensor, src=0)
                score_list = score_tensor.tolist()
                # best_score/best_indiv are also synchronized to rank 0
                best_idx = score_list.index(max(score_list))
                if max(score_list) > best_score:
                    best_score = max(score_list)
                    best_indiv = new_population[best_idx]

            if best_score == before_best:
                patience += 1
            if patience >= 10:
                break
            before_best = best_score
            temp_history = {"d_model": [], "intermediate_size": [], "n_layers": [], "n_heads": []}
            for popul in new_population:
                if popul == {}:
                    continue
                for key in temp_history.keys():
                    temp_history[key].append(int(popul[key],2))
            for key in temp_history.keys():
                history[key].append(sum(temp_history[key]) / len(temp_history[key]))
            #print(f"best individual {best_indiv}")
            print(f"best score {best_score}")

            population = [fun for _, fun in sorted(zip(score_list, new_population), key=lambda pair: pair[0], reverse=True)]
            score_list.sort(reverse=True)
            k = max(2, int(0.05 * population_size))
            parent_candidate = population[:k]
            elite_score_list.append(sum(score_list[:k])/k)
            elite_cnt = max(2, int(0.05 * population_size))
            elites = deepcopy(population[:elite_cnt])
            for elite in elites:
                if elite.get("total_params"):
                    elite.pop("total_params")
                none_key = random.choice(self.MODULE_LIST)
                elite[none_key] = None
            offspring = self.make_new_gen(parent_candidate, population_size - elite_cnt)
            immigrant_cnt = max(1, int(0.03 * population_size))
            new_population = self.get_random_population(immigrant_cnt)
            self.population = elites + offspring[:-immigrant_cnt] + new_population
            # mutate_parent = random.choice(parent_candidate)
            # mutated_indiv = self.mutate_gen(mutate_parent)
            ## added mutation
            # self.intialize_population(10, mutant=True)

            # population[-1] = mutated_indiv
     

            iter_index.append(i)
            score_index.append(best_score)
        int_config_set = {k:int(v, 2) if isinstance(v, str) else v for k,v in best_indiv.items()}
        # int_config_set.pop("total_params")
        # print(f"after {iter_num} iterations, the best individual is {int_config_set}")
        # last_config = infer_missing_llama_config(self.target_params, **int_config_set)
        print(f"after {iter_num} iterations, score:{best_score}, the best individual is {int_config_set}")
        iters = list(range(len(history["d_model"])))
        for key, values in history.items():
            plt.figure()
            plt.plot(iters, values, marker='o')
            plt.title(key)
            plt.xlabel("iter")
            plt.ylabel(key)
            plt.grid(True)
            plt.savefig("./evolution_search{}_{}.png".format(formatted_current_time(), key))
            plt.close()

        plt.figure()
        plt.plot(list(range(len(elite_score_list))), elite_score_list, marker='o')
        plt.title("mean elite score")
        plt.xlabel("generation")
        plt.ylabel("score")
        plt.grid(True)
        plt.savefig("./evolution_elite_score_{}.png".format(formatted_current_time()))
        plt.close()
        
        ## compute the elite mean
        output_dict = {
            "best_indiv": best_indiv,
            "logits" : top_logits
        }
        return output_dict


    def apply_config(self, config_set, x, just_input=False):
        #config_set['n_heads'] = 16
        int_config_set = {k:int(v, 2) if isinstance(v, str) else v for k,v in config_set.items()}
        print(int_config_set)
        if just_input:
            last_config = config_set
        else:
            last_config = self.infer_missing_llama_config(self.target_params, **int_config_set)
        if last_config is None:
            return -320000.0, {}, None
        print(last_config)
        last_config = SimpleNamespace(**last_config)
        l0_module = self.model.model.l0_module
        target_module_cfg = l0_module.set_model_info(last_config, 3)
        origin_target_model_cfg = deepcopy(l0_module.target_model_info)
        l0_module.target_model_info = target_module_cfg

        # Change the mask sparsity of each module
        origin_target_mask_sizes = {}
        origin_eval_target_model = {}
        for mask_name in l0_module.masks:
            origin_target_mask_sizes[mask_name] = l0_module.masks[mask_name].target_mask_size
            origin_eval_target_model[mask_name] = l0_module.masks[mask_name].eval_target_model
            l0_module.masks[mask_name].eval_target_model = True
        change_target_mask_size(l0_module, last_config)

        # Call model forward in eval mode → l0_module.deterministic_z() uses the changed target_mask_size
        model_param = next(self.model.parameters())
        def change_device(tensor_dict, device="cuda", dtype=None):
            return {k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() 
                    else v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in tensor_dict.items()}
        x = change_device(x, model_param.device, model_param.dtype)
        print("start model forward")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=model_param.dtype):
            output = self.model(x)
        print("end model forward")

        # target_output is pre-computed in the callback and set on self.target_output

        # Restore the original state
        l0_module.target_model_info = origin_target_model_cfg
        for mask_name in l0_module.masks:
            l0_module.masks[mask_name].target_mask_size = origin_target_mask_sizes[mask_name]
            l0_module.masks[mask_name].eval_target_model = origin_eval_target_model[mask_name]

        bit_config = {k: format(v, 'b') for k, v in vars(last_config).items()}
        score = self.sim_func(output['logits'], self.target_output['logits'])
        print("SIM_FN:", getattr(self.sim_func, "__name__", str(self.sim_func)))
        logits = [output['logits'].detach().cpu(), self.target_output['logits']]
        return score, bit_config, logits

    def make_new_gen(self, parents_, population_size):
        population = []
        while len(population) < population_size:
            indiv = {}
            parents = []
            parents.append(random.choice(parents_))
            parents.append(random.choice(parents_))
                    
            for key in self.MODULE_LIST:
                if random.random() < 0.5:
                    gene = parents[0][key]
                else:
                    gene = parents[1][key]
                prob = random.random()
                if prob < 0.5:
                    gene = self.mutate_hamming_k(gene)
                else:
                    gene = self.mutate_bitflip(gene, 0.05)
                indiv[key] = gene
                # indiv[key] = self.shuffle_genetic(parents[0][key], parents[1][key])
            none_key = random.choice(self.MODULE_LIST)
            indiv[none_key] = None
            population.append(indiv)
        return population

    def make_new_gen_old(self, parents_, population_size):
        population = []
        while len(population) < population_size:
            indiv = {}
            parents = []
            parents.append(random.choice(parents_))
            parents.append(random.choice(parents_))
            for key in self.MODULE_LIST:
                # if random.random() < 0.5:
                #     gene = parents[0][key]
                # else:
                #     gene = parents[1][key]
                gene = self.shuffle_genetic(parents[0][key], parents[1][key])
                prob = random.random()
                if prob < 0.5:
                    gene = self.mutate_hamming_k(gene)
                else:
                    gene = self.mutate_bitflip(gene, 0.05)
                indiv[key] = gene
                # indiv[key] = self.shuffle_genetic(parents[0][key], parents[1][key])
            none_key = random.choice(self.MODULE_LIST)
            indiv[none_key] = None
            population.append(indiv)
        return population

    def mutate_hamming_k(self, bitstr: str, k: int = 1) -> str:
        L = len(bitstr)
        if L == 0:
            return bitstr
        if L >= 5:
            k = 2
        k = max(1, min(L, k))
        idxs = random.sample(range(L), k)
        bits = list(bitstr)
        for i in idxs:
            bits[i] = '0' if bits[i] == '1' else '1'
        s = ''.join(bits)
        if int(s, 2) == 0:
            s = s[:-1] + '1'
        return s

    def tournament_select(self, pop, scores, k=5):
        idxs = random.sample(range(len(pop)), k)
        best = max(idxs, key=lambda i: scores[i])
        return pop[best]

    # def make_new_gen(self, parents_, population_size):
    #     population = []
    #     while len(population) < population_size:
    #         indiv = {}
    #         parents = []
    #         parents.append(random.choice(parents_))
    #         parents.append(random.choice(parents_))
    #         none_module = {}
    #         parent1_none_key = [k for k, v in parents[0].items() if v is None]
    #         parent2_none_key = [k for k, v in parents[1].items() if v is None]
    #         # print(parent1_none_key)
    #         if parent1_none_key[0] == parent2_none_key[0]:
    #             none_module['same'] = parent1_none_key[0]
    #         else:
    #             none_module[0] = parent1_none_key[0]
    #             none_module[1] = parent2_none_key[0]
    #         chose_none = random.random()
    #         chose_none = round(chose_none)
    #         if none_module.get("same") is None:
    #             indiv[none_module[chose_none]] = None
    #             indiv[none_module[1-chose_none]] = parents[chose_none][none_module[1-chose_none]]

    #         else:
    #             indiv[none_module['same']] = None

    #         for key in self.MODULE_LIST:
    #             if key not in indiv.keys():
    #                 new_gene = self.shuffle_genetic(parents[0][key], parents[1][key])
    #                 indiv[key] = new_gene

    #         population.append(indiv)
    #     return population

    def mutate_bitflip(self, bitstr: str, p_mut: float = 0.01, avoid_all_zero: bool = True) -> str:
        bits = list(bitstr)
        for i in range(len(bits)):
            if random.random() < p_mut:
                bits[i] = '1' if bits[i] == '0' else '0'
        s = ''.join(bits)
        if avoid_all_zero and int(s, 2) == 0:  # turn on the last bit to avoid 0
            s = s[:-1] + '1'
        return s

    def shuffle_genetic(self, gene1, gene2):
        # print("##9955: ", gene1, gene2)
        min_len = min([len(gene1), len(gene2)])
        # random.seed(time.time())
                # If both parents are identical, introduce slight mutation to avoid clones

        rand_idx = random.randrange(1, min_len)
        # rand_idx = random.randint(0, len(gene1)-1)
        part_gene1 = gene1[:rand_idx]
        part_gene2 = gene2[rand_idx:]
        new_gene = part_gene1 + part_gene2

        p_mut = 0.01
        bits = list(new_gene)
        for i in range(len(bits)):
            if random.random() < p_mut:
                bits[i] = '1' if bits[i] == '0' else '0'
        new_gene = ''.join(bits)
        # random.shuffle(new_gene)
        # new_gene = "".join(new_gene)
        int_new_gene = int(new_gene, 2)
        if int_new_gene == 0:
            new_gene = new_gene[:-1] + '1'
        # print("new gene ", new_gene)
        return new_gene


if __name__=="__main__":
    # population = intialize_population(100, 1)
    # evolution(population, input_tensor, target, population_size=100)
    pass

