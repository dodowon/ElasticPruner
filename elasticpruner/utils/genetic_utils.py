# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

"""Collection of general-purpose utility functions to make the GA search efficient/predictable.

The functions in this module are pure functions not tied to any particular GA class,
and are later imported and used by several GA classes (e.g., `AirGeneticSparsity*`
in `genetic_contraint_w_head.py`).

The architecture (individual) is represented as a dict, and the chromosome keys are the same as
`MODULE_LIST` in `genetic_contraint_w_head.py`:

    {"d_model": ..., "intermediate_size": ..., "n_layers": ..., "n_heads": ...}

Each value takes one of the following three forms.
    - Binary string (e.g., "10110"): decoded with int(v, 2).
    - Integer (e.g., 22).
    - None: a "dropped" axis that will be inferred later to match target_params.

The functions in this module preserve the form of the input value (binary/integer/None) as is.
"""

import random
from copy import deepcopy

# Architecture chromosome keys (same as MODULE_LIST in genetic_contraint_w_head.py)
MODULE_LIST = ["d_model", "intermediate_size", "n_layers", "n_heads"]

# The "multiple" constraint that each axis must satisfy.
#   - d_model(hidden)      : multiple of 128
#   - intermediate_size    : multiple of 64
#   - n_heads(head)        : even (multiple of 2)
# Keys not listed here (e.g., n_layers, total_params, vocab_size) are passed through
# unchanged without applying constraints.
ARCH_MULTIPLE_CONSTRAINTS = {
    "d_model": 128,
    "intermediate_size": 64,
    "n_heads": 2,
}


def _is_binary_gene(value):
    """Whether the gene value is in binary string form."""
    return isinstance(value, str)


def _decode_gene(value):
    """Decode the gene value to an integer (binary interpretation if it is a binary string)."""
    if _is_binary_gene(value):
        return int(value, 2)
    return int(value)


def _encode_gene(int_value, as_binary):
    """Encode an integer gene value back to its original form (binary string/integer)."""
    if as_binary:
        return format(int_value, "b")
    return int(int_value)


def _round_to_multiple(value, multiple):
    """Snap value to the nearest multiple of `multiple` (round-half-up).

    To preserve the value, rounding is used instead of floor,
    and the minimum lower bound is set to `multiple` so it never becomes 0 or negative.
    """
    if multiple <= 0:
        return value
    n = (value + multiple // 2) // multiple  # round-half-up
    snapped = n * multiple
    if snapped < multiple:
        snapped = multiple
    return snapped


def apply_architecture_constraints(arch):
    """Apply consistency constraints to the architecture dictionary and snap it to a valid architecture.

    Constraints:
        - d_model(hidden)   : multiple of 128
        - intermediate_size : multiple of 64
        - n_heads(head)     : even

    Keys that are not constrained (n_layers, total_params, etc.) and None values are left as is.
    Preserves the form of the input values (binary string/integer) and returns a dict of the same form.

    Args:
        arch: Architecture dictionary. Values are binary string/integer/None.

    Returns:
        A new dictionary with the same keys/form as the input that satisfies the constraints.
    """
    constrained = {}
    for key, value in arch.items():
        # Axes that are None or not constrained pass through unchanged.
        if value is None or key not in ARCH_MULTIPLE_CONSTRAINTS:
            constrained[key] = value
            continue

        multiple = ARCH_MULTIPLE_CONSTRAINTS[key]
        as_binary = _is_binary_gene(value)
        int_val = _decode_gene(value)
        snapped = _round_to_multiple(int_val, multiple)
        constrained[key] = _encode_gene(snapped, as_binary)

    return constrained


def compute_gradual_size(origin_module_arch, curr_generation_count, target_pruning_ratio):
    """Compute the per-axis maximum mutation width (gradual_size) that shrinks gradually with generations.

    As generations progress, the mutation width is halved (2^-generation), so that the search is wide
    early on and fine-tuned later. Scheduling the mutation width this way makes the
    search space converge to the valid range and makes the search time predictable.

        ratio = target_pruning_ratio * 2^(-curr_generation_count)
        gradual_size[key] = round(origin_module_arch[key] * ratio)

    If a computed value is less than 0, it is replaced with 0.

    Args:
        origin_module_arch: Original model architecture dict
            (e.g., {"n_layers": 32, "d_model": 4096, ...}). Values are integers or binary strings.
        curr_generation_count: Current generation number.
        target_pruning_ratio: Target pruning ratio.

    Returns:
        A per-axis maximum mutation width dict (integers, 0 or greater). Used as is as the
        `gradual_size` argument of `gradual_mutation`.
    """
    ratio = target_pruning_ratio * (2.0 ** (-curr_generation_count))

    gradual_size = {}
    for key, value in origin_module_arch.items():
        if value is None:
            gradual_size[key] = 0
            continue
        base = _decode_gene(value)
        delta = int(round(base * ratio))
        gradual_size[key] = max(0, delta)  # replace with 0 if less than 0
    return gradual_size


def _mutate_gene(value, max_delta):
    """Gradually mutate a single gene by an integer in the range [-max_delta, +max_delta].

    delta includes 0, so the gene can "stay put". The value form (binary/integer) is preserved,
    and None genes (dropped axes) are left as is without mutation. The result is clamped to 1 or more.
    """
    if value is None or max_delta <= 0:
        return value

    as_binary = _is_binary_gene(value)
    int_val = _decode_gene(value)
    delta = random.randint(-max_delta, max_delta)  # negative/0/positive are all possible
    new_val = max(1, int_val + delta)
    return _encode_gene(new_val, as_binary)


def gradual_mutation(elites, population_size, gradual_size, apply_constraints=True):
    """Generate the next-generation population by elite-based gradual mutation.

    - Elites (the top individuals of the current generation) are carried over to the next generation without mutation.
    - The remaining slots are filled by choosing elites as parents and randomly moving each chromosome by +/-
      (or staying put) the amount defined in `gradual_size`. Limiting the mutation width this way
      makes the search space converge to the valid range and the search time predictable.
    - The final population size is `population_size`.

    Args:
        elites: List of elite individuals (dicts) to carry over.
        population_size: Size of the next-generation population to generate.
        gradual_size: Per-axis maximum mutation width dict (e.g., {"d_model": 256, "n_layers": 2}).
            Axes without a key are not mutated (delta=0).
        apply_constraints: If True, re-apply the consistency (multiple/even) constraints with
            `apply_architecture_constraints` after mutation.

    Returns:
        List of the next-generation population (length == population_size).
    """
    if not elites:
        raise ValueError("gradual_mutation: elites must contain at least one individual.")

    gradual_size = gradual_size or {}

    # 1) Carry over the elites as they are (deepcopy so they are independent of the parents).
    next_population = [deepcopy(elite) for elite in elites]

    # If the elites alone already fill or exceed the capacity, truncate and return.
    if len(next_population) >= population_size:
        return next_population[:population_size]

    # 2) Fill the remaining slots with elite-based gradual mutation.
    while len(next_population) < population_size:
        parent = random.choice(elites)
        child = {}
        for key, value in parent.items():
            max_delta = gradual_size.get(key, 0)
            child[key] = _mutate_gene(value, max_delta)

        if apply_constraints:
            child = apply_architecture_constraints(child)

        next_population.append(child)

    return next_population
