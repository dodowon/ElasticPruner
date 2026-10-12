# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import math
import matplotlib.pyplot as plt
import math
from scipy.stats import beta as beta_dist

import numpy as np

def get_beta_peak_step(alpha, beta_param, total_steps=3200):
    """
    Takes the alpha and beta parameters of a Beta distribution and computes the step at which the actual peak occurs.
    
    Formula: mode = (alpha - 1) / (alpha + beta_param - 2)
          peak_step = mode * total_steps
    
    :param alpha: First shape parameter of the Beta distribution (a > 0)
    :param beta_param: Second shape parameter of the Beta distribution (b > 0)
    :param total_steps: Total number of steps (default: 3200)
    :return: dict {
        'mode': normalized peak position in the range [0, 1],
        'peak_step': peak position in actual step units,
        'shape': description of the distribution shape
    }
    """
    # Parameter validity check
    if alpha <= 0 or beta_param <= 0:
        raise ValueError(f"alpha and beta_param must both be greater than 0. "
                         f"Input: alpha={alpha}, beta_param={beta_param}")
    
    # Case 1: both greater than 1 → an interior mode exists (unimodal, normal case)
    if alpha > 1 and beta_param > 1:
        mode = (alpha - 1) / (alpha + beta_param - 2)
        peak_step = mode * total_steps
        shape = "unimodal (peak exists in the interior)"
    
    # Case 2: alpha < 1 and beta_param < 1 → bimodal (peaks at both ends)
    elif alpha < 1 and beta_param < 1:
        mode = None
        peak_step = None
        shape = "bimodal (peaks at both endpoints, no mode in the center)"
    
    # Case 3: alpha <= 1 and beta_param > 1 → maximum at the left end (0)
    elif alpha <= 1 and beta_param > 1:
        mode = 0.0
        peak_step = 0.0
        shape = "monotonically decreasing (maximum at x=0)"
    
    # Case 4: alpha > 1 and beta_param <= 1 → maximum at the right end (1)
    elif alpha > 1 and beta_param <= 1:
        mode = 1.0
        peak_step = float(total_steps)
        shape = "monotonically increasing (maximum at x=1)"
    
    # Case 5: alpha == 1 and beta_param == 1 → uniform distribution (no peak)
    else:  # alpha == 1 and beta_param == 1
        mode = None
        peak_step = None
        shape = "uniform (uniform distribution, no peak)"
    
    return {
        'mode': mode,
        'peak_step': peak_step,
        'shape': shape
    }


def find_beta_param_for_peak(alpha, target_peak_step, total_steps=3200):
    """
    Back-computes the beta_param value needed to match the desired peak position (target_peak_step).
    
    Formula: beta_param = (alpha - 1) / (target_peak / total_steps) - alpha + 2
    
    :param alpha: The alpha value to fix (requires a > 1)
    :param target_peak_step: The desired peak step position
    :param total_steps: Total number of steps
    :return: The beta_param value that produces that peak
    """
    if alpha <= 1:
        raise ValueError(f"alpha > 1 is required for the back-computation. Input: alpha={alpha}")
    
    if target_peak_step <= 0 or target_peak_step >= total_steps:
        raise ValueError(f"target_peak_step must be between (0, {total_steps}). "
                         f"Input: {target_peak_step}")
    
    target_mode = target_peak_step / total_steps
    beta_param = (alpha - 1) / target_mode - alpha + 2
    
    if beta_param <= 1:
        print(f"⚠️ Warning: the computed beta_param={beta_param:.3f} is 1 or less. "
              f"In this case no interior mode exists. Choose a larger alpha.")
    
    return beta_param

def get_dirichlet_schedule(total_steps=3200, peak_step=640, concentration=100):
    # Design each step's concentration parameter centered on peak_step
    # (Building the alpha vector in a Beta shape ends up similar to the Beta approach above)
    alphas = np.array([
        concentration * beta_dist.pdf(i/total_steps, 3.0, 9.0) + 0.1
        for i in range(total_steps)
    ])
    sample = np.random.dirichlet(alphas)
    return sample / sample.max()  # normalize the peak to 1.0

def get_beta_dist_value(current_step, total_steps=3200, peak_step=640,
                       alpha=1.5, beta_param=2.5):
    """
    Based on a Beta distribution. The influence of the later part is maintained much better than with a Gaussian.

    Parameter guide:
    - (2.0, 5.0): similar to Gaussian, slightly asymmetric
    - (1.5, 2.5): clearly maintains the influence of the later part (recommended starting point)
    - (1.3, 1.8): later part that is close to flat
    - The smaller β is, the thicker the tail becomes
    """
    x = current_step / total_steps

    if x <= 0 or x >= 1:
        return 0.0

    mode = (alpha - 1) / (alpha + beta_param - 2)
    pdf_max = beta_dist.pdf(mode, alpha, beta_param)

    return beta_dist.pdf(x, alpha, beta_param) / pdf_max

def get_normal_dist_value(current_step, total_steps=3200, peak_step=640):
    """
    Returns a value on the normal distribution curve based on the current step.
    
    :param current_step: The current step
    :param total_steps: Total number of steps (default: 3200)
    :param peak_step: The step that is both the mean and the maximum (1.0) (default: 640)
    :return: A normal distribution value between 0.0 and 1.0
    """
    mu = peak_step
    
    # Set the standard deviation (sigma) so that the value is nearly 0 at the target total step (3200)
    # (3200 - 640) / 3 = about 853.33
    sigma = (total_steps - peak_step) / 3.0 
    
    # Exception handling: avoid errors if the total step and the peak step are equal
    if sigma == 0:
        return 1.0 if current_step == peak_step else 0.0

    # Apply the Gaussian (normal distribution) formula (maximum value of 1.0)
    value = math.exp(-((current_step - mu) ** 2) / (2 * (sigma ** 2)))
    
    return value


if __name__=="__main__":
    

    # Generate a list of values from step 0 to 3200
    steps = list(range(3200))
    alpha = 1.5
    beta = 2.5
    #values = [get_normal_dist_value(step) for step in steps]
    values = [get_beta_dist_value(step, alpha=alpha, beta_param=beta) for step in steps]
    #schedule = get_dirichlet_schedule()
    #values = [schedule[step] for step in steps]
    peak = get_beta_peak_step(alpha, beta, total_steps=3200)
    peak = round(peak['peak_step'])
    
    table_str = f"Beta Distribution alpha={alpha}, beta={beta} (Peak at {peak})"
    #table_str = "Dirichlet distribution"
    # Draw the graph
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, label=table_str, color='blue')
    plt.axvline(x=peak, color='gray', linestyle='--') # line marking step 640

    # Give it a dark-theme look like the image you drew
    #plt.style.use('dark_background')
    plt.title("Normal Distribution Curve")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.xticks([0, peak, 3200])
    plt.legend()
    plt.show()
