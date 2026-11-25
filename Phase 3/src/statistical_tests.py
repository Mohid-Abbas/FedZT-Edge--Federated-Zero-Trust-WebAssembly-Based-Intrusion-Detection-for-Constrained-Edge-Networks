"""
Statistical Analysis and Pareto Frontier Generator (Step 9)
Implements Shapiro-Wilk, Welch's t-test, Wilcoxon signed-rank test,
Holm-Bonferroni p-value correction, Cohen's d effect sizes, and 2D Pareto frontiers.
"""

import numpy as np
from scipy import stats

def compute_cohens_d(x, y):
    """Computes Cohen's d effect size between two independent/paired samples."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    if dof <= 0:
        return 0.0
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)

def holm_bonferroni_correction(p_values):
    """
    Applies Holm-Bonferroni step-down procedure to control family-wise error rate.
    Returns: list of adjusted p-values in original order.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0
    
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        adj = min(1.0, p_val * multiplier)
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = float(cum_max)
        
    return adjusted

def compare_configurations_statistically(sample_a, sample_b, config_name="FedZT vs Baseline"):
    """
    Runs normality check (Shapiro-Wilk) and performs Welch's t-test or Wilcoxon test.
    """
    sample_a = np.array(sample_a)
    sample_b = np.array(sample_b)
    
    # Shapiro-Wilk test for normality
    shapiro_a_p = stats.shapiro(sample_a).pvalue if len(sample_a) >= 3 else 1.0
    shapiro_b_p = stats.shapiro(sample_b).pvalue if len(sample_b) >= 3 else 1.0
    
    is_normal = (shapiro_a_p > 0.05) and (shapiro_b_p > 0.05)
    
    if is_normal:
        test_type = "Welch t-test"
        t_stat, p_val = stats.ttest_ind(sample_a, sample_b, equal_var=False)
    else:
        test_type = "Wilcoxon / Mann-Whitney"
        try:
            res = stats.mannwhitneyu(sample_a, sample_b, alternative='two-sided')
            t_stat = res.statistic
            p_val = res.pvalue
        except Exception:
            t_stat = 0.0
            p_val = 1.0
            
    cohen_d = compute_cohens_d(sample_a, sample_b)
    
    return {
        "comparison": config_name,
        "mean_a": float(np.mean(sample_a)),
        "std_a": float(np.std(sample_a)),
        "mean_b": float(np.mean(sample_b)),
        "std_b": float(np.std(sample_b)),
        "test_type": test_type,
        "statistic": float(t_stat),
        "raw_p_value": float(p_val),
        "cohens_d": float(cohen_d),
        "normality_a_p": float(shapiro_a_p),
        "normality_b_p": float(shapiro_b_p)
    }

def compute_pareto_frontier(points, maximize_x=True, maximize_y=False):
    """
    Extracts the non-dominated Pareto frontier from a list of (x, y) tuples or dicts.
    E.g. maximize Detection (x) while minimizing Energy/Latency (y).
    """
    sorted_pts = sorted(points, key=lambda p: (p[0] if maximize_x else -p[0]), reverse=True)
    pareto_pts = []
    
    best_y = -float('inf') if maximize_y else float('inf')
    
    for pt in sorted_pts:
        x, y = pt[0], pt[1]
        if maximize_y:
            if y > best_y:
                pareto_pts.append(pt)
                best_y = y
        else: # minimize y
            if y < best_y:
                pareto_pts.append(pt)
                best_y = y
                
    return pareto_pts
