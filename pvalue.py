import scipy.stats as stats
import numpy as np
np.random.seed(42)  # For reproducibility

dataset = "mars" #mars,moon
adapt_list = np.loadtxt("bal_acc_list_real_"+dataset+"_adapt.txt")
adaptlin_list = np.loadtxt("bal_acc_list_real_"+dataset+"_adaptlin.txt")
full_list = np.loadtxt("bal_acc_list_real_"+dataset+"_full.txt")

#compute differences
list1 = adapt_list
list2 = full_list
differences = list1 - list2
# print(differences)

# Check normality using Shapiro-Wilk test
_, p_normality = stats.shapiro(differences)
# print(p_normality)

if p_normality > 0.05:
    # If differences are normal, use paired t-test
    t_stat, p_value = stats.ttest_rel(list1, list2)
    print("Using paired t-test.")
else:
    # If not normal, use Wilcoxon signed-rank test
    t_stat, p_value = stats.wilcoxon(list1, list2)
    print("Using Wilcoxon signed-rank test.")

# Output p-value
print(f"P-value: {p_value:.10f}")

# Interpretation
if p_value < 0.05:
    print("The difference is statistically significant.")
else:
    print("No significant difference between the two algorithms.")
