import pandas as pd
from scipy.stats import wilcoxon

# 1. Load the experimental data
# Expected columns: vignette_id, model, prompt_condition, run_number, instrument, parsed_score, ground_truth_score
df = pd.read_csv('results.csv')

# 2. Calculate Item-Level Absolute Deviation
# This is the core of your Fidelity metric 
df['abs_deviation'] = (df['parsed_score'] - df['ground_truth_score']).abs()

# 3. Calculate MAD per Vignette, Model, and Condition
# Aggregates across items (9 for PHQ-9 / 7 for GAD-7) and across all 3 runs 
mad_summary = df.groupby(['vignette_id', 'model', 'prompt_condition', 'instrument'])['abs_deviation'].mean().reset_index()
mad_summary.rename(columns={'abs_deviation': 'MAD'}, inplace=True)

# 4. Calculate Mean Total Scores for Instrument-Level Fidelity
# First, sum scores for each individual run
run_totals = df.groupby(['vignette_id', 'model', 'prompt_condition', 'instrument', 'run_number'])['parsed_score'].sum().reset_index()

# Second, average those totals across the 3 runs [cite: 229]
total_score_summary = run_totals.groupby(['vignette_id', 'model', 'prompt_condition', 'instrument'])['parsed_score'].mean().reset_index()
total_score_summary.rename(columns={'parsed_score': 'LLM_Mean_Total'}, inplace=True)

# 5. Merge results for a final analysis table
final_results = pd.merge(mad_summary, total_score_summary, on=['vignette_id', 'model', 'prompt_condition', 'instrument'])

print("--- Fidelity Analysis Results ---")
print(final_results)

# 6. Statistical Comparison: GPT-4o vs Llama 3.3
# We compare the MAD of both models across the 8 vignettes (Label-Removed condition)
gpt_mads = final_results[(final_results['model'] == 'gpt-4o') & 
                         (final_results['prompt_condition'] == 'label_removed')].sort_values('vignette_id')['MAD'].values

llama_mads = final_results[(final_results['model'] == 'llama-3.3-70b') & 
                           (final_results['prompt_condition'] == 'label_removed')].sort_values('vignette_id')['MAD'].values

# Only run Wilcoxon if we have pairs for all vignettes
if len(gpt_mads) == len(llama_mads) and len(gpt_mads) > 0:
    stat, p_val = wilcoxon(gpt_mads, llama_mads)
    print(f"\n--- Statistical Significance (Wilcoxon Signed-Rank Test) ---")
    print(f"Comparison: GPT-4o MAD vs Llama 3.3 MAD")
    print(f"p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Result: Statistically Significant difference in fidelity.")
    else:
        print("Result: No statistically significant difference (p >= 0.05).")