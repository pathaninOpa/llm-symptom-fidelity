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
run_totals = df.groupby(['vignette_id', 'model', 'prompt_condition', 'instrument', 'run_number'])[['parsed_score', 'ground_truth_score']].sum().reset_index()

# Second, average those totals across the 3 runs 
total_score_summary = run_totals.groupby(['vignette_id', 'model', 'prompt_condition', 'instrument'])[['parsed_score', 'ground_truth_score']].mean().reset_index()
total_score_summary.rename(columns={'parsed_score': 'LLM_Mean_Total', 'ground_truth_score': 'GT_Mean_Total'}, inplace=True)

# 5. Merge results for a final analysis table
final_results = pd.merge(mad_summary, total_score_summary, on=['vignette_id', 'model', 'prompt_condition', 'instrument'])

print("--- Fidelity Analysis Results ---")
print(final_results)

# 6. Statistical Comparison: GPT-4o vs Llama 3.3
def run_wilcoxon(df_res, condition):
    gpt = df_res[(df_res['model'] == 'gpt4o') & (df_res['prompt_condition'] == condition)].sort_values(['vignette_id', 'instrument'])
    llama = df_res[(df_res['model'] == 'llama33') & (df_res['prompt_condition'] == condition)].sort_values(['vignette_id', 'instrument'])
    
    # Merge to ensure we only compare matching pairs (vignette + instrument)
    merged = pd.merge(gpt, llama, on=['vignette_id', 'instrument'], suffixes=('_gpt', '_llama'))
    
    if len(merged) > 1: # Wilcoxon needs at least 2 samples (and usually more for power)
        stat, p_val = wilcoxon(merged['MAD_gpt'], merged['MAD_llama'])
        print(f"\n--- Statistical Significance (Wilcoxon Signed-Rank Test) - {condition} ---")
        print(f"Comparison: GPT-4o MAD vs Llama 3.3 MAD")
        print(f"Pairs: {len(merged)}")
        print(f"p-value: {p_val:.4f}")
        if p_val < 0.05:
            print("Result: Statistically Significant difference in fidelity.")
        else:
            print("Result: No statistically significant difference (p >= 0.05).")
    else:
        print(f"\nInsufficient pairs found for condition: {condition}")

run_wilcoxon(final_results, 'label_removed')
run_wilcoxon(final_results, 'standard')
