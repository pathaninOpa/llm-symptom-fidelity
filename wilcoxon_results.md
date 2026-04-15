--- Fidelity Analysis Results ---
   vignette_id    model prompt_condition instrument       MAD  LLM_Mean_Total  GT_Mean_Total
0           V1    gpt4o    label_removed       PHQ9  0.111111       25.000000           24.0
1           V1    gpt4o         standard       PHQ9  0.111111       25.000000           24.0
2           V1  llama33    label_removed       PHQ9  0.222222       26.000000           24.0
3           V1  llama33         standard       PHQ9  0.222222       26.000000           24.0
4           V2    gpt4o    label_removed       PHQ9  0.296296       16.666667           14.0
5           V2    gpt4o         standard       PHQ9  0.222222       16.000000           14.0
6           V2  llama33    label_removed       PHQ9  0.370370       15.333333           14.0
7           V2  llama33         standard       PHQ9  0.444444       16.000000           14.0
8           V3    gpt4o    label_removed       GAD7  1.142857       18.000000           10.0
9           V3    gpt4o         standard       GAD7  1.047619       17.333333           10.0
10          V3  llama33    label_removed       GAD7  1.000000       17.000000           10.0
11          V3  llama33         standard       GAD7  1.000000       17.000000           10.0
12          V4    gpt4o    label_removed       GAD7  0.285714       21.000000           19.0
13          V4    gpt4o         standard       GAD7  0.285714       21.000000           19.0
14          V4  llama33    label_removed       GAD7  0.285714       21.000000           19.0
15          V4  llama33         standard       GAD7  0.285714       21.000000           19.0
16          V5    gpt4o    label_removed       GAD7  0.857143       16.000000           10.0
17          V5    gpt4o    label_removed       PHQ9  0.444444       14.000000           10.0
18          V5    gpt4o         standard       GAD7  0.857143       16.000000           10.0
19          V5    gpt4o         standard       PHQ9  0.444444       14.000000           10.0
20          V5  llama33    label_removed       GAD7  0.857143       16.000000           10.0
21          V5  llama33    label_removed       PHQ9  0.666667       16.000000           10.0
22          V5  llama33         standard       GAD7  0.857143       16.000000           10.0
23          V5  llama33         standard       PHQ9  0.666667       16.000000           10.0
24          V6    gpt4o    label_removed       GAD7  0.285714       21.000000           19.0
25          V6    gpt4o    label_removed       PHQ9  0.666667       21.000000           15.0
26          V6    gpt4o         standard       GAD7  0.285714       21.000000           19.0
27          V6    gpt4o         standard       PHQ9  0.666667       21.000000           15.0
28          V6  llama33    label_removed       GAD7  0.285714       21.000000           19.0
29          V6  llama33    label_removed       PHQ9  1.000000       24.000000           15.0
30          V6  llama33         standard       GAD7  0.285714       21.000000           19.0
31          V6  llama33         standard       PHQ9  0.962963       23.666667           15.0
32          V7    gpt4o    label_removed       GAD7  1.714286       17.000000            5.0
33          V7    gpt4o         standard       GAD7  1.714286       17.000000            5.0
34          V7  llama33    label_removed       GAD7  1.571429       16.000000            5.0
35          V7  llama33         standard       GAD7  1.619048       16.333333            5.0
36          V8    gpt4o    label_removed       PHQ9  0.000000        9.000000            9.0
37          V8    gpt4o         standard       PHQ9  0.000000        9.000000            9.0
38          V8  llama33    label_removed       PHQ9  0.222222       11.000000            9.0
39          V8  llama33         standard       PHQ9  0.185185       10.666667            9.0

--- Statistical Significance (Wilcoxon Signed-Rank Test) - label_removed ---
Comparison: GPT-4o MAD vs Llama 3.3 MAD
Pairs: 10
p-value: 0.2812
Result: No statistically significant difference (p >= 0.05).

--- Statistical Significance (Wilcoxon Signed-Rank Test) - standard ---
Comparison: GPT-4o MAD vs Llama 3.3 MAD
Pairs: 10
p-value: 0.0781
Result: No statistically significant difference (p >= 0.05).