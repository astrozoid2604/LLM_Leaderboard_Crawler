# Pipeline script to facilitate Stratified Sampling on top-K models in HuggingFace's LLM leaderboard

The script crawls top-K models from [HuggingFace's LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) by cloning the latest LLM evaluation results from the leaderboard, traversing through all models result json files, and computing the evaluation metrics of interest with full dataset and sampled dataset (default sampling percentage is 25% of full dataset).

## One-time Environment Setup

```bash
conda deactivate; jupyter kernelspec uninstall tf; conda env remove --name tf; conda env create -f env.yml; conda activate tf; python -m ipykernel install --user --name tf --display-name "tf"; python -m spacy download en_core_web_sm
```

## Command to run Jupyter Notebook (Can run multiple times)

The output file of below command will be saved at ./LLM_Leaderboard_Crawler/LLM_Leaderboard_Crawler.html

```bash
jupyter nbconvert --execute --to html LLM_Leaderboard_Crawler/LLM_Leaderboard_Crawler.ipynb
```

## Detailed explanation of how the pipeline script works:

All the codes are contained within 1 notebook file: [LLM_LeaderBoard_Crawler.ipynb](LLM_LeaderBoard_Crawler.ipynb)
- First, we run `get_top_models()` function in which we specify how many top best models from [HuggingFace's LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) we want to inspect.
  - We clone the results/ from this [LINK](https://huggingface.co/datasets/open-llm-leaderboard/results). If there's already an existing directory with same name, skip this cloning process
  - Next, we get the list of all result JSON file from each model because we would like to calculate the average score shown in the [HuggingFace's LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) using `calculate_avg_score()` function by averaging 6 LLM benchmarks, namely, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K, and 57-domain MMLU.
  - In `get_top_models()` function, we have 2 data structures, namely, `aug_model_result_dict`, and `top_models`
    - Explanation for `aug_model_result_dict`:
      - `aug_model_result_dict` has outer dictionary key of `aug`, inner dictionary key of `model`, and inner dictionary value of list of tuples. 
      - In simplicity, the data structure of `aug_model_result_dict` is a nested dictionary with list of tuples as the value
      - Each tuple corresponds to different evaluation timestamps for same `aug` and same `model`
      - Each tuple comprises of directory for timestamp, and the average score from aforementioned 6 LLM benchmarks
      - Since this data structure is pretty massive as K increases, `aug_model_result_dict` is deleted at the end of `aug_model_result_dict` function to prevent memory leak.
    - Explanation for `top_models`:
      - This structure is a list of tuples
      - Each tuple contains the average score in percentage, and a string comprises of `aug`/`model`/`timedir` (i.e. timestamp at which best evaluation result is achieved by a particular `aug` and `model`)
      - `top_models` is the output of  `get_top_models()` function
  - Once we get output of `get_top_models()` function, we pass it along with a list of metrics of interest as arguments to `tabulate_score()` function which will output our eventual DataFrame containing mean score of each top-K model from [HuggingFace's LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) and each metrics of interest using full dataset and sampled dataset
    - Data structure called `metrics_dict` helps us to capture all necessary elements to create eventual output DataFrame
      - `metrics_dict` is in the for of metric of interest as dictionary key, and 2D-list each element of which comprises of `aug`/`model`, full-data score, and sampled-data score.
      - A few corner cases include inexistence of parquet dataset under `timedir` directory within respective repository https://huggingface.co/datasets/open-llm-leaderboard/details_`aug`__`model`. On such cases, we skip them from being included in the eventual output DataFrame
    - `final_dict` is the last data structure under `tabulate_score()` function which serves as origin data to form eventual output DataFrame
      - `final_dict` is a dictionary with `aug`/`model` as dictionary key, and a 2-element list for full-data score and sampled-data score.

 
## Code needed to run and expected output DataFrame

There are 2 scenarios to run this pipeline:
  - Expecting ranking of predetermined list of models from LLM leaderbord
  - Expecting ranking of top 10 models in LLM leaderboard


### Non-Empty LLM_list + sampling_method='random'
```python
LLM_list        = ['meta-llama/Llama-2-7b', 'facebook/opt-6.7b', 'databricks/dolly-v2-13b'] 
benchmark_name  = 'truthfulqa'
sampling_method = 'random'

full_ranking, subset_ranking = get_subset_ranking(benchmark_name, LLM_list, sampling_method)
```

`full_ranking` dictionary result:
```python
{1: [38.98, 'mc2_meta-llama/Llama-2-7b-hf'],
 2: [35.12, 'mc2_facebook/opt-6.7b']}
```

`subset_ranking` dictionary result:
```python
{1: [40.18, 'mc2_meta-llama/Llama-2-7b-hf'],
 2: [36.67, 'mc2_facebook/opt-6.7b']}
```


### Empty LLM_list (default top 10 model from LLM Leaderboard) + sampling_method='random'
```python
LLM_list        = []
benchmark_name  = 'truthfulqa'
sampling_method = 'random'
num_top_models  = 10

full_ranking, subset_ranking = get_subset_ranking(benchmark_name, LLM_list, sampling_method, num_top_models=10)
```

`full_ranking` dictionary result:
```python
{1: [81.5, 'mc2_cloudyu/Mixtral_7Bx2_MoE_DPO'],
 2: [78.02, 'mc2_yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B'],
 3: [76.73, 'mc2_Sao10K/SOLAR-10.7B-NahIdWin'],
 4: [75.38, 'mc2_yunconglong/10.7Bx2_DPO_200'],
 5: [74.95, 'mc2_ycros/BagelMIsteryTour-8x7B'],
 6: [74.57, 'mc2_RubielLabarta/LogoS-7Bx2-MoE-13B-v0.1'],
 7: [73.55, 'mc2_one-man-army/UNA-34Beagles-32K-bf16-v1'],
 8: [73.37, 'mc2_fblgit/LUNA-SOLARkrautLM-Instruct'],
 9: [73.27, 'mc2_sumo43/SOLAR-10.7B-Instruct-DPO-v1.0']}
```

`subset_ranking` dictionary result:
```python
{1: [79.64, 'mc2_cloudyu/Mixtral_7Bx2_MoE_DPO'],
 2: [78.15, 'mc2_Sao10K/SOLAR-10.7B-NahIdWin'],
 3: [77.32, 'mc2_ycros/BagelMIsteryTour-8x7B'],
 4: [76.54, 'mc2_yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B'],
 5: [75.37, 'mc2_one-man-army/UNA-34Beagles-32K-bf16-v1'],
 6: [75.2, 'mc2_RubielLabarta/LogoS-7Bx2-MoE-13B-v0.1'],
 7: [74.6, 'mc2_yunconglong/10.7Bx2_DPO_200'],
 8: [74.01, 'mc2_fblgit/LUNA-SOLARkrautLM-Instruct'],
 9: [73.76, 'mc2_sumo43/SOLAR-10.7B-Instruct-DPO-v1.0']}
```

### Empty LLM_list (default top 10 model from LLM Leaderboard) + sampling_method='topic'
```python
LLM_list        = []
benchmark_name  = 'truthfulqa'
sampling_method = 'topic'
num_top_models  = 10

full_ranking, subset_ranking = get_subset_ranking(benchmark_name, LLM_list, sampling_method, num_top_models=10)
```

`full_ranking` dictionary result:
```python
{1: [81.5, 'mc2_cloudyu/Mixtral_7Bx2_MoE_DPO'],
 2: [78.02, 'mc2_yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B'],
 3: [76.73, 'mc2_Sao10K/SOLAR-10.7B-NahIdWin'],
 4: [75.38, 'mc2_yunconglong/10.7Bx2_DPO_200'],
 5: [74.95, 'mc2_ycros/BagelMIsteryTour-8x7B'],
 6: [74.57, 'mc2_RubielLabarta/LogoS-7Bx2-MoE-13B-v0.1'],
 7: [73.55, 'mc2_one-man-army/UNA-34Beagles-32K-bf16-v1'],
 8: [73.37, 'mc2_fblgit/LUNA-SOLARkrautLM-Instruct'],
 9: [73.27, 'mc2_sumo43/SOLAR-10.7B-Instruct-DPO-v1.0']}
```

`subset_ranking` dictionary result:
```python
{1: [80.09, 'mc2_cloudyu/Mixtral_7Bx2_MoE_DPO'],
 2: [77.45, 'mc2_yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B'],
 3: [74.96, 'mc2_yunconglong/10.7Bx2_DPO_200'],
 4: [74.54, 'mc2_RubielLabarta/LogoS-7Bx2-MoE-13B-v0.1'],
 5: [74.12, 'mc2_Sao10K/SOLAR-10.7B-NahIdWin'],
 6: [73.86, 'mc2_ycros/BagelMIsteryTour-8x7B'],
 7: [72.86, 'mc2_one-man-army/UNA-34Beagles-32K-bf16-v1'],
 8: [71.13, 'mc2_sumo43/SOLAR-10.7B-Instruct-DPO-v1.0'],
 9: [69.89, 'mc2_fblgit/LUNA-SOLARkrautLM-Instruct']}
```


### Aggregated MMLU + Top 50 models, cherry-picked every 20 best models

![xlim_mmlu_50model_20modelsamplinginterval_0.01initialpct_0.01pctincrement.JPG](visualization_results/xlim_mmlu_50model_20modelsamplinginterval_0.01initialpct_0.01pctincrement.JPG)

### TruthfulQA + Top 50 models, cherry-picked every 20 best models

![xlim_truthfulqa_50model_20modelsamplinginterval_0.01initialpct_0.01pctincrement](visualization_results/xlim_truthfulqa_50model_20modelsamplinginterval_0.01initialpct_0.01pctincrement.JPG)

### GSM8K + Top 50 models, cherry-picked every 20 best models

![xlim_gsm8k_50model_20modelsamplinginterval_0.01initialpct_0.01pctincrement](visualization_results/xlim_gsm8k_50model_20modelsamplinginterval_0.01initialpct_0.01pctincrement.JPG)

### HellaSwag + Top 50 models, cherry-picked every 20 best models

![xlim_hellaswag_50model_20modelsamplinginterval_0.01initialpct_0.01pctincrement](visualization_results/xlim_hellaswag_50model_20modelsamplinginterval_0.01initialpct_0.01pctincrement.JPG)

### Winogrande + Top 50 models, cherry-picked every 20 best models

![xlim_winogrande_50model_20modelsamplinginterval_0.01initialpct_0.01pctincrement](visualization_results/xlim_winogrande_50model_20modelsamplinginterval_0.01initialpct_0.01pctincrement.JPG)

### ARC + Top 50 models, cherry-picked every 20 best models

![xlim_arc_50model_20modelsamplinginterval_0.01initialpct_0.01pctincrement](visualization_results/xlim_arc_50model_20modelsamplinginterval_0.01initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_government_and_politics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_government_and_politics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_government_and_politics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-abstract_algebra + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-abstract_algebra_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-abstract_algebra_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-anatomy + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-anatomy_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-anatomy_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-astronomy + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-astronomy_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-astronomy_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-business_ethics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-business_ethics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-business_ethics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-clinical_knowledge + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-clinical_knowledge_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-clinical_knowledge_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-college_biology + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-college_biology_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-college_biology_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-college_chemistry + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-college_chemistry_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-college_chemistry_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-college_computer_science + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-college_computer_science_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-college_computer_science_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-college_mathematics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-college_mathematics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-college_mathematics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-college_medicine + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-college_medicine_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-college_medicine_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-college_physics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-college_physics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-college_physics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-computer_security + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-computer_security_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-computer_security_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-conceptual_physics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-conceptual_physics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-conceptual_physics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-econometrics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-econometrics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-econometrics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-electrical_engineering + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-electrical_engineering_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-electrical_engineering_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-elementary_mathematics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-elementary_mathematics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-elementary_mathematics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-formal_logic + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-formal_logic_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-formal_logic_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-global_facts + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-global_facts_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-global_facts_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_biology + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_biology_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_biology_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_chemistry + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_chemistry_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_chemistry_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_computer_science + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_computer_science_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_computer_science_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_european_history + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_european_history_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_european_history_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_geography + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_geography_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_geography_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_macroeconomics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_macroeconomics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_macroeconomics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_mathematics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_mathematics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_mathematics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_microeconomics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_microeconomics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_microeconomics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_physics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_physics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_physics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_psychology + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_psychology_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_psychology_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_statistics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_statistics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_statistics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_us_history + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_us_history_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_us_history_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-high_school_world_history + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-high_school_world_history_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-high_school_world_history_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-human_aging + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-human_aging_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-human_aging_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-human_sexuality + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-human_sexuality_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-human_sexuality_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-international_law + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-international_law_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-international_law_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-jurisprudence + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-jurisprudence_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-jurisprudence_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-logical_fallacies + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-logical_fallacies_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-logical_fallacies_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-machine_learning + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-machine_learning_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-machine_learning_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-management + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-management_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-management_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-marketing + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-marketing_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-marketing_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-medical_genetics + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-medical_genetics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-medical_genetics_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-miscellaneous + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-miscellaneous_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-miscellaneous_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-moral_disputes + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-moral_disputes_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-moral_disputes_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-moral_scenarios + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-moral_scenarios_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-moral_scenarios_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-nutrition + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-nutrition_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-nutrition_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-philosophy + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-philosophy_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-philosophy_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-prehistory + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-prehistory_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-prehistory_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-professional_accounting + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-professional_accounting_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-professional_accounting_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-professional_law + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-professional_law_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-professional_law_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-professional_medicine + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-professional_medicine_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-professional_medicine_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-professional_psychology + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-professional_psychology_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-professional_psychology_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-public_relations + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-public_relations_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-public_relations_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-security_studies + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-security_studies_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-security_studies_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-sociology + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-sociology_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-sociology_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-us_foreign_policy + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-us_foreign_policy_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-us_foreign_policy_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-virology + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-virology_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-virology_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)

### hendrycksTest-world_religions + Top 50 models, cherry-picked every 20 best models

![xlim_hendrycksTest-world_religions_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement](visualization_results/xlim_hendrycksTest-world_religions_50model_20modelsamplinginterval_0.21initialpct_0.01pctincrement.JPG)
