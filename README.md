# fairness_eval_ascvd

Code to accompany "Evaluating algorithmic fairness in the presence of clinical guidelines: the case of atherosclerotic cardiovascular disease risk estimation. (in press) BMJ Health & Care Informatics, 2022.", by A Foryciarz, SR Pfohl, B Patel and NH Shah.

### Setup
```
pip install -e . 
```

### Cohort extraction
Use the `fairness_ascvd/scripts/cohort_extraction_script.sh` script to preprocess data and extract cohorts.

### Experiments
Use the `fairness_ascvd/scripts/training_script.sh.sh` script to define and run experiments.

### Postprocessing and aggregating results
Use the `fairness_ascvd/scripts/postprocess_script.sh` script to run recalibration, aggregate results, generate original PCE model predictions.

### Postprocessing and aggregating results
Use the `fairness_ascvd/scripts/evaluation_script.sh` script to generate bootstrapped evaluation, and reproduce tables and figures from the paper.

If you find any issues, please email us at agataf@stanford.edu.
