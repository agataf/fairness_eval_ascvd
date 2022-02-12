# evaluating-fairness-ascvd-risk

This repository accompanies paper xxxx.

### Setup
```
pip install -r requirements.txt
```

TODO: add R requirements. Add Python version.

### Training
1. Generate a training grid
```
python create_grid_all.py
```

2. Run training, using flags
```
python train_model.py --config_path $CONFIG_PATH \
--result_path $RESULT_PATH \
--cohort_path $COHORT_PATH \
--logging_path $LOGGING_PATH \
--linear_layer \
--print_every 100 \
--save_outputs \
--logging_evaluate_by_group \
--stopping_fold_id $FOLD_ID \
--num_epochs 200 \
--early_stopping_patience 30   
```

Can also use the script

```
sh run_grid_training_thr.sh
```

3. Aggregate results
```
python eval/aggregate_results.py --experiment_name apr14_erm --model_type erm
python eval/aggregate_results.py --experiment_name apr14_thr --model_type eqodds_thr
python eval/aggregate_results.py --experiment_name apr14_mmd --model_type eqodds_mmd
```

4. Run recalibration
```
python eval/recalibrate_erm.py --experiment_name apr14_erm --new_model_type recalib_erm --new_model_id logx_logreg
```

### Evaluation
1. Run PCE inference
```
for EXPERIMENT_NAME in "original_pce" "revised_pce";
do



python $REPO_PATH/eval/pce_inference.py --base_path $BASE_PATH \
                                        --cohort_path $COHORT_PATH \
                                        --experiment_name $EXPERIMENT_NAME \
                                        --result_path ${RESULT_PATH}${EXPERIMENT_NAME}"/performance/"
done
```

2. Evaluate calibration of all models with bootstrap
```
python $REPO_PATH/eval/bootstrap_eval.py --base_path $BASE_PATH \
                                         --cohort_path $COHORT_PATH \
                                         --result_path $RESULT_PATH
```

3. Run the `eval/plot_performance_bymodel.ipynb` notebook to generate model-specific perfoamence plots. They will be saved under `eval/figures`.


5. Plot tradeoffs between ISGD(FNR), ISGD(FPR) and ISGD(TCE) by running `eval/fairness_violation_tradeoffs.ipynb`



If you find any issues, please email us at xxx.