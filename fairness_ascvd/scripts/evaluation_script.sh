BASE_PATH=/labs/shahlab/projects/agataf/bmj_paper
REPO_PATH=$BASE_PATH/fairness_ascvd/fairness_ascvd

COHORT_PATH=$BASE_PATH/pce_data/all_cohorts_old.csv
RESULT_PATH=$BASE_PATH/experiments
EVAL_PATH=$BASE_PATH/final_evaluation

# Aggregate results from all folds of our experiments

for EXPERIMENT_NAME in "final_erm" "final_eq_odds_thr";
do
python $REPO_PATH/eval/aggregate_results.py --experiment_name $EXPERIMENT_NAME \
                                            --experiment_path ${RESULT_PATH}/${EXPERIMENT_NAME} \
                                            --model_type $EXPERIMENT_NAME
done


# Run recalibration on the ERM dataset
EXPERIMENT_NAME="final_erm"
TRANSFORM="logit"
python $REPO_PATH/eval/recalibrate_erm.py --experiment_name $EXPERIMENT_NAME \
                                          --new_model_type r_erm \
                                          --transform $TRANSFORM \
                                          --input_model_path ${RESULT_PATH}/${EXPERIMENT_NAME} \
                                          --result_path ${RESULT_PATH}/${EXPERIMENT_NAME}_rec_${TRANSFORM}

# Generate PCE and revised PCE predictions on our data.

for EXPERIMENT_NAME in "original_pce" "revised_pce";

do

python $REPO_PATH/eval/pce_inference.py --cohort_path $COHORT_PATH \
                                        --experiment_name $EXPERIMENT_NAME \
                                        --result_path ${RESULT_PATH}/${EXPERIMENT_NAME}
done

# Run bootstrapped evaluation.

python $REPO_PATH/eval/bootstrap_eval.py --exp_path $RESULT_PATH \
                                         --cohort_path $COHORT_PATH \
                                         --output_path $REPO_PATH/eval/figures_data \
                                         --n_boot 1000
                                         
# Make a cohort table.

python $REPO_PATH/eval/cohort_table.py --preds_path $BASE_PATH/pce_data/experiments/erm_predictions.csv \
                                         --cohort_path $COHORT_PATH \
                                         --result_path $REPO_PATH/eval/tables
                                         
# Make ISGD plot

python $REPO_PATH/eval/igsd_plot.py --input_file $REPO_PATH/eval/figures_data/IGSD_results.csv \
                                    --output_path $REPO_PATH/eval/figures
                                    
# Use R to generate the remaining plots

Rscript $REPO_PATH/eval/plot_performance_bymodel.R
