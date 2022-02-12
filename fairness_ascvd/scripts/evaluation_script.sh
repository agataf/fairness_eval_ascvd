BASE_PATH=/labs/shahlab/projects/agataf/bmj_paper
REPO_PATH=$BASE_PATH/fairness_ascvd/fairness_ascvd

COHORT_PATH=$BASE_PATH/pce_data/all_cohorts_old.csv
RESULT_PATH=$BASE_PATH/experiments
EVAL_PATH=$BASE_PATH/final_evaluation

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
