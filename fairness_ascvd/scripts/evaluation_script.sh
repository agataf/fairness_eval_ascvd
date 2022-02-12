HOME_DIR="/labs/shahlab/projects/agataf/bmj_paper"
REPO_PATH=$HOME_DIR"/fairness_ascvd/fairness_ascvd"
# TODO: rename to DATA_PATH
# BASE_PATH=$HOME_DIR"/data/cohorts/pooled_cohorts"
BASE_PATH=$HOME_DIR"/data_final"
#COHORT_PATH=$BASE_PATH"/cohort/all_cohorts.csv"
COHORT_PATH=$BASE_PATH"/cohort/all_cohorts_old.csv"

RESULT_PATH=$BASE_PATH"/experiments"
EVAL_PATH=$BASE_PATH"/final_evaluation"

# STEP xx. Aggregate results from all folds of our experiments

for EXPERIMENT_NAME in "apr14_erm" "scratch_thr";
do
python $REPO_PATH/eval/aggregate_results.py --experiment_name $EXPERIMENT_NAME \
                                            --experiment_path ${RESULT_PATH}/${EXPERIMENT_NAME} \
                                            --model_type $EXPERIMENT_NAME
done


                                     #   --experiment_name $EXPERIMENT_NAME \

# STEP xx. Run recalibration on the ERM dataset
# TODO: finish
EXPERIMENT_NAME="apr14_erm"
TRANSFORM="logit"
python $REPO_PATH/eval/recalibrate_erm.py --experiment_name $EXPERIMENT_NAME \
                                          --new_model_type r_erm \
                                          --transform $TRANSFORM \
                                          --input_model_path ${RESULT_PATH}/${EXPERIMENT_NAME} \
                                          --result_path ${RESULT_PATH}/${EXPERIMENT_NAME}_rec_${TRANSFORM}

# STEP xx. Generate PCE and revised PCE predictions on our data.

for EXPERIMENT_NAME in "original_pce" "revised_pce";

do

python $REPO_PATH/eval/pce_inference.py --cohort_path $COHORT_PATH \
                                        --experiment_name $EXPERIMENT_NAME \
                                        --result_path ${RESULT_PATH}/${EXPERIMENT_NAME}
done

# STEP xx. Run bootstrapped evaluation.

python $REPO_PATH/eval/bootstrap_eval.py --exp_path $RESULT_PATH \
                                         --cohort_path $COHORT_PATH \
                                         --output_path $EVAL_PATH \
                                         --n_boot 1 #000
