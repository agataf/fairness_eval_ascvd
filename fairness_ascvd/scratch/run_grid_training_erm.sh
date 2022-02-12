EXPERIMENT_NAME=jul29_erm

DATA_PATH="/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts/cohort"
BASE_PATH="/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts"

COHORT_PATH=$DATA_PATH"/all_cohorts.csv"
CONFIG_ID=0
for FOLD_ID in {1..10}
do

echo "config: "$CONFIG_ID
echo "fold: "$FOLD_ID

CONFIG_PATH=$BASE_PATH"/experiments/"$EXPERIMENT_NAME"/config/"$CONFIG_ID".yaml"

RESULT_PATH=$BASE_PATH"/experiments/"$EXPERIMENT_NAME"/performance/"$CONFIG_ID".yaml/"$FOLD_ID
LOGGING_PATH=$RESULT_PATH"/training_log.log"

python train_model.py --config_path $CONFIG_PATH --result_path $RESULT_PATH \
                                        --cohort_path $COHORT_PATH --logging_path $LOGGING_PATH \
                                        --linear_layer --print_every 50 --save_model_weights \
                                        --save_outputs --logging_evaluate_by_group \
                                        --stopping_fold_id $FOLD_ID --num_epochs 200 --early_stopping_patience 30 # --disable_metric_logging
done




                         

                         
