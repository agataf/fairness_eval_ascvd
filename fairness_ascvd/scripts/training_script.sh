BASE_PATH="/labs/shahlab/projects/agataf/bmj_paper"
REPO_PATH=$BASE_PATH"/fairness_ascvd/fairness_ascvd"

#RAW_DATA_PATH=$HOME_DIR"/pce_data/raw"

COHORT_PATH=$BASE_PATH"/pce_data/all_cohorts.csv"
DATA_PATH=$BASE_PATH

#COHORT_PATH1="/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts/cohort/all_cohorts.csv"

# generate grid
python $REPO_PATH/train/create_grid_all.py --data_path $DATA_PATH --experiment_name_prefix final


# ERM models
# TODO: update name

EXPERIMENT_NAME=final_erm

EXPERIMENT_PATH=$BASE_PATH"/experiments/"$EXPERIMENT_NAME

CONFIG_ID=0
for FOLD_ID in {1..10}
do

echo "config: "$CONFIG_ID
echo "fold: "$FOLD_ID

CONFIG_PATH=$EXPERIMENT_PATH"/config/"$CONFIG_ID".yaml"
RESULT_PATH=$EXPERIMENT_PATH"/performance/"$CONFIG_ID".yaml/"$FOLD_ID
LOGGING_PATH=$RESULT_PATH"/training_log.log"

python $REPO_PATH/train/train_model.py --config_path $CONFIG_PATH --result_path $RESULT_PATH \
                                        --cohort_path $COHORT_PATH --logging_path $LOGGING_PATH \
                                        --linear_layer --print_every 50 --save_model_weights \
                                        --save_outputs --logging_evaluate_by_group \
                                        --stopping_fold_id $FOLD_ID --num_epochs 1 --early_stopping_patience 30 
done


# Equalized odds models 

EXPERIMENT_NAME=final_eq_odds_thr

EXPERIMENT_PATH=$BASE_PATH"/experiments/"$EXPERIMENT_NAME

for FOLD_ID in {1..10}
do
for CONFIG_ID in {0..3}
do

echo "config: "$CONFIG_ID
echo "fold: "$FOLD_ID

CONFIG_PATH=$EXPERIMENT_PATH"/config/"$CONFIG_ID".yaml"
RESULT_PATH=$EXPERIMENT_PATH"/performance/"$CONFIG_ID".yaml/"$FOLD_ID
LOGGING_PATH=$RESULT_PATH"/training_log.log"
                                        
python $REPO_PATH/train/train_model.py --config_path $CONFIG_PATH --result_path $RESULT_PATH \
                                        --cohort_path $COHORT_PATH --logging_path $LOGGING_PATH \
                                        --linear_layer --print_every 100 --save_model_weights \
                                        --save_outputs --logging_evaluate_by_group \
                                        --stopping_fold_id $FOLD_ID --num_epochs 1 # \ --early_stopping_patience 30
done
done
                         
