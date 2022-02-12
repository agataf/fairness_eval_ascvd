#!/bin/sh

HOME_DIR="/labs/shahlab/projects/agataf/bmj_paper"
REPO_PATH=$HOME_DIR"/fairness_ascvd/fairness_ascvd"

BASE_PATH=$HOME_DIR
RAW_DATA_PATH=$HOME_DIR"/pce_data/raw"
COHORT_PATH=$BASE_PATH"/pce_data"

var_dir=$COHORT_PATH'/cohort_extraction/variables'
coh_dir=$COHORT_PATH'/cohort_extraction/cohorts'
final_cohort_file=$COHORT_PATH'/all_cohorts.csv'


# TODO: include original paths?

for study in 'mesa' 'fhs_os' 'chs' 'aric' 'jhs' 'cardia'
do
    python ${REPO_PATH}/preproc/${study}.py --input_dir $RAW_DATA_PATH --output_dir $var_dir
    
done

python ${REPO_PATH}/preproc/cohort_definition_aggregate.py --input_dir $var_dir --output_dir $coh_dir --print_info 1

python ${REPO_PATH}/preproc/aggregate_and_split.py --cohort_path $coh_dir --output_path $final_cohort_file


