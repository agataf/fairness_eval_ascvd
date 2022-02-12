python pce_inference.py --experiment_name original_pce
python pce_inference.py --experiment_name revised_pce

python eval/recalibrate_erm.py --experiment_name apr14_erm --new_model_type recalib_erm --new_model_id logx_logreg

python aggregate_results.py --experiment_name apr14_erm --model_type erm
python aggregate_results.py --experiment_name apr14_erm_recalib --model_type erm_recalib
python aggregate_results.py --experiment_name apr14_thr --model_type eqodds_thr



