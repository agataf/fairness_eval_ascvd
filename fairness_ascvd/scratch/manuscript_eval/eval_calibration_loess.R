library(dplyr)
library(ggplot2)


args.calib_model_id = 'loess'
args.base_path = '/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts'
args.experiment_name = 'apr14_thr'
args.cohort_path = '/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts/cohort/all_cohorts.csv'
args.n_bootstrap = 100
args.eval_fold = 'test'

aggregate_path = file.path(args.base_path, 'experiments', 
                              args.experiment_name, 'performance',
                              'all')
preds_path = file.path(aggregate_path, 'predictions.csv')
preds = read.csv(preds_path)

if (!('fold_id' %in% colnames(preds))) {
    preds = preds %>% mutate(fold_id=0)
}
if (!('model_id' %in% colnames(preds))) {
    preds = preds %>% mutate(model_id=0)
}

df_to_calibrate <- preds %>% filter(phase==args.eval_fold)

lin_calibs=c()
thr_calibs=c()
for (iter_idx in 1:args.n_bootstrap) {
    print(iter_idx)
    for (group in 1:4) {
        for (model_id in unique(preds$model_id)) {
            group_df = df_to_calibrate %>% filter(group==!!group, model_id==!!model_id)
            max_pred_prob = group_df$pred_probs %>% max
            for (fold_id in unique(group_df$fold_id)) {
                df <- group_df %>% filter(fold_id==!!fold_id)
                df = sample(df, nrow(df), replace = TRUE)
                loess_fit = loess(labels ~ pred_probs, df, weights)
                thr_calib = predict(loess_fit, data.frame(pred_probs = c(0.075, 0.2)), se = FALSE)
                lin_range = c(1e-15, seq(0.025, as.integer(max_pred_prob/0.025)*0.025, length.out = as.integer((max_pred_prob)/0.025)))
                lin_calib = predict(loess_fit, data.frame(pred_probs = lin_range), se = FALSE)

                lin_calib_frame = data.frame(pred_probs = lin_range,
                                             calibration_density = lin_calib) %>% 
                                  mutate(group = group, 
                                         fold_id = fold_id,
                                         phase = args.eval_fold,
                                         model_type = as.character(unique(preds$model_type)[1]),
                                         model_id = model_id)
                
                thr_calib_frame = data.frame(pred_probs = c(0.075, 0.2),
                                             calibration_density = thr_calib) %>% 
                                  mutate(group = group, 
                                         fold_id = fold_id,
                                         phase = args.eval_fold,
                                         model_type = as.character(unique(preds$model_type)[1]),
                                         model_id = model_id)


                lin_calibs = rbind(lin_calibs, lin_calib_frame)
                thr_calibs = rbind(thr_calibs, thr_calib_frame)
    
            }
        }
    } 
}

output_path = file.path(aggregate_path, 'calibration', args.calib_model_id)

if (!dir.exists(output_path)) {
    dir.create(output_path, recursive=TRUE)
}

write.csv(lin_calibs, file.path(output_path, 'calibration_sensitivity_test_raw.csv'))
write.csv(thr_calibs, file.path(output_path, 'calibration_sensitivity_thresholds_raw.csv'))