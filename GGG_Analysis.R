library(ggplot2)
library(ggmosaic)
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(discrim)
library(themis)

train_ggg <- vroom("train.csv") %>% 
  mutate(type = factor(type),
         color = factor(color))
test_ggg <- vroom("test.csv")



# SVM
# ============================================================================
ggg_recipe <- recipe(type ~ ., data = train_ggg) %>%
  update_role(id, new_role = "ID") %>%
  step_other(color, threshold = 0.001, other = "other") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_numeric_predictors(), threshold = 0.90)


svmPoly <- svm_poly(degree = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# svmRadial <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")

# svmLinear <- svm_linear(cost = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")


ggg_workflow <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(svmLinear)

svm_params <- parameters(degree(), cost())

tuning_grid <- grid_space_filling(svm_params, size = 15)

folds <- vfold_cv(train_ggg, v = 5, repeats = 2)

CV_results <- ggg_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy, f_meas, recall, precision))
# roc_auc, f_meas, sens, recall, spec, precision, accuracy

bestTune <- CV_results %>% select_best(metric = "accuracy")

final_wf <- ggg_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_ggg)

ggg_predictions <- final_wf %>%
  predict(new_data = test_ggg) %>%
  bind_cols(test_ggg %>% select(id)) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(ggg_predictions, "svm_linear.csv", delim = ',')






# KNN
# ============================================================================

# ggg_recipe <- recipe(type ~ ., data = train_ggg) %>%
#   update_role(id, new_role = "ID") %>%
#   step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
#   step_dummy(all_nominal_predictors()) %>%
#   step_zv(all_predictors()) %>%
#   step_normalize(all_nominal_predictors())
# 
# 
# knn_model <- nearest_neighbor(neighbors = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kknn")
# 
# ggg_workflow <- workflow() %>%
#   add_recipe(ggg_recipe) %>%
#   add_model(knn_model)
# 
# 
# tuning_grid <- grid_regular(neighbors(),
#                             levels = 5) # L^2 total tuning possibilities
# 
# folds <- vfold_cv(train_ggg, v = 5, repeats = 2)
# 
# CV_results <- ggg_workflow %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(accuracy, f_meas))
# # roc_auc, f_meas, sens, recall, spec, precision, accuracy
# 
# bestTune <- CV_results %>% select_best(metric = "accuracy")
# 
# final_wf <- ggg_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = train_ggg)
# 
# ggg_predictions <- final_wf %>%
#   predict(new_data = test_ggg) %>%
#   bind_cols(test_ggg %>% select(id)) %>%
#   select(id, .pred_class) %>%
#   rename(type = .pred_class)
# 
# vroom_write(ggg_predictions, "knn.csv", delim = ',')





# NAIVE BAYES
# ============================================================================
# ggg_recipe <- recipe(type ~ ., data = train_ggg) %>%
#   update_role(id, new_role = "ID") %>%
#   step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
#   step_dummy(all_nominal_predictors()) %>%
#   step_normalize(all_numeric_predictors())
# 
# nb_model <- naive_Bayes(Laplace = tune(),
#                         smoothness = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes")
# 
# ggg_workflow <- workflow() %>%
#   add_recipe(ggg_recipe) %>%
#   add_model(nb_model)
# 
# tuning_grid <- grid_regular(Laplace(),
#                             smoothness(),
#                             levels = 5) # L^2 total tuning possibilities
# 
# folds <- vfold_cv(train_ggg, v = 5, repeats = 2)
# 
# CV_results <- ggg_workflow %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(accuracy))
# # roc_auc, f_meas, sens, recall, spec, precision, accuracy
# 
# bestTune <- CV_results %>% select_best(metric = "accuracy")
# 
# final_wf <- ggg_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = train_ggg)
# 
# ggg_predictions <- final_wf %>%
#   predict(new_data = test_ggg) %>%
#   bind_cols(test_ggg %>% select(id)) %>%
#   select(id, .pred_class) %>%
#   rename(type = .pred_class)
# 
# vroom_write(ggg_predictions, "naive_bayes.csv", delim = ',')




# RANDOM FORESTS
# ============================================================================
# ggg_recipe <- recipe(type ~ ., data = train_ggg) %>%
#   update_role(id, new_role = "ID") %>%
#   step_other(color, threshold = 0.001, other = "other") %>%
#   step_dummy(all_nominal_predictors()) %>%
#   step_zv(all_predictors())
# 
# 
# rf_model <- rand_forest(mtry = tune(),
#                         trees = 1000,
#                         min_n = tune()) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# 
# ggg_workflow <- workflow() %>%
#   add_recipe(ggg_recipe) %>%
#   add_model(rf_model)
# 
# rf_params <- parameters(mtry(),
#                         min_n()) %>% finalize(train_ggg)
# 
# tuning_grid <- grid_regular(rf_params,
#                             levels = 5) # L^2 total tuning possibilities
# 
# folds <- vfold_cv(train_ggg, v = 5, repeats = 2)
# 
# CV_results <- ggg_workflow %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(accuracy))
# # roc_auc, f_meas, sens, recall, spec, precision, accuracy
# 
# bestTune <- CV_results %>% select_best(metric = "accuracy")
# 
# final_wf <- ggg_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = train_ggg)
# 
# ggg_predictions <- final_wf %>%
#   predict(new_data = test_ggg) %>%
#   bind_cols(test_ggg %>% select(id)) %>%
#   select(id, .pred_class) %>%
#   rename(type = .pred_class)
# 
# vroom_write(ggg_predictions, "random_forests.csv", delim = ',')
