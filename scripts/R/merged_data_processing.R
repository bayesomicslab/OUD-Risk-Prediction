library(caret)

# Train-Test split
set.seed(42)

merged_original_fs$case = as.factor(merged_original_fs$case)

inTrain_og <- createDataPartition(y=merged_original_fs$case, p=.8, list = FALSE)
merged_original_fs.train <- merged_original_fs[inTrain_og, ]
merged_original_fs.test <- merged_original_fs[- inTrain_og, ]

# LOGISTIC REGRESSION -CLOGLOG ON VARIANTS DATA ONLY
cloglog <- glm(case ~ rs1436171_A + rs6551609_A + rs9285541_G + rs1294092_G + rs7219141_A + rs7211295_G,
                      family = binomial(link = "cloglog"), 
                      data = merged_original_fs.train)
#examining coefficients
summary(cloglog)

# Implementing K-folds cross-validation with auto hyperparameter selection
fitControl <- trainControl(method = 'repeatedcv', number = 10, repeats = 3, search = 'random')

model_logistic_cloglog <- train(case ~ rs1436171_A + rs6551609_A + rs9285541_G + rs1294092_G + rs7219141_A + rs7211295_G,
                                data = merged_original_fs.train,
                                method = 'glm',
                                family = binomial(link = "cloglog"),
                                trControl = fitControl)
model_logistic_cloglog$results
model_logistic_cloglog$finalModel

# Calculate false discovery rate
calc_fdr = function(fp, tp) {
  return(fp / (fp + tp))
}

pred = predict(model_logistic_cloglog, newdata = merged_original_fs.test)
confusionMatrix(data = pred, merged_original_fs.test$case)
print(paste('False Discovery Rate (Cloglog - Variants Data)', calc_fdr(2, 1)))