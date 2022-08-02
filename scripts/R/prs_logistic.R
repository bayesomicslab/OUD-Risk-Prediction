library(caret)

# Train-Test split
set.seed(42)

merged_prs$case = as.factor(merged_prs$case)

inTrain_merged <- createDataPartition(y=merged_prs$case, p=.8, list = FALSE)
merged_prs.train <- merged_prs[inTrain_merged, ]
merged_prs.test <- merged_prs[- inTrain_merged, ]

# LOGISTIC REGRESSION - LOGIT
logit_risk <- glm(case ~ PRS_risk, family = 'binomial',
            data = merged_prs.train)
logit_merged <- glm(case ~ PRS_risk + PRS_depression + PRS_neuro, family = 'binomial', 
                      data = merged_prs.train)
#examining coefficients
summary(logit_risk)
summary(logit_merged)

# LOGISTIC REGRESSION -CLOGLOG
cloglog_risk <- glm(case ~ PRS_risk, family = binomial(link = "cloglog"), 
            data = merged_prs.train)

cloglog_merged <- glm(case ~ PRS_risk + PRS_depression + PRS_neuro, family = binomial(link = "cloglog"), 
                    data = merged_prs.train)
#examining coefficients
summary(cloglog_risk)
summary(cloglog_merged)

# Likelihood ratio test
anova(cloglog_merged, cloglog_risk, test='Chisq')
anova(logit_merged, logit_risk, test='Chisq')

# Implementing K-folds cross-validation
fit.control <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)

model_logistic_cloglog <- train(case ~ PRS_risk + PRS_depression + PRS_neuro,
                         data = merged_prs.train,
                         method = 'glm',
                         family = binomial(link = "cloglog"),
                         trControl = fit.control)
model_logistic_cloglog_2 <- train(case ~ PRS_risk,
                                data = merged_prs.train,
                                method = 'glm',
                                family = binomial(link = "cloglog"),
                                trControl = fit.control)
#examining coefficients
summary(model_logistic_cloglog)
summary(model_logistic_cloglog_2)

# Training accuracy
model_logistic_cloglog$results
model_logistic_cloglog$finalModel

model_logistic_cloglog_2$results
model_logistic_cloglog_2$finalModel


model_logistic_logit <- train(case ~ PRS_risk + PRS_depression + PRS_neuro,
                               data = merged_prs.train,
                               method = 'glm',
                               family = binomial(link = "logit"),
                               trControl = fit.control)
model_logistic_logit_2 <- train(case ~ PRS_risk,
                              data = merged_prs.train,
                              method = 'glm',
                              family = binomial(link = "logit"),
                              trControl = fit.control)
#examining coefficients
summary(model_logistic_logit)
summary(model_logistic_logit_2)

# Training accuracy
model_logistic_logit$results
model_logistic_logit$finalModel

model_logistic_logit_2$results
model_logistic_logit_2$finalModel

# Calculate false discovery rate
calc_fdr = function(fp, tp) {
  return(fp / (fp + tp))
}

# Obtaining test accuracy
pred_logit = predict(model_logistic_logit, newdata = merged_prs.test)
confusionMatrix(data = pred_logit, merged_prs.test$case)
print(paste('False Discovery Rate (Logit - all PRS)', calc_fdr(2, 1)))

pred_logit_2 = predict(model_logistic_logit_2, newdata = merged_prs.test)
confusionMatrix(data = pred_logit_2, merged_prs.test$case)
print(paste('False Discovery Rate (Logit - risk PRS)', calc_fdr(1, 1)))

pred = predict(model_logistic_cloglog, newdata = merged_prs.test)
confusionMatrix(data = pred, merged_prs.test$case)
print(paste('False Discovery Rate (Cloglog - all PRS)', calc_fdr(2, 1)))

pred_2 = predict(model_logistic_cloglog_2, newdata = merged_prs.test)
confusionMatrix(data = pred_2, merged_prs.test$case)
print(paste('False Discovery Rate (Cloglog - risk PRS)', calc_fdr(1, 1)))

# Plotting
predicted.data <- data.frame(probability.of.op=clog$fitted.values,
                             op=risk_prs$case)
predicted.data <- predicted.data[
  order(predicted.data$probability.of.op, decreasing = FALSE),] #sorting
predicted.data$rank <- 1:nrow(predicted.data)

library(ggplot2)
library(cowplot)

p <- ggplot(data = predicted.data, aes(x=rank, y=probability.of.op))
p + geom_point(aes(color=op), alpha=1, shape=4, stroke=2)
xlab('Index')
ylab('Predicted probability of being an opioid user')
ggsave('opioid_use_probabilities.pdf')

# Logit Regression
summary(logit <- glm(case ~ PRS, family = binomial(link = "logit"), 
                    na.action = na.omit, data = risk_prs))

c(clog$aic, logit$aic)