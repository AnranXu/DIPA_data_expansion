library(nnet)
library(rstatix)
library(dplyr)
# reference https://rpubs.com/malshe/214303
data <- read.csv(file = './image_wise_regression_table.csv')
#data <- data %>% convert_as_factor(age, gender, platform, ifPrivacy)
formula <- "ifPrivacy ~ gender + age + nationality + frequency"
log.model <- glm(formula = formula, family = "binomial", data = data)
summary(log.model)
summary(log.model)$coefficients
#odd ratio
exp(coefficients(log.model))
# 95% CI
confint.default(log.model)
