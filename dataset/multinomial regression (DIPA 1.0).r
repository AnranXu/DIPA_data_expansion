library(rstatix)
library(dplyr)
library(nnet)
# reference https://rpubs.com/malshe/214303
data <- read.csv(file = './mega_table.csv')
data <- data %>% convert_as_factor(extraversion, agreeableness, conscientiousness,neuroticism, openness, age, gender, platform, category, informationType)
formula <- "informativeness ~ gender + platform + informationType"
mlogit <- nnet::multinom(formula, data = data)
output <- summary(mlogit)
print(output)
z <- output$coefficients/output$standard.errors
p <- (1 - pnorm(abs(z), 0, 1))*2 # we are using two-tailed z test

informativeness2 <- rbind(output$coefficients[1,],output$standard.errors[1,],z[1,],p[1,])
rownames(informativeness2) <- c("Coefficient","Std. Errors","z stat","p value")
knitr::kable(informativeness2)
