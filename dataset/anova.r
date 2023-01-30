library(rstatix)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(multcompView)
library(multcomp)
library(ez)
library(car)
data <- read.csv(file = './for_anova (mycat).csv')
data <- data %>% convert_as_factor(ID,my_cat,informativeness, reason)
#homogneity <- data %>% group_by(category) %>% levene_test(importance ~ category)
#print(homogneity)
anova <- aov(sharing ~ informativeness + Error(ID/informativeness), data = data)
summary(anova)

