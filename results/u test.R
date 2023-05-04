library(coin)
data1 <- read.csv(file = './mega_table (strict).csv')
data2 <- read.csv(file = '../dataset/mega_table.csv')
GroupA <- data1$informativeness
GroupB <- data2$informativeness
GroupA <- GroupA - 3
GroupB <- GroupB - 3
mean(GroupA)
var(GroupA)
mean(GroupB)
var(GroupB)
wilcox.test(GroupA, GroupB)
g = factor(c(rep("GroupA", length(GroupA)), rep("GroupB", length(GroupB))))
v = c(GroupA, GroupB)
wilcox_test(v ~ g, alternative = "less", distribution = "asymptotic")



dataCrowdWorks <- data1[data1$platform=='CrowdWorks', ]
dataProlific <- data1[data1$platform=='Prolific', ]
GroupA <- dataCrowdWorks$informativeness
GroupB <- dataProlific$informativeness
GroupA <- GroupA - 3
GroupB <- GroupB - 3
mean(GroupA)
var(GroupA)
mean(GroupB)
var(GroupB)
wilcox.test(GroupA, GroupB)
g = factor(c(rep("GroupA", length(GroupA)), rep("GroupB", length(GroupB))))
v = c(GroupA, GroupB)
wilcox_test(v ~ g, alternative = "less", distribution = "asymptotic")

