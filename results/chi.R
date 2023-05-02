data <- matrix(c(2896, 160, 5687, 37), nrow = 2, byrow = TRUE)
rownames(data) <- c("Prolific", "CrowdWorks")
colnames(data) <- c("Non-private", "Private")
chisq.test(data)

