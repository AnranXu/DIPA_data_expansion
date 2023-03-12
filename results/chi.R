data <- matrix(c(582, 913, 336, 1159), nrow = 2, byrow = TRUE)
rownames(data) <- c("Prolific", "CrowdWorks")
colnames(data) <- c("Non-private", "Private")
chisq.test(data)
