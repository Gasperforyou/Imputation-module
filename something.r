library(ggplot2)
setwd("C:\\Users\\Gasper\\Documents\\Faks\\3. Letnik\\2. semester\\DS\\orange-imputation-module")
dat1 = read.csv("correlation.csv", header = TRUE)
dat1 <- as.data.frame(dat1[-c(1, 2), -2])
dat <- as.matrix(dat1)
storage.mode(dat) <- "numeric"
d <- density(dat) # returns the density data 
plot(d, main="format distriucij", xlab="Spearman") # plots the results
