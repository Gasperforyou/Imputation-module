library(ggplot2)
setwd("C:\\Users\\Gasper\\Documents\\Faks\\3. Letnik\\2. semester\\DS\\Testi\\Average_biological")
dat1 = read.csv("correlation_n.csv", header = TRUE)
dat1 <- as.data.frame(dat1[-c(1, 2), -2])
dat <- as.matrix(dat1)
storage.mode(dat) <- "numeric"
d <- density(dat, bw = 0.05) # returns the density data
plot(d, main="Cell distribution", xlab="Spearman") # plots the results
