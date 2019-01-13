library(ggplot2)
setwd("C:\\Users\\Gasper\\Documents\\Faks\\3. Letnik\\2. semester\\DS\\Diplomsko delo\\sinteticna\\Average")
dat1 = read.csv("correlation_c.csv", header = TRUE)
dat1 <- as.data.frame(dat1[-c(1, 2), -2])
dat <- as.matrix(dat1)
storage.mode(dat) <- "numeric"
d <- density(dat) # returns the density data 
plot(d, main="Cell distribution", xlab="Spearman") # plots the results
