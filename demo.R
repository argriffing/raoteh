
rao.data <- read.table('timeseries.data')
forward.data <- read.table('forward.timeseries.data')

pdf(file='out.pdf')

hist(rao.data$V1, xlim=c(0, 60),
     main='Rao-Teh sampling',
     xlab='Num of event pairs')

hist(forward.data$V1, xlim=c(0, 60),
     main='forward sampling',
     xlab='Num of event pairs')

dev.off()
