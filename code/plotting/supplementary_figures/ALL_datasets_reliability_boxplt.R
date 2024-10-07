
library(tidyverse)
library("ggpubr")


HCP = read.csv('/home/mgell/Work/reliability/res/HCP_YA_reliability_accuracy.csv')
HCP = HCP[,c(-1,-17,-18)]
HCP$dataset = 'HCP-YA'



UKB = read.csv('/home/mgell/Work/reliability/res/UKB_reliability_accuracy.csv')
UKB$dataset = 'UKB'
colnames(UKB)[c(14,15)] = c('reliability_r', 'reliability_icc2')



abcd = read.csv('/home/mgell/Work/reliability/res/ABCD_reliability_accuracy.csv')
abcd = abcd[,c(-1,-17)]
abcd$dataset = 'ABCD'



df = rbind(HCP,UKB,abcd)

pltr = ggplot(df, aes(dataset,reliability_r,fill=dataset)) + 
  geom_boxplot(alpha=0.7, outlier.size = 0) +
  #geom_point() +
  geom_point(position = position_jitter(width = 0.1), alpha = 0.55) + 
  theme_classic() + ylab('Reliability (r)') +
  ylim(c(0,1.0))

#ggsave('/home/mgell/Work/reliability/plots/reliability_r_all_datasets.png')

summary(HCP$reliability_r)
summary(UKB$reliability_r)
summary(abcd$reliability_r)
summary(df$reliability_r)


plticc = ggplot(df, aes(dataset,reliability_icc2,fill=dataset)) + 
  geom_boxplot(alpha=0.7, outlier.size = 0) + 
  geom_point(position = position_jitter(width = 0.1), alpha = 0.55) + 
  theme_classic() + ylab('Reliability (ICC)') +
  ylim(c(0,1.0))

#ggsave('/home/mgell/Work/reliability/plots/reliability_icc_all_datasets.png')

summary(HCP$reliability_icc2)
summary(UKB$reliability_icc2)
summary(abcd$reliability_icc2)
summary(df$reliability_icc)


fig <- ggarrange(
  pltr, plticc, labels = c("a", "b"),
  common.legend = TRUE, legend = "bottom"
)

ggsave('/home/mgell/Work/reliability/plots/reliability_all_datasets.png', plot = fig, width = 4.5, height = 5)

