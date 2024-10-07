
library(tidyverse)

df <- read.csv('/data/project//ukb_reliability_in_prediction/text_files/increase_rel_HGS.csv')

plt <- ggplot(df, aes(Reliability,test_R2,colour=Beh)) + 
  geom_point() +
  geom_line() +
  geom_text(aes(label = Time_point), colour = 'black', nudge_y = -0.002, nudge_x = 0.004) +
  theme_classic() +
  xlim(c(0.8,0.88)) +
  ylim(c(0.28,0.34)) + 
  ylab('Prediction Accuracy (R2)')

ggsave('/data/project//ukb_reliability_in_prediction/plots/UKB/increase_rel_HGS_avg.png', plt, width=4.5, height=4)

