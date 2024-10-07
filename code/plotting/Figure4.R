
library(data.table)
library(tidyverse)


### Figure 4 ###
# AGE

#net_cols <- c("#0072B2", "#009E73", "#D55E00", "#56B4E9", "#E69F00", "#F0E442", "#CC79A7")
net_cols <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

plt_out = '/data/project/ukb_reliability_in_prediction/plots/Figures/'

# data
df_sum = read_csv('/data/project/ukb_reliability_in_prediction/input/UKB_age_mean_sd_samples.csv')
df_sum$reliability = df_sum$reliability*df_sum$reliability

df_sum$reliability = as.factor(df_sum$reliability)

pdf(file = paste0(plt_out, "Fig4.pdf"), width = 4.5, height = 3, useDingbats = FALSE)

ggplot(df_sum, aes(x = sample)) + 
  geom_point(aes(y = R2_mean, colour = reliability), size=1) +
  geom_line(aes(y = R2_mean, colour = reliability), size=0.8) +
  geom_ribbon(aes(y = R2_mean, ymin = R2_mean - 2*R2_sd, ymax = R2_mean + 2*R2_sd, fill = reliability), alpha = .1) +
  theme_classic() + 
  scale_x_continuous(limits = c(200,4600), breaks = c(250,500,1000,2000,3000,4000)) + 
  scale_y_continuous(limits = c(-0.05,0.4), breaks = c(-0.05,0,0.05,0.15,0.25,0.35)) +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5, hjust=0.3)) +
  scale_colour_manual(values = net_cols, name = "Reliability",
                      labels = unique(df_sum$reliability)) +
  scale_fill_manual(values = net_cols, name = "Reliability",
                    labels = unique(df_sum$reliability)) +
  xlab('Training set size') +
  ylab('Prediction Accuracy (R2)')

dev.off()



