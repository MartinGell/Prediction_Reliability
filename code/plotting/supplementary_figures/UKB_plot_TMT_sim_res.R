
library(data.table)
library(tidyverse)

#net_cols <- c("#0072B2", "#009E73", "#D55E00", "#56B4E9", "#E69F00", "#F0E442", "#CC79A7")
net_cols <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


# TMT
# A
d_subsample = read_csv('/data/project/ukb_reliability_in_prediction/input/learning_curve_collected/ridgeCV_zscore-source_Schaefer400x17_nodenoise_UKB_5000_z-beh_TMT_B_duration_to_complete_wnoise_mean_all.csv')

d <- melt(setDT(d_subsample), id.vars = "reliability", variable.name = "sample",variable.factor = FALSE)
d$sample <- as.numeric(d$sample)

d_subsample %>% group_by(reliability) %>% summarise_at(colnames(d_subsample)[1:7], list(name = mean))
d_subsample %>% group_by(reliability) %>% summarise_at(colnames(d_subsample)[1:7], list(name = sd))

d$reliability <- d$reliability*d$reliability*0.78


plt = ggplot(d, aes(reliability,value,colour=as.factor(sample))) + geom_point(size = 1.5, alpha = 0.3) +
  geom_smooth(method = lm, formula = y ~ splines::bs(x, 3),se = FALSE) +
  theme_classic() + ylim(c(-0.05,0.1)) + 
  ylab('Prediction Accuracy (R2)') +
  xlab('Reliability') +
  scale_colour_manual(values = net_cols, name = 'Training set size')
ggsave(paste0('/data/project/ukb_reliability_in_prediction/plots/UKB/subsample_res/square_rel_TMT_B_R2_wnoise_rel_07_to_095.png'), plt, width=4, height=3)



# B
df_sum = read_csv('/data/project//ukb_reliability_in_prediction/input/UKB_TMT_mean_sd_samples.csv')
df_sum$reliability = round(df_sum$reliability*df_sum$reliability*0.78, digits = 2)

df_sum$reliability = as.factor(df_sum$reliability)


plt2 = ggplot(df_sum, aes(x = sample)) + 
  geom_point(aes(y = R2_mean, colour = reliability), size=1) +
  geom_line(aes(y = R2_mean, colour = reliability), size=0.8) +
  geom_ribbon(aes(y = R2_mean, ymin = R2_mean - 2*R2_sd, ymax = R2_mean + 2*R2_sd, fill = reliability), alpha = .1) +
  theme_classic() + 
  scale_x_continuous(limits = c(200,4600), breaks = c(250,500,1000,2000,3000,4000)) + 
  scale_y_continuous(limits = c(-0.05,0.1), breaks = c(-0.05,0,0.05,0.1)) +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5, hjust=0.3)) +
  scale_colour_manual(values = net_cols, name = "Reliability",
                      labels = unique(df_sum$reliability)) +
  scale_fill_manual(values = net_cols, name = "Reliability",
                    labels = unique(df_sum$reliability)) +
  xlab('Training set size') +
  ylab('Prediction Accuracy (R2)')
ggsave(paste0('/data/project/ukb_reliability_in_prediction/plots/UKB/subsample_res/square_rel_TMT_B_R2_wnoise_rel_038_to_78_B.png'), plt2, width=4.3, height=3)




