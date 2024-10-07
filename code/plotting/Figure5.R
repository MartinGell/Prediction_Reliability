
library(data.table)
library(tidyverse)

#net_cols <- c("#0072B2", "#009E73", "#D55E00", "#56B4E9", "#E69F00", "#F0E442", "#CC79A7")
net_cols <- c("#009E73", "#E69F00", "#0072B2", "#F0E442", "#D55E00", "#56B4E9","#CC79A7")

plt_out = '/data/project/ukb_reliability_in_prediction/plots/Figures/'


# Empirical learning curves for UKB
d_subsample = read_csv('/data/project/ukb_reliability_in_prediction/input/test-pipe_ridgeCV-source_Schaefer400x17_nodenoise_UKB_5000_z-beh_UKB_5000_subs_FC_all_cogs_empirical_high_rel.csv')
colnames(d_subsample) <- c("250","403","652","1054","1704","2753","4450", "reliability")

rel <- read_csv('/data/project/ukb_reliability_in_prediction/res/UKB_reliability.csv')

d <- melt(setDT(d_subsample), id.vars = "reliability", variable.name = "sample",variable.factor = FALSE)
d$sample <- as.numeric(d$sample)
colnames(d)[colnames(d) == 'value'] = 'R2_mean'

df_sum <- d

d_subsample = read_csv('/data/project/ukb_reliability_in_prediction/input/test-pipe_ridgeCV-source_Schaefer400x17_nodenoise_UKB_5000_z-beh_UKB_5000_subs_FC_all_cogs_empirical_high_rel_sd.csv')
colnames(d_subsample) <- c("250","403","652","1054","1704","2753","4450", "reliability")
d <- melt(setDT(d_subsample), id.vars = "reliability", variable.name = "sample",variable.factor = FALSE)
d$sample <- as.numeric(d$sample)
colnames(d)[colnames(d) == 'value'] = 'R2_sd'

df_sum$R2_sd <- d$R2_sd

df_sum$reliability[df_sum$reliability == 'Fluid_intelligence_score-2.0'] <- 'Fluid intelligence'
df_sum$reliability[df_sum$reliability == 'Number_of_symbol_digit_matches_attempted-2.0'] <- 'Associative learning (SDST)'
df_sum$reliability[df_sum$reliability == 'TMT_B_duration_to_complete_log_trans-2.0'] <- 'Cognitive flexibility (TMT-B)'
df_sum$reliability[df_sum$reliability == 'Hand_grip_strength_mean_lr-2.0'] <- 'Grip strength'
df_sum$reliability[df_sum$reliability == 'Age_when_attended_assessment_centre-2.0'] <- 'Age'


pdf(file = paste0(plt_out, "Fig5.pdf"), width = 5.4, height = 3, useDingbats = FALSE)

ggplot(df_sum, aes(x = sample)) + 
  geom_point(aes(y = R2_mean, colour = reliability), size=1) +
  geom_line(aes(y = R2_mean, colour = reliability), size=0.8) +
  #geom_line(data = df_sum_sim, aes(x = sample, y = R2_mean), colour = 'black') +
  geom_ribbon(aes(y = R2_mean, ymin = R2_mean - 2*R2_sd, ymax = R2_mean + 2*R2_sd, fill = reliability), alpha = .1) +
  theme_classic() + 
  scale_x_continuous(limits = c(200,4600), breaks = c(250,500,1000,2000,3000,4000)) + 
  scale_y_continuous(limits = c(-0.071,0.419), breaks = c(-0.05,0,0.05,0.15,0.25,0.35)) +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5, hjust=0.3)) +
  scale_colour_manual(values = net_cols, name = "Behaviour",
                      labels = sort(unique(df_sum$reliability))) +
  scale_fill_manual(values = net_cols, name = "Behaviour",
                    labels = sort(unique(df_sum$reliability))) +
  xlab('Training set size') +
  ylab('Prediction Accuracy (R2)')

dev.off()

