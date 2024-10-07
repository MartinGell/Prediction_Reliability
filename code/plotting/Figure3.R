
library(data.table)
library(tidyverse)


### Figure 3 A ###
# AGE

#net_cols <- c("#0072B2", "#009E73", "#D55E00", "#56B4E9", "#E69F00", "#F0E442", "#CC79A7")
net_cols <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

plt_out = '/data/project/ukb_reliability_in_prediction/plots/Figures/'

# data
d_subsample = read_csv('/data/project/ukb_reliability_in_prediction/input/learning_curve_collected/ridgeCV_zscore-source_Schaefer400x17_nodenoise_UKB_5000_z-beh_Age_when_attended_assessment_centre_wnoise_mean_all_new.csv')
colnames(d_subsample) <- c("250","403","652","1054","1704","2753","4450", "reliability")

d <- melt(setDT(d_subsample), id.vars = "reliability", variable.name = "sample",variable.factor = FALSE)
d$sample <- as.numeric(d$sample)

d$reliability = d$reliability*d$reliability


# plot
pdf(file = paste0(plt_out, "Fig3_A.pdf"), width = 4.5, height = 3, useDingbats = FALSE)

ggplot(d, aes(reliability,value,colour=as.factor(sample))) + 
  geom_point(size = 1.5, alpha = 0.3) +
  geom_smooth(method = lm, se = FALSE) + #, formula = y ~ splines::bs(x, 3),se = FALSE) +
  theme_classic() + ylim(c(-0.1,0.4)) + 
  xlim(c(0.2,1.0)) +
  ylab('Prediction Accuracy (R2)') +
  xlab('Reliability') +
  scale_colour_manual(values = net_cols, name = 'Training set size') +
  theme(axis.text = element_text(size = 10.5),
        axis.title = element_text(size = 10.5),
        legend.text = element_text(size = 11),
        legend.title = element_text(size = 10.5))

dev.off()




### FIGURE 3 B ###
rm(list = ls())

# HGS
net_cols <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

plt_out = '/data/project/ukb_reliability_in_prediction/plots/Figures/'

# data
d_subsample = read_csv('/data/project/ukb_reliability_in_prediction/input/learning_curve_collected/ridgeCV_zscore-source_Schaefer400x17_nodenoise_UKB_5000_z-beh_Hand_grip_strength_mean_lr_wnoise_mean_all.csv')

d <- melt(setDT(d_subsample), id.vars = "reliability", variable.name = "sample",variable.factor = FALSE)
d$sample <- as.numeric(d$sample)

# adjust reliability
d$reliability <- d$reliability*d$reliability*0.93


#plot
pdf(file = paste0(plt_out, "Fig3_B.pdf"), width = 4.5, height = 3, useDingbats = FALSE)

ggplot(d, aes(reliability,value,colour=as.factor(sample))) + geom_point(size = 1.5, alpha = 0.3) +
  geom_smooth(method = lm, se = FALSE) + #, formula = y ~ splines::bs(x, 3),se = FALSE) +
  theme_classic() + ylim(c(-0.1,0.4)) + 
  ylab('Prediction Accuracy (R2)') +
  xlab('Reliability') +
  xlim(c(0.2,1.0)) +
  scale_colour_manual(values = net_cols, name = 'Training set size') +
  theme(axis.text = element_text(size = 10.5),
        axis.title = element_text(size = 10.5),        
        legend.text = element_text(size = 11),
        legend.title = element_text(size = 10.5))

dev.off()