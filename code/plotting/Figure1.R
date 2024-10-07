
library(tidyverse)



### FIGURE 1A ###
# HCP A plot results for Age prediction - violin plot

beh = 'Age'

plt_out = '/home/mgell/Work/reliability/plots/Figures/'


f = 'ridgeCV_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_interview_age_wnoise_all.csv'

d = read_csv(paste0('/home/mgell/Work/reliability/input/',f))
d$test_MAE = d$test_MAE*-1

empirical = d[1,]
d = d[-1,]

metrics = colnames(d)
rel_max = max(d$reliability)
rel_min = min(d$reliability)

d$reliability = d$reliability*d$reliability

# for saving fig
pdf(file = paste0(plt_out, "Fig1_A.pdf"), width = 3.5, height = 3, useDingbats = FALSE)

# Plot
ggplot(d,aes(reliability,test_R2,group=as.factor(reliability))) + 
  geom_violin(color = 'deeppink3', width=0.1, position=position_dodge(width = 0)) + 
  geom_boxplot(colour = "grey", width=0.01, position=position_dodge(width = 0),outlier.size = 0) +
  theme_classic() +
  ylim(c(-0.1,0.8)) + 
  scale_x_continuous(limits = c(0.2,1.05), breaks = c(seq(0.2,1.0,0.2))) +
  xlab('Reliability') + 
  ylab('Age Prediction Accuracy (R2)') +
  theme(legend.position="none") +
  geom_point(aes(empirical$reliability,empirical$test_R2), size = 3, colour = 'skyblue3') +
  theme(axis.text = element_text(size = 10.5),
        axis.title = element_text(size = 10.5))

dev.off()






### FIGURE 1B ###
# HCP A plot results for prediction of simulated variables together
rm(list = ls())

#net_cols <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") # original order
net_cols <- c("#0072B2", "#009E73", "#D55E00") # reordering for 3 colours only



plt_in = '/home/mgell/Work/reliability/input/'
plt_out = '/home/mgell/Work/reliability/plots/Figures/'


f = 'ridgeCV_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_HCP_A_total_wnoise_all.csv'
d_t = read_csv(paste0(plt_in,f))

d_t$reliability = d_t$reliability * d_t$reliability * 0.9

f = 'ridgeCV_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_HCP_A_motor_wnoise_all.csv'
d_m = read_csv(paste0(plt_in,f))

d_m$reliability = d_m$reliability * d_m$reliability * 0.93

f = 'ridgeCV_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_HCP_A_cryst_wnoise_all.csv'
d_c = read_csv(paste0(plt_in,f))

d_c$reliability = d_c$reliability * d_c$reliability * 0.92


#d_t$test_MAE = d_t$test_MAE*-1

empirical = d_t[1,]
d = d_t[-1,]

metrics = colnames(d)
rel_max = max(d$reliability)
rel_min = min(d$reliability)


## Calculate mean and sd
df_m = d_t %>% group_by(reliability) %>%
  summarise_at(vars(metrics[c(-1,-2,-13,-14)]), list(name = mean))
colnames(df_m)[2:11] <- metrics[c(-1,-2,-13,-14)]

df_sd = d_t %>% group_by(reliability) %>%
  summarise_at(vars(metrics[c(-1,-2,-13,-14)]), list(name = sd))
colnames(df_sd)[2:11] <- metrics[c(-1,-2,-13,-14)]

df_m = df_m %>% select(c('reliability', 'test_R2'))
df_t = cbind(df_m,df_sd$test_R2)
colnames(df_t) = c('reliability', 'R2_mean', 'R2_sd')
df_t$variable = 'total_cog'

df_sum = df_t


df_m = d_m %>% group_by(reliability) %>%
  summarise_at(vars(metrics[c(-1,-2,-13,-14)]), list(name = mean))
colnames(df_m)[2:11] <- metrics[c(-1,-2,-13,-14)]

df_sd = d_m %>% group_by(reliability) %>%
  summarise_at(vars(metrics[c(-1,-2,-13,-14)]), list(name = sd))
colnames(df_sd)[2:11] <- metrics[c(-1,-2,-13,-14)]

df_m = df_m %>% select(c('reliability', 'test_R2'))
df_g = cbind(df_m,df_sd$test_R2)
colnames(df_g) = c('reliability', 'R2_mean', 'R2_sd')
df_g$variable = 'grip_strength'   

df_sum = rbind(df_sum, df_g)


df_m = d_c %>% group_by(reliability) %>%
  summarise_at(vars(metrics[c(-1,-2,-13,-14)]), list(name = mean))
colnames(df_m)[2:11] <- metrics[c(-1,-2,-13,-14)]

df_sd = d_c %>% group_by(reliability) %>%
  summarise_at(vars(metrics[c(-1,-2,-13,-14)]), list(name = sd))
colnames(df_sd)[2:11] <- metrics[c(-1,-2,-13,-14)]

df_m = df_m %>% select(c('reliability', 'test_R2'))
df_c = cbind(df_m,df_sd$test_R2)
colnames(df_c) = c('reliability', 'R2_mean', 'R2_sd')
df_c$variable = 'cryst_cog'

df_sum = rbind(df_sum, df_c)


# for saving fig
pdf(file = paste0(plt_out, "Fig1_B.pdf"), width = 4.5, height = 3, useDingbats = FALSE)

# plot all together
ggplot(df_sum, aes(x = reliability)) +
  geom_point(aes(y = R2_mean, colour = variable), size=1) +
  geom_line(aes(y = R2_mean, colour = variable), size=0.8) +
  geom_ribbon(aes(y = R2_mean, ymin = R2_mean - 2*R2_sd, ymax = R2_mean + 2*R2_sd, fill = variable), alpha = .1) +
  theme_classic() + 
  scale_y_continuous(limits = c(-0.1,0.3), breaks = c(0.3, 0.2, 0.1, 0, -0.1), name = 'Prediction Accuracy (R2)') +
  scale_x_continuous(limits = c(0.2,1.0), breaks = c(seq(0.2,1.0,0.2))) +
  scale_colour_manual(values = net_cols, name = "Phenotype",
                      labels = c("Cryst. cog.", "Grip strength", "Total cog.")) +
  scale_fill_manual(values = net_cols, name = "Phenotype",
                      labels = c("Cryst. cog.", "Grip strength", "Total cog.")) +
  xlab('Reliability') +
  theme(axis.text = element_text(size = 10.5),
        axis.title = element_text(size = 10.5),
        legend.text = element_text(size = 11),
        legend.title = element_text(size = 10.5))

dev.off()







### FIGURE 1C ###
# HCP A plot results for prediction of simulated variables together
rm(list = ls())

#net_cols <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") # original order
net_cols <- c("#0072B2", "#009E73", "#D55E00") # reordering for 3 colours only



plt_in = '/home/mgell/Work/reliability/input/'
plt_out = '/home/mgell/Work/reliability/plots/Figures/'

f = 'ridgeCV_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_HCP_A_total_wnoise_all.csv'
d_t = read_csv(paste0(plt_in,f))
d_t$reliability = d_t$reliability * d_t$reliability * 0.9

f = 'ridgeCV_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_HCP_A_motor_wnoise_all.csv'
d_m = read_csv(paste0(plt_in,f))
d_m$reliability = d_m$reliability * d_m$reliability * 0.93

f = 'ridgeCV_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_HCP_A_cryst_wnoise_all.csv'
d_c = read_csv(paste0(plt_in,f))
d_c$reliability = d_c$reliability * d_c$reliability * 0.92

d_t$beh = 'total_cog'
d_m$beh = 'grip'
d_c$beh = 'cryst_cog'

d = rbind(d_t,d_m,d_c)

# Restrict to reliability of 0.6 to 0.4
d2 = d
d2 = d2[d2$reliability >= 0.35,]
d2 = d2[d2$reliability <= 0.55,]

d2$reliability = round(d2$reliability,digits = 3)
# optionally, for visualization purposes only:
# round 0.38 to 0.39, 0.46 to 0.45, 0.44 to 0.45 and 0.51 to 0.52 otherwise 
# some behaviours get plotted on their own tick marks and others dont so its hard to follow
# d2$reliability[d2$reliability == 0.38] = 0.39
# d2$reliability[d2$reliability == 0.46] = 0.45
# d2$reliability[d2$reliability == 0.44] = 0.45
# d2$reliability[d2$reliability == 0.51] = 0.52


# for saving fig
pdf(file = paste0(plt_out, "Fig1_Cround.pdf"), width = 7, height = 2.5, useDingbats = FALSE)

# plot all together
ggplot(d2,aes(as.factor(reliability),test_R2,colour=beh,fill=beh)) + 
  geom_violin(alpha = 0.2) +
  geom_dotplot(binaxis="y", stackdir="center", binwidth=0.004, alpha = 0.7, position=position_dodge(0.9)) + 
  theme_classic() + 
  ylim(c(-0.05,0.15)) +
  xlab('Reliability') + ylab('Prediction Accuracy (R2)') +
  scale_colour_manual(values = net_cols, name = "Variable",
                      labels = c("Cryst. cog.", "Grip strength", "Total cog.")) +
  scale_fill_manual(values = net_cols, name = "Variable",
                    labels = c("Cryst. cog.", "Grip strength", "Total cog.")) +
  theme(legend.position="none",
        axis.text = element_text(size = 10.5),
        axis.title = element_text(size = 10.5))

dev.off()

