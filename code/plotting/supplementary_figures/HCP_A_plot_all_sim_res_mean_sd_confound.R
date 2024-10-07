
library(tidyverse)



#net_cols <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") # original order
net_cols <- c("#0072B2", "#009E73", "#D55E00", "#56B4E9", "#E69F00", "#F0E442", "#CC79A7") # reordering for 3 colours only



plt_in = '/home/mgell/Work/reliability/input/'
plt_out = '/home/mgell/Work/reliability/plots/simulation_res/'

f = 'ridgeCV_zscore_confound_removal_wcategorical_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_HCP_A_total_wnoise_all.csv'
d_t = read_csv(paste0(plt_in,f))

d_t$reliability = d_t$reliability * d_t$reliability * 0.9
  
f = 'ridgeCV_zscore_confound_removal_wcategorical_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_HCP_A_motor_wnoise_all.csv'
d_m = read_csv(paste0(plt_in,f))

d_m$reliability = d_m$reliability * d_m$reliability * 0.93

f = 'ridgeCV_zscore_confound_removal_wcategorical_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_HCP_A_cryst_wnoise_all.csv'
d_c = read_csv(paste0(plt_in,f))

d_c$reliability = d_c$reliability * d_c$reliability * 0.92

#d_t$test_MAE = d_t$test_MAE*-1

empirical = d_t[1,]
d = d_t[-1,]

metrics = colnames(d)
rel_max = max(d$reliability)
rel_min = min(d$reliability)


# total cog
df_m = d_t %>% group_by(reliability) %>%
  summarise_at(vars(metrics[c(-1,-2,-13,-14)]), list(name = mean))
colnames(df_m)[2:11] <- metrics[c(-1,-2,-13,-14)]

df_sd = d_t %>% group_by(reliability) %>%
  summarise_at(vars(metrics[c(-1,-2,-13,-14)]), list(name = sd))
colnames(df_sd)[2:11] <- metrics[c(-1,-2,-13,-14)]

df_m = df_m %>% select(c('reliability', 'test_R2'))
df = cbind(df_m,df_sd$test_R2)
colnames(df) = c('reliability', 'R2_mean', 'R2_sd')
df$variable = 'total_cog'

df_sum = df


# HGS
df_m = d_m %>% group_by(reliability) %>%
  summarise_at(vars(metrics[c(-1,-2,-13,-14)]), list(name = mean))
colnames(df_m)[2:11] <- metrics[c(-1,-2,-13,-14)]

df_sd = d_m %>% group_by(reliability) %>%
  summarise_at(vars(metrics[c(-1,-2,-13,-14)]), list(name = sd))
colnames(df_sd)[2:11] <- metrics[c(-1,-2,-13,-14)]

df_m = df_m %>% select(c('reliability', 'test_R2'))
df = cbind(df_m,df_sd$test_R2)
colnames(df) = c('reliability', 'R2_mean', 'R2_sd')
df$variable = 'grip_strength'   

df_sum = rbind(df_sum, df)


# crystal cog
df_m = d_c %>% group_by(reliability) %>%
  summarise_at(vars(metrics[c(-1,-2,-13,-14)]), list(name = mean))
colnames(df_m)[2:11] <- metrics[c(-1,-2,-13,-14)]

df_sd = d_c %>% group_by(reliability) %>%
  summarise_at(vars(metrics[c(-1,-2,-13,-14)]), list(name = sd))
colnames(df_sd)[2:11] <- metrics[c(-1,-2,-13,-14)]

df_m = df_m %>% select(c('reliability', 'test_R2'))
df = cbind(df_m,df_sd$test_R2)
colnames(df) = c('reliability', 'R2_mean', 'R2_sd')
df$variable = 'cryst_cog'

df_sum = rbind(df_sum, df)


# plot all together
pltm = ggplot(df_sum, aes(x = reliability)) + 
  geom_point(aes(y = R2_mean, colour = variable), size=1) +
  geom_line(aes(y = R2_mean, colour = variable), size=0.8) +
  geom_ribbon(aes(y = R2_mean, ymin = R2_mean - 2*R2_sd, ymax = R2_mean + 2*R2_sd, fill = variable), alpha = .1) +
  theme_classic() + 
  scale_y_continuous(limits = c(-0.1,0.3), breaks = c(0.3, 0.2, 0.1, 0, -0.1), name = 'Prediction Accuracy (R2)') +
  scale_colour_manual(values = net_cols, name = "Phenotype",
                      labels = c("Cryst. cog.", "Grip strength", "Total cog.")) +
  scale_fill_manual(values = net_cols, name = "Phenotype",
                    labels = c("Cryst. cog.", "Grip strength", "Total cog.")) +
  scale_x_continuous(limits = c(0.15,1.0), breaks = c(0.2,0.4,0.6,0.8,1.0)) +
  xlab('Reliability') #+
ggsave(paste0(plt_out,'rel_squared_All_variables_conf_R2_mean_sd_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), pltm, width=4, height=3)


