
library(tidyverse)


beh = 'Grip'

f = 'ridgeCV_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_HCP_A_motor_wnoise_all.csv'
plt_out = '/home/mgell/Work/reliability/plots/simulation_res/'

d = read_csv(paste0('/home/mgell/Work/reliability/input/',f))
d$test_MAE = d$test_MAE*-1

empirical = d[1,]
d = d[-1,]

metrics = colnames(d)
rel_max = max(d$reliability)
rel_min = min(d$reliability)

d$reliability = round(d$reliability*d$reliability,digits = 2)

# Scatter
# plt1 = ggplot(d,aes(reliability,test_R2)) + geom_point(size = 1.5, color = 'deeppink3') + 
#   theme_classic() + ylim(c(-0.1,0.2)) + xlab('r(original,simulated)') + theme(legend.position="none") +
#   geom_point(aes(empirical$reliability,empirical$test_R2), size = 3, colour = 'skyblue3')
# ggsave(paste0(plt_out,beh,'_',metrics[7],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt1, width=3.5, height=3)
# 
# 
# plt2 = ggplot(d,aes(reliability,test_MAE)) + geom_point(size = 1.5, color = 'deeppink3') + 
#   theme_classic() + ylim(c(9,13)) + xlab('r(original,simulated)') + theme(legend.position="none") +
#   geom_point(aes(empirical$reliability,empirical$test_MAE), size = 3, colour = 'skyblue3')
# ggsave(paste0(plt_out,beh,'_',metrics[5],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt2, width=3.5, height=3)
# 
# 
# plt3 = ggplot(d,aes(reliability,test_r)) + geom_point(size = 1.5, color = 'deeppink3') + 
#   theme_classic() + ylim(c(0.0,0.5)) + xlab('r(original,simulated)') + theme(legend.position="none") +
#   geom_point(aes(empirical$reliability,empirical$test_r), size = 3, colour = 'skyblue3')
# ggsave(paste0(plt_out,beh,'_',metrics[9],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt3, width=3.5, height=3)


# Violin plots
plt1 = ggplot(d,aes(as.factor(reliability),test_R2)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(-0.1,0.3)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_R2), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5)) +
  ylab('R2')
ggsave(paste0(plt_out,'rel_squared_',beh,'_',metrics[7],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt1, width=3.5, height=3)

plt2 = ggplot(d,aes(as.factor(reliability),test_MAE)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(8.5,14)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_MAE), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5)) +
  ylab('MAE')
ggsave(paste0(plt_out,'rel_squared_',beh,'_',metrics[5],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt2, width=3.5, height=3)

plt3 = ggplot(d,aes(as.factor(reliability),test_r)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(0.0,0.6)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_r), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5)) +
  ylab('r(predicted,observed')
ggsave(paste0(plt_out,'rel_squared_',beh,'_',metrics[9],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt3, width=3.5, height=3)








# total cognition
beh = 'Total'

f = 'ridgeCV_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_HCP_A_total_wnoise_all.csv'
plt_out = '/home/mgell/Work/reliability/plots/simulation_res/'

d = read_csv(paste0('/home/mgell/Work/reliability/input/',f))
d$test_MAE = d$test_MAE*-1

empirical = d[1,]
d = d[-1,]

metrics = colnames(d)
rel_max = max(d$reliability)
rel_min = min(d$reliability)

d$reliability = round(d$reliability*d$reliability,digits = 2)


# Violins
plt1 = ggplot(d,aes(as.factor(reliability),test_R2)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(-0.1,0.3)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_R2), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5)) +
  ylab('R2')
ggsave(paste0(plt_out,'rel_squared_',beh,'_',metrics[7],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt1, width=3.5, height=3)


plt2 = ggplot(d,aes(as.factor(reliability),test_MAE)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(8.5,14)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_MAE), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5)) +
  ylab('MAE')
ggsave(paste0(plt_out,'rel_squared_',beh,'_',metrics[5],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt2, width=3.5, height=3)


plt3 = ggplot(d,aes(as.factor(reliability),test_r)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(0.0,0.6)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_r), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5)) +
  ylab('r(predicted,observed')
ggsave(paste0(plt_out,'rel_squared_',beh,'_',metrics[9],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt3, width=3.5, height=3)








# crystalised cognition
beh = 'crycog'

f = 'ridgeCV_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_HCP_A_cryst_wnoise_all.csv'
plt_out = '/home/mgell/Work/reliability/plots/simulation_res/'

df = read_csv(paste0('/home/mgell/Work/reliability/input/',f))
df$test_MAE = df$test_MAE*-1

empirical = df[1,]
d = df[-1,]

metrics = colnames(d)
rel_max = max(d$reliability)
rel_min = min(d$reliability)

d$reliability = round(d$reliability*d$reliability,digits = 2)

# Violin plots
plt1 = ggplot(d,aes(as.factor(reliability),test_R2)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(-0.1,0.3)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_R2), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5)) +
  ylab('R2')
ggsave(paste0(plt_out,'rel_squared_',beh,'_',metrics[7],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt1, width=3.5, height=3)


plt2 = ggplot(d,aes(as.factor(reliability),test_MAE)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(8.5,14)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_MAE), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5)) +
  ylab('MAE')
ggsave(paste0(plt_out,'rel_squared_',beh,'_',metrics[5],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt2, width=3.5, height=3)


plt3 = ggplot(d,aes(as.factor(reliability),test_r)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(0.0,0.6)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_r), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5)) +
  ylab('r(predicted,observed')
ggsave(paste0(plt_out,'rel_squared_',beh,'_',metrics[9],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt3, width=3.5, height=3)


