
library(tidyverse)


beh = 'Age'

f = 'ridgeCV_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_interview_age_wnoise_all.csv'
plt_out = '/home/mgell/Work/reliability/plots/simulation_res/'

d = read_csv(paste0('/home/mgell/Work/reliability/input/',f))
d$test_MAE = d$test_MAE*-1

empirical = d[1,]
d = d[-1,]

metrics = colnames(d)
rel_max = max(d$reliability)
rel_min = min(d$reliability)

d$reliability = d$reliability*d$reliability

# Violin plots
plt1 = ggplot(d,aes(reliability,test_R2,group=as.factor(reliability))) + 
  geom_violin(color = 'deeppink3', width=0.1, position=position_dodge(width = 0)) + 
  geom_boxplot(colour = "grey", width=0.01, position=position_dodge(width = 0),outlier.size = 0) +
  theme_classic() +
  ylim(c(-0.1,0.8)) + 
  scale_x_continuous(limits = c(0.2,1.05), breaks = c(seq(0.2,1.0,0.1))) +
  xlab('Reliability') + 
  ylab('Age Prediction Accuracy (R2)') +
  theme(legend.position="none") +
  geom_point(aes(empirical$reliability,empirical$test_R2), size = 3, colour = 'skyblue3')
ggsave(paste0(plt_out,'rel_squared',beh,'_',metrics[7],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt1, width=3.5, height=3)



plt2 = ggplot(d,aes(as.factor(reliability),test_MAE)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(80,180)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_MAE), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5))
ggsave(paste0(plt_out,'rel_squared',beh,'_',metrics[5],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt2, width=3.5, height=3)


plt3 = ggplot(d,aes(as.factor(reliability),test_r)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(0.0,1.0)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_r), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5))
ggsave(paste0(plt_out,'rel_squared',beh,'_',metrics[9],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt3, width=3.5, height=3)







# replication with different nodes
rm(list = ls())

beh = 'Age_Seitzman'

f = 'ridgeCV_zscore_averaged-source_Seitzman_nodes300_WM+CSF+GS_hcpaging_650_zscored-beh_interview_age_wnoise_all.csv'
plt_out = '/home/mgell/Work/reliability/plots/simulation_res/'

d = read_csv(paste0('/home/mgell/Work/reliability/input/',f))
d$test_MAE = d$test_MAE*-1

empirical = d[1,]
d = d[-1,]

metrics = colnames(d)
rel_max = max(d$reliability)
rel_min = min(d$reliability)

d$reliability = d$reliability*d$reliability


# Violins
plt1 = ggplot(d,aes(as.factor(reliability),test_R2)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(-0.1,0.8)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_R2), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5))
ggsave(paste0(plt_out,'rel_squared',beh,'_',metrics[7],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt1, width=3.5, height=3)


plt2 = ggplot(d,aes(as.factor(reliability),test_MAE)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(80,180)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_MAE), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5))
ggsave(paste0(plt_out,'rel_squared',beh,'_',metrics[5],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt2, width=3.5, height=3)


plt3 = ggplot(d,aes(as.factor(reliability),test_r)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(0.0,1.0)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_r), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5))
ggsave(paste0(plt_out,'rel_squared',beh,'_',metrics[9],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt3, width=3.5, height=3)









# replication with SVR alg.
rm(list = ls())

beh = 'age_SVR'

f = 'svr_heuristic_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_interview_age_wnoise_all.csv'
plt_out = '/home/mgell/Work/reliability/plots/simulation_res/'

d = read_csv(paste0('/home/mgell/Work/reliability/input/',f))
d$test_MAE = d$test_MAE*-1

empirical = d[1,]
d = d[-1,]

metrics = colnames(d)
rel_max = max(d$reliability)
rel_min = min(d$reliability)

d$reliability = d$reliability*d$reliability


# Violins
plt1 = ggplot(d,aes(as.factor(reliability),test_R2)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(-0.11,0.8)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_R2), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5))
ggsave(paste0(plt_out,'rel_squared',beh,'_',metrics[7],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt1, width=3.5, height=3)


plt2 = ggplot(d,aes(as.factor(reliability),test_MAE)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(80,180)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_MAE), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5))
ggsave(paste0(plt_out,'rel_squared',beh,'_',metrics[5],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt2, width=3.5, height=3)


plt3 = ggplot(d,aes(as.factor(reliability),test_r)) + geom_violin(color = 'deeppink3') + 
  theme_classic() + ylim(c(0.0,1.0)) + xlab('Reliability') + theme(legend.position="none") +
  geom_point(aes(as.factor(empirical$reliability),empirical$test_r), size = 3, colour = 'skyblue3') +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5))
ggsave(paste0(plt_out,'rel_squared',beh,'_',metrics[9],'_wnoise_rel_',rel_max,'_to_',rel_min,'.png'), plt3, width=3.5, height=3)



