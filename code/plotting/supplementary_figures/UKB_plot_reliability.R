
library(tidyverse)
library(psych)


plt_out = '/data/project/ukb_reliability_in_prediction/plots/UKB/'


# load data
T1 <- read.csv('/data/project/ukb_reliability_in_prediction/input/cog_plus_2.0_data.csv', check.names = FALSE)
T2 <- read.csv('/data/project/ukb_reliability_in_prediction/input/cog_plus_3.0_data.csv', check.names = FALSE)


cormat <- cor(T1,T2,use = "pairwise.complete.obs")
rel <- cormat[row(cormat)==col(cormat)]
df <- data.frame('rel_r' = rel)

beh = colnames(T1)
colnames(T2) = behs
ICC2 = numeric(length(behs))
ICC2_upper = numeric(length(behs))
ICC2_lower = numeric(length(behs))
i = 1
for (beh_i in behs) {
  x <- ICC(data.frame(T1[,beh_i],T2[,beh_i]))
  ICC <- x$results$ICC[2]
  ICC2[i] = ICC
  ICC2_upper[i] = x$results$`upper bound`[2]
  ICC2_lower[i] = x$results$`lower bound`[2]
  i = i+1
}

df$rel_ICC2 <- ICC2
df$ICC2_low <- ICC2_upper
df$ICC2_upp <- ICC2_lower
df$beh <- beh

rel=df

# plot reliabilities
#r
names = read.delim('/data/project/ukb_reliability_in_prediction/res/cog_plus_cors_pearson_all.csv', header = FALSE, sep = '-')
rel$behs = names$V1
rel = rel[c(-16,-20),]

plt = ggplot(rel, aes(x = behs, y = rel_r)) +
  geom_bar(stat = "identity", width = .75) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90),
        axis.title.x=element_blank()) +
  ylab('cor(neuroimaging session, follow-up session)') +
  xlab('Behaviour')
ggsave(paste0(plt_out,'UKB_all_behs_reliability.png'), plt, width=6, height=8)

#icc
names = read.delim('/data/project/ukb_reliability_in_prediction/res/cog_plus_cors_pearson_all.csv', header = FALSE, sep = '-')
rel=df
rel$behs = names$V1
rel = rel[c(-16,-20),]

plt = ggplot(rel, aes(x = behs, y = rel_ICC2)) +
  geom_bar(stat = "identity", width = .75) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90),
        axis.title.x=element_blank()) +
  ylab('Reliability (ICC2)') +
  xlab('Behaviour')
ggsave(paste0(plt_out,'UKB_all_behs_reliability_ICC.png'), plt, width=6, height=8)




# load accuracy and save again with reliability colls
d = read.csv('/data/project/ukb_reliability_in_prediction/input/collected/ridgeCV_zscore_confound_removal_wcategorical_averaged-source_Schaefer400x17_nodenoise_UKB_5000_z-beh_UKB_5000_subs_FC_all_cogs_all_behs.csv')

d = d[-1,] # repeated value
d$test_MAE = d$test_MAE*-1

d <- left_join(d, df, by=c("beh"))

mean(d$rel_ICC2)
mean(d$rel_r)

# save
write_csv(d, '/data/project/ukb_reliability_in_prediction/res/UKB_reliability_accuracy.csv')




### for plotting together with HPC_YA
DF = d %>% select(test_R2, reliability, beh)
DF$dataset = 'UKB'

write_csv(DF,'/data/project/ukb_reliability_in_prediction/res/UKB_rel_ICC.csv')

