
# Correlate HCP test retest data to get a sense of reliability

library(tidyverse)
library(corrplot)
library(psych)

#net_cols <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") # original order


# Load
T1 <- read.csv('/home/mgell/Work/reliability/text_files/HCP_YA_retest_T1_beh.csv')
T2 <- read.csv('/home/mgell/Work/reliability/text_files/HCP_YA_retest_T2_beh.csv')

mean(T1$Age)
sd(T1$Age)

mean(T2$Age)
sd(T2$Age)

behs <- c(
    'PicSeq_AgeAdj', 'CardSort_AgeAdj', 'Flanker_AgeAdj', 'PMAT24_A_CR',
    'ReadEng_AgeAdj', 'PicVocab_AgeAdj', 'ProcSpeed_AgeAdj', 'DDisc_AUC_200',
    'DDisc_AUC_40K', 'ListSort_AgeAdj', 'SCPT_SPEC',
    'CogFluidComp_AgeAdj', 'CogTotalComp_AgeAdj', 'CogCrystalComp_AgeAdj',
    'ER40_CR', 'ER40_CRT', 
    'Strength_AgeAdj', 'Dexterity_AgeAdj',
    'WM_Task_2bk_Acc', 'WM_Task_2bk_Median_RT',
    'WM_Task_0bk_Acc','WM_Task_0bk_Median_RT',
    'Language_Task_Math_Acc', 'Language_Task_Math_Median_RT', 
    'Language_Task_Story_Acc','Language_Task_Story_Median_RT',
    'Relational_Task_Match_Acc', 'Relational_Task_Match_Median_RT',
    'Relational_Task_Rel_Acc', 'Relational_Task_Rel_Median_RT',
    'Emotion_Task_Face_Acc','Emotion_Task_Face_Median_RT',
    'Emotion_Task_Shape_Acc','Emotion_Task_Shape_Median_RT',
    'Social_Task_Perc_Random','Social_Task_Perc_TOM',
    'Social_Task_Median_RT_Random', 'Social_Task_Median_RT_TOM',
    'Gambling_Task_Perc_Larger', 'Gambling_Task_Perc_Smaller', 
    'Gambling_Task_Median_RT_Larger', 'Gambling_Task_Median_RT_Smaller'
    )

T1 <- T1 %>% select(all_of(behs))
T2 <- T2 %>% select(all_of(behs))


T1$SCPT_SPEC_log <- log(T1$SCPT_SPEC)
#T1$DDisc_AUC_200 <- log(T1$DDisc_AUC_200)
T1$WM_Task_2bk_Acc_log <- log(T1$WM_Task_2bk_Acc)
#T1$Language_Task_Acc_log <- log(T1$Language_Task_Acc)
T1$Language_Task_Math_Acc_log <- log(T1$Language_Task_Math_Acc)
T1$Social_Task_Median_RT_Random_log <- log(T1$Social_Task_Median_RT_Random)
T1$Social_Task_Median_RT_TOM_log <- log(T1$Social_Task_Median_RT_TOM)
T1$Relational_Task_Match_Acc_log <- log(T1$Relational_Task_Match_Acc)
T1$Gambling_Task_Median_RT_Larger_log <- log(T1$Gambling_Task_Median_RT_Larger)
T1$Gambling_Task_Median_RT_Smaller_log <- log(T1$Gambling_Task_Median_RT_Smaller)

T2$SCPT_SPEC_log <- log(T2$SCPT_SPEC)
#T2$DDisc_AUC_200 <- log(T2$DDisc_AUC_200)
T2$WM_Task_2bk_Acc_log <- log(T2$WM_Task_2bk_Acc)
#T2$Language_Task_Acc_log <- log(T2$Language_Task_Acc)
T2$Language_Task_Math_Acc_log <- log(T2$Language_Task_Math_Acc)
T2$Social_Task_Median_RT_Random_log <- log(T2$Social_Task_Median_RT_Random)
T2$Social_Task_Median_RT_TOM_log <- log(T2$Social_Task_Median_RT_TOM)
T2$Relational_Task_Match_Acc_log <- log(T2$Relational_Task_Match_Acc)
T2$Gambling_Task_Median_RT_Larger_log <- log(T2$Gambling_Task_Median_RT_Larger)
T2$Gambling_Task_Median_RT_Smaller_log <- log(T2$Gambling_Task_Median_RT_Smaller)

behs = c(behs,c('SCPT_SPEC_log','WM_Task_2bk_Acc_log',
                'Language_Task_Math_Acc_log',
                'Social_Task_Median_RT_Random_log','Social_Task_Median_RT_TOM_log',
                'Relational_Task_Match_Acc_log',
                'Gambling_Task_Median_RT_Larger_log','Gambling_Task_Median_RT_Smaller_log'))


cormat <- cor(T1,T2,use = "pairwise.complete.obs")
rel <- cormat[row(cormat)==col(cormat)]
#d <- data.frame(t(rel))
d <- data.frame('beh' = behs,'reliability_r' = rel)


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

d$reliability_icc2 = ICC2
d$reliability_icc2_upper = ICC2_upper
d$reliability_icc2_lower = ICC2_lower


# Plot reliability of all data and check for oddities
ggplot(data = d, aes(x = behs, y = reliability_r)) +
  geom_bar(stat = "identity", width = .75) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90),
        axis.title.x=element_blank()) +
  #ylim(c(min(d),1.0)) +
  ylab('correlation(T1,T2)')

# Now remove behaviours with heavy tails and non-normal distributions for prediction

new_behs <- c(
  'PicSeq_AgeAdj', 'CardSort_AgeAdj', 'Flanker_AgeAdj', 'PMAT24_A_CR',
  'ReadEng_AgeAdj', 'PicVocab_AgeAdj', 'ProcSpeed_AgeAdj', 'DDisc_AUC_200',
  'DDisc_AUC_40K', 'ListSort_AgeAdj', 'SCPT_SPEC',
  'CogFluidComp_AgeAdj', 'CogTotalComp_AgeAdj', 'CogCrystalComp_AgeAdj',
  'ER40_CR', 'ER40_CRT', 
  'Strength_AgeAdj', 'Dexterity_AgeAdj', 
  'WM_Task_2bk_Acc', 'WM_Task_2bk_Median_RT',
  'WM_Task_0bk_Median_RT',
  'Language_Task_Math_Acc', 'Language_Task_Math_Median_RT', 
  'Language_Task_Story_Median_RT',
  'Social_Task_Perc_Random','Social_Task_Perc_TOM',
  'Social_Task_Median_RT_Random', 'Social_Task_Median_RT_TOM',
  'Relational_Task_Match_Acc', 'Relational_Task_Match_Median_RT',
  'Relational_Task_Rel_Acc', 'Relational_Task_Rel_Median_RT',
  'Emotion_Task_Face_Median_RT',
  'Emotion_Task_Shape_Median_RT',
  'Gambling_Task_Median_RT_Larger', 'Gambling_Task_Median_RT_Smaller',
  'SCPT_SPEC_log','WM_Task_2bk_Acc_log',
  'Language_Task_Math_Acc_log',
  'Social_Task_Median_RT_Random_log','Social_Task_Median_RT_TOM_log',
  'Relational_Task_Match_Acc_log',
  'Gambling_Task_Median_RT_Smaller_log'
)


df <- d %>% filter(beh %in% new_behs)

mean(df$reliability_r)
mean(df$reliability_icc2)


# plot final reliability
plt = ggplot(data = df, aes(x = beh, y = reliability_icc2)) +
  geom_bar(stat = "identity", width = .75) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90),
        axis.title.x=element_blank()) +
  ylim(c(0,1.0)) +
  ylab('Reliability (ICC)')
ggsave('/home/mgell/Work/reliability/plots/HCP_YA_behaviour/HCP_YA_rel_ICC.png', plt, width=8, height=5)

all_behs = data.frame('all_behs' = new_behs)
write_delim(all_behs, '/home/mgell/Work/Prediction_HCP/code/opts/HCP_YA_behs2predict.txt', col_names = FALSE)




# load accuracy and save again with reliability colls
d = read_csv('/home/mgell/Work/reliability/input/ridgeCV_zscore_stratified_KFold_confound_removal_wcategorical_averaged-source_Schaefer400x17_WM+CSF+GS_hcpya_771_zscored-beh_HCP_YA_beh_all_all_behs.csv')

d = d[-1,] # repeated value
d$test_MAE = d$test_MAE*-1

# remove rows that were not log transformed (they were kept for plotting reliability)
#d <- d[c(-11,-19,-22,-27,-28,-29,-36),]
new_behs <- c(
  "SCPT_SPEC", "WM_Task_2bk_Acc",
  "Language_Task_Math_Acc",
  "Social_Task_Median_RT_Random", "Social_Task_Median_RT_TOM",
  "Relational_Task_Match_Acc",
  "Gambling_Task_Median_RT_Smaller"
)

d <- d %>% filter(!beh %in% new_behs)
d <- left_join(d, df, by=c("beh"))


# save
write.csv(d,file = '/home/mgell/Work/reliability/res/HCP_YA_reliability_accuracy.csv', row.names = FALSE)




