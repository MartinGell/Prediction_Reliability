#

library(tidyverse)

subs = read_csv('/home/mgell/Work/reliability/text_files/HCP_YA_subs.csv')

tab =  read_csv('/home/mgell/Work/reliability/text_files/HCP_YA_S1200_beh.csv')

ages = read_csv('/home/mgell/Work/reliability/text_files/HCP_YA_restricted.csv')

tab$Age_years = ages$Age_in_Yrs
tab$Family_ID = ages$Family_ID

behs <- c(
  'Subject',
  'WM_Task_Acc', 'WM_Task_Median_RT',
  'WM_Task_2bk_Acc', 'WM_Task_2bk_Median_RT',
  'WM_Task_0bk_Acc','WM_Task_0bk_Median_RT',
  'Language_Task_Acc', 'Language_Task_Median_RT',
  'Language_Task_Math_Acc', 'Language_Task_Math_Median_RT', 
  'Language_Task_Story_Acc','Language_Task_Story_Median_RT',
  'Relational_Task_Acc', 'Relational_Task_Median_RT',
  'Relational_Task_Match_Acc', 'Relational_Task_Match_Median_RT',
  'Relational_Task_Rel_Acc', 'Relational_Task_Rel_Median_RT',
  'Emotion_Task_Acc','Emotion_Task_Median_RT',
  'Emotion_Task_Face_Acc','Emotion_Task_Face_Median_RT',
  'Emotion_Task_Shape_Acc','Emotion_Task_Shape_Median_RT',
  'Age_years','Gender',
  'FS_IntraCranial_Vol', 'FS_Total_GM_Vol',
  'Family_ID'
)


d = tab

d <- d %>% select(all_of(behs))

d_all <- na.omit(d)

# remove high motion
FDs <- read_csv('/home/mgell/Work/Prediction_HCP/text_files/HCP_YA_FD.csv')
d_all <- filter(d_all, Subject %in% c(FDs$subjects))
FDs <- filter(FDs, subjects %in% c(d_all$Subject))

d_all$FD_REST1 <- FDs$REST1
d_all$FD_REST2 <- FDs$REST2
d_all <- na.omit(d_all)

d_all$FD <- rowMeans(data.frame(d_all$FD_REST1, d_all$FD_REST2))

# export the csv
d_all$FDz = (d_all$FD - mean(d_all$FD))/sd(d_all$FD)
d_lowmo = d_all[d_all$FDz < 3,]

names(d_lowmo)[names(d_lowmo) == 'Age_years'] = 'Age'

d_lowmo$Sex <- 1
d_lowmo$Sex[d_lowmo$Gender == 'F'] <- 0


d_lowmo$WM_Task_Acc_log <- log(d_lowmo$WM_Task_Acc)
d_lowmo$WM_Task_2bk_Acc_log <- log(d_lowmo$WM_Task_2bk_Acc)
d_lowmo$WM_Task_0bk_Acc_log <- log(d_lowmo$WM_Task_0bk_Acc)


d_lowmo$Language_Task_Acc_log <- log(d_lowmo$Language_Task_Acc)
d_lowmo$Language_Task_Math_Acc_log <- log(d_lowmo$Language_Task_Math_Acc)
d_lowmo$Language_Task_Story_Acc_log <- log(d_lowmo$Language_Task_Story_Acc)


# save
write_csv(d_lowmo, '/home/mgell/Work/Prediction_HCP/text_files/HCP_YA_beh_all_increase_rel.csv')
write.table(colnames(d_lowmo), '/home/mgell/Work/Prediction_HCP/code/opts/HCP_YA_behs2predict_increase_rel.txt',row.names = FALSE, quote = FALSE)


# demographics
subs_leo = read_csv('/home/mgell/Work/reliability/text_files/HCP_YA_subs_leo.csv')
sample <- filter(d_lowmo, Subject %in% c(subs_leo$subs))
sample$Gender = as.factor(sample$Gender)
summarise(sample)




# PCA on RT measures
d_pca <- d_lowmo %>% select('WM_Task_Median_RT','Language_Task_Median_RT', 'Relational_Task_Median_RT','Emotion_Task_Median_RT')

results <- prcomp(d_pca, scale = TRUE)
results$x <- -1*results$x

#calculate total variance explained by each principal component
var_explained = results$sdev^2 / sum(results$sdev^2)

#create scree plot
qplot(c(1:12), var_explained) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab("Variance Explained") +
  ggtitle("Scree Plot") +
  ylim(0, 1)

