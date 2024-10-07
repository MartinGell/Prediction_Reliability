
library(psych)
library(tidyverse)

# load
baseline <- read.csv('/home/mgell/Work/reliability/text_files/abcd/baseline.csv')
followup <- read.csv('/home/mgell/Work/reliability/text_files/abcd/followup.csv')

#summary(baseline)
#summary(followup)

# remove vars with too many nans
behs <- c("src_subject_id","eventname",
          "nihtbx_picvocab_fc","nihtbx_flanker_fc",
          "nihtbx_pattern_fc","nihtbx_picture_fc","nihtbx_reading_fc",
          "nihtbx_cryst_fc",
          "pea_ravlt_sd_trial_vi_tc","pea_ravlt_ld_trial_vii_tc",
          "lmt_scr_perc_correct","lmt_scr_perc_wrong",
          "lmt_scr_avg_rt","lmt_scr_rt_correct",
          "tfmri_mid_all_beh_srwpfb_nt","tfmri_mid_all_beh_lrwpfb_mrt",
          "tfmri_mid_all_beh_srwpfb_mrt","tfmri_mid_all_beh_lrwpfb_nt",
          "tfmri_sst_all_beh_total_mssrt",
          #          "tfmri_nb_all_beh_ctotal_rate", "tfmri_nb_all_beh_ctotal_mrt",
          "tfmri_nb_all_beh_c2b_rate","tfmri_nb_all_beh_c2b_mrt",
          "tfmri_nb_all_beh_c0b_rate","tfmri_nb_all_beh_c0b_mrt",
          # "tfmri_nb_all_beh_cpf_rate","tfmri_nb_all_beh_cpf_mrt",
          # "tfmri_nb_all_beh_cnf_rate","tfmri_nb_all_beh_cnf_mrt",
          # "tfmri_nb_all_beh_cngf_rate","tfmri_nb_all_beh_cngf_mrt",
          # "tfmri_nb_all_beh_cplace_rate","tfmri_nb_all_beh_cplace_mrt",
          "tfmri_rec_all_beh_place_dp","tfmri_rec_all_beh_negf_dp",
          "tfmri_rec_all_beh_neutf_dp","tfmri_rec_all_beh_posf_dpr",
          'interview_age','sex')

followup <- followup %>% select(all_of(behs))
#summary(followup)
followup <- na.omit(followup)


baseline <- baseline %>% select(all_of(behs))
baseline <- na.omit(baseline)


# 
# baseline_predict <- baseline %>% drop_na()
# subs <- sapply(baseline_predict$src_subject_id, function(string) gsub("NDAR_", "sub-NDAR", string))
# baseline_predict$src_subject_id <- subs
# baseline_predict$gender <- 1
# baseline_predict$gender[baseline_predict$sex == 'M'] <- 0
# write.csv(baseline_predict, '/home/mgell/Work/Prediction_HCP/text_files/ABCD_beh_all.csv', row.names = FALSE)



# make sure subjects are the same
baseline <- filter(baseline, src_subject_id %in% c(followup$src_subject_id))
followup <- filter(followup, src_subject_id %in% c(baseline$src_subject_id))

baseline <- baseline[order(baseline$src_subject_id), ]
followup <- followup[order(followup$src_subject_id), ]

mean(followup$interview_age - baseline$interview_age)/12

# For demographics
mean(baseline$interview_age)/12
sd(baseline$interview_age)/12

mean(followup$interview_age)/12
sd(followup$interview_age)/12



#subs <- baseline$src_subject_id

T1 <- followup %>% select(-c(eventname,src_subject_id,sex,interview_age))
T2 <- baseline %>% select(-c(eventname,src_subject_id,sex,interview_age))

#write.table(colnames(T1),'/home/mgell/Work/Prediction_HCP/code/opts/ABCD_behs2predict.txt',row.names = FALSE, col.names = FALSE, quote = FALSE)



# correlate and plot
cormat <- cor(T1,T2,use = "pairwise.complete.obs")
rel <- cormat[row(cormat)==col(cormat)]

ICC2 = numeric(length(colnames(T1)))
ICC3 = numeric(length(colnames(T1)))

i = 1
for (beh_i in colnames(T1)) {
  print(beh_i)
  x <- ICC(data.frame(T1[,beh_i],T2[,beh_i]))
  ICC2[i] = x$results$ICC[2]
  ICC3[i] = x$results$ICC[3]
  i = i+1
}

df <- data.frame('behs' = colnames(cormat), 'Reliability_r' = rel, 'Reliability_ICC2' = ICC2, 'Reliability_ICC3' = ICC3)



# load accuracy and save again with reliability colls
#d = read.csv('/home/mgell/Work/reliability/input/ridgeCV_zscore_averaged-source_HCP2016FreeSurferSubcortical_abcd_baselineYear1Arm1_rest_3517_zscored-beh_ABCD_beh_all_all_behs.csv')
d = read.csv('/home/mgell/Work/reliability/input/ridgeCV_zscore_confound_removal_wcategorical_averaged-source_HCP2016FreeSurferSubcortical_abcd_baselineYear1Arm1_rest_3517_zscored-beh_ABCD_beh_all_all_behs.csv')

d = d[-1,]

if (all(d$beh == df$behs)) {
  d$reliability_r = df$Reliability_r
  d$reliability_icc2 = df$Reliability_ICC2
  d$reliability_icc3 = df$Reliability_ICC3
} 


# save
write.csv(d,file = '/home/mgell/Work/reliability/res/ABCD_reliability_accuracy.csv', row.names = FALSE)
