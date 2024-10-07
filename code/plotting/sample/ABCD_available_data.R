
library(tidyverse)
library(dplyr)

headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_tbss01.txt', header = F, nrows = 1, as.is = T)
NIHTBX <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_tbss01.txt', skip = 2)
colnames(NIHTBX) <- headers


# behs = c('src_subject_id','eventname',
#          'nihtbx_picvocab_agecorrected', 'nihtbx_flanker_agecorrected', 'nihtbx_list_agecorrected',
#          'nihtbx_cardsort_agecorrected', 'nihtbx_pattern_agecorrected', 'nihtbx_picture_agecorrected',
#          'nihtbx_reading_agecorrected', 
#          'nihtbx_fluidcomp_agecorrected', 'nihtbx_cryst_agecorrected', 'nihtbx_totalcomp_agecorrected')

# alternatively use below for for t-scores
behs = c('src_subject_id','eventname',
         'interview_age','sex',
         'nihtbx_picvocab_fc', 'nihtbx_flanker_fc', 'nihtbx_list_fc',
         'nihtbx_cardsort_fc', 'nihtbx_pattern_fc', 'nihtbx_picture_fc',
         'nihtbx_reading_fc', 
         'nihtbx_fluidcomp_fc', 'nihtbx_cryst_fc', 'nihtbx_totalcomp_fc')

NIHTBX <- NIHTBX %>% select(all_of(behs))



headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_ps01.txt', header = F, nrows = 1, as.is = T)
RAVLT <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_ps01.txt', skip = 2)
colnames(RAVLT) <- headers

behs = c('src_subject_id','eventname',
         'pea_wiscv_tss',
         'pea_ravlt_sd_trial_vi_tc',  # immediate recall, I-V are learning trials
         'pea_ravlt_ld_trial_vii_tc') # 30 minute delay recall

RAVLT <- RAVLT %>% select(all_of(behs))



headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/lmtp201.txt', header = F, nrows = 1, as.is = T)
little_man <- read.table('/home/mgell/Work/abcd_pheno/phenotype/lmtp201.txt', skip = 2)
colnames(little_man) <- headers

behs = c('src_subject_id','eventname',
         'lmt_scr_perc_correct','lmt_scr_perc_wrong',     #percent
         'lmt_scr_num_correct','lmt_scr_num_wrong',       #number
         'lmt_scr_avg_rt','lmt_scr_rt_correct','lmt_scr_rt_wrong')   # RT

little_man <- little_man %>% select(all_of(behs))



# Followup only
# headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_yddss01.txt', header = F, nrows = 1, as.is = T)
# DD <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_yddss01.txt', skip = 2)
# colnames(DD) <- headers
# 
# behs <- c('ddis_scr_val_immedcho', # <3 indicates some inattentive (or at least irrational) behavior by the participant
#           'ddis_scr_expr_medrt_immedcho', 'ddis_scr_expr_mnrt_immcho',  #RT
#           'ddis_scr_val_indif_pnt_1yr','ddis_scr_val_indif_pnt_5yr','ddis_scr_val_indif_pnt_3mth')  # indif points



# Followup only
# headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_yest01.txt', header = F, nrows = 1, as.is = T)
# emo_stroop <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_yest01.txt', skip = 2)
# colnames(emo_stroop) <- headers
# 
# behs = c('src_subject_id','eventname',
#          )


# Followup only
# headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_gdss01.txt', header = F, nrows = 1, as.is = T)
# little_man <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_gdss01.txt', skip = 2)
# colnames(little_man) <- headers
# 
# behs = c('src_subject_id','eventname',
# )
# 
# 
# 
# 
# headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_siss01.txt', header = F, nrows = 1, as.is = T)
# little_man <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_siss01.txt', skip = 2)
# colnames(little_man) <- headers
# 
# behs = c('src_subject_id','eventname',
# )


headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_mid02.txt', header = F, nrows = 1, as.is = T)
MID <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_mid02.txt', skip = 2)
colnames(MID) <- headers

behs = c('src_subject_id','eventname',
         'tfmri_mid_beh_performflag',
         'tfmri_mid_all_beh_srwpfb_nt','tfmri_mid_all_beh_srwpfb_mrt',
         'tfmri_mid_all_beh_lrwpfb_nt','tfmri_mid_all_beh_lrwpfb_mrt')


MID <- MID %>% select(all_of(behs))
MID <- MID[MID$tfmri_mid_beh_performflag == 1,]
MID <- MID %>% select(-tfmri_mid_beh_performflag)


headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_sst02.txt', header = F, nrows = 1, as.is = T)
SST <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_sst02.txt', skip = 2)
colnames(SST) <- headers

behs = c('src_subject_id','eventname',
         'tfmri_sst_beh_performflag','tfmri_sst_beh_glitchflag',
         'tfmri_sst_all_beh_total_mssrt')

SST <- SST %>% select(all_of(behs))
SST <- SST[SST$tfmri_sst_beh_performflag == 1,]
SST <- SST[SST$tfmri_sst_beh_glitchflag == 0,]
SST <- SST %>% select(-tfmri_sst_beh_performflag)
SST <- SST %>% select(-tfmri_sst_beh_glitchflag)



headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_mrinback02.txt', header = F, nrows = 1, as.is = T)
eNback <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_mrinback02.txt', skip = 2)
colnames(eNback) <- headers

behs = c('src_subject_id','eventname',
         'tfmri_nback_beh_performflag',
         #'tfmri_nb_all_beh_ctotal_rate','tfmri_nb_all_beh_ctotal_mrt',
         'tfmri_nb_all_beh_c2b_rate','tfmri_nb_all_beh_c2b_mrt',
         'tfmri_nb_all_beh_c0b_rate','tfmri_nb_all_beh_c0b_mrt')#,
         # 'tfmri_nb_all_beh_cpf_rate','tfmri_nb_all_beh_cpf_mrt',     # this is for 0 and 2back. for 2back onl see: tfmri_nb_all_beh_c2bpf_mrt
         # 'tfmri_nb_all_beh_cnf_rate','tfmri_nb_all_beh_cnf_mrt',
         # 'tfmri_nb_all_beh_cngf_rate','tfmri_nb_all_beh_cngf_mrt',
         # 'tfmri_nb_all_beh_cplace_rate','tfmri_nb_all_beh_cplace_mrt',
         # 'tfmri_nb_all_beh_c2bpf_rate','tfmri_nb_all_beh_c2bpf_mrt',
         # 'tfmri_nb_all_beh_c2bnf_rate','tfmri_nb_all_beh_c2bnf_mrt',
         # 'tfmri_nb_all_beh_c2bngf_rate','tfmri_nb_all_beh_c2bngf_mrt',
         # 'tfmri_nb_all_beh_c2bp_rate','tfmri_nb_all_beh_c2bp_mrt')

eNback <- eNback %>% select(all_of(behs))
eNback <- eNback[eNback$tfmri_nback_beh_performflag == 1,]
eNback <- eNback %>% select(-tfmri_nback_beh_performflag)



headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/mribrec02.txt', header = F, nrows = 1, as.is = T)
RECMEM <- read.table('/home/mgell/Work/abcd_pheno/phenotype/mribrec02.txt', skip = 2)
colnames(RECMEM) <- headers

behs = c('src_subject_id','eventname',
         'tfmri_rec_all_beh_place_dp','tfmri_rec_all_beh_negf_dp',
         'tfmri_rec_all_beh_neutf_dp','tfmri_rec_all_beh_posf_dpr')

RECMEM <- RECMEM %>% select(all_of(behs))




# Mobile
headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_ssmty01.txt', header = F, nrows = 1, as.is = T)
screen <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_ssmty01.txt', skip = 2)
colnames(screen) <- headers

behs = c('src_subject_id','eventname',
         'stq_y_ss_weekday','stq_y_ss_weekend')

screen <- screen %>% select(all_of(behs))


# 
# # Fitbit weekly sum averages
# headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_fbwpas01.txt', header = F, nrows = 1, as.is = T)
# activity <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_fbwpas01.txt', skip = 2)
# colnames(activity) <- headers
# 
# behs = c('src_subject_id','eventname',
#          'fit_ss_meet_abcd_rule',
#          'fit_ss_wk_avg_steps','fit_ss_wk_avg_sedentary_min',
#          'fit_ss_wk_avg_light_active_min', 'fit_ss_wk_avg_farily_at_min', 'fit_ss_wk_avg_very_active_min',
#          'fit_ss_fitbit_rest_hr')
# 
# activity <- activity %>% select(all_of(behs))
# activity <- activity[activity$fit_ss_meet_abcd_rule == 1,]
# activity <- activity %>% select(-fit_ss_meet_abcd_rule)
# 
# averaged_activity <- aggregate(. ~ src_subject_id + eventname, data = activity, FUN = mean)
# 
# 
# 
# headers <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_fbwss01.txt', header = F, nrows = 1, as.is = T)
# sleep <- read.table('/home/mgell/Work/abcd_pheno/phenotype/abcd_fbwss01.txt', skip = 2)
# colnames(sleep) <- headers
# 
# behs = c('src_subject_id','eventname',
#          'fit_ss_sleep_avg_light_minutes','fit_ss_sleep_avg_deep_minutes','fit_ss_sleep_avg_rem_minutes')
# 
# sleep <- sleep %>% select(all_of(behs))
# averaged_sleep <- aggregate(. ~ src_subject_id + eventname, data = sleep, FUN = mean)





### MERGE 

merged_df <- inner_join(NIHTBX, RAVLT, by = c("src_subject_id", "eventname")) %>%
  inner_join(little_man, by = c("src_subject_id", "eventname")) %>%
  inner_join(MID, by = c("src_subject_id", "eventname")) %>%
  inner_join(SST, by = c("src_subject_id", "eventname")) %>%
  inner_join(eNback, by = c("src_subject_id", "eventname")) %>%
  inner_join(RECMEM, by = c("src_subject_id", "eventname")) %>%
  inner_join(screen, by = c("src_subject_id", "eventname")) #%>%


# before averaged_activity n = 14k, after n = 4k, with sleep n = 3.2k

baseline <- filter(merged_df, eventname == "baseline_year_1_arm_1")
followup <- filter(merged_df, eventname == "2_year_follow_up_y_arm_1")


write.csv(baseline, '/home/mgell/Work/reliability/text_files/abcd/baseline.csv', row.names = FALSE)
write.csv(followup, '/home/mgell/Work/reliability/text_files/abcd/followup.csv', row.names = FALSE)


# For demographics
mean(baseline$interview_age)/12
sd(baseline$interview_age)/12
