
# Get HCP A unrelated sample with all 4 REST scans

library(tidyverse)

# all data
all_data <- tibble(read_csv('/home/mgell/Work/Preprocess_HCP/text_files/HCA_LS_2.0_subject_completeness.csv'))
all_data <- all_data[-1,]

#all_data <- subset(all_data, select = c(participant_id, session, SUB_STUDY, VISIT, age, sex, isPatient, wasPatient))

# Filter only unrelated subjects
d <- filter(all_data, unrelated_subset == TRUE)
d <- filter(d, !is.na(interview_age)) # all have age recorded
d <- filter(d, `RS-fMRI_Count` == 4)
#d <- filter(d, `tMRI_CARIT_PctCompl` == 100)

# Save sub list for denoising
subs = d %>% select(src_subject_id)
#write_delim(subs, '/home/mgell/Work/Preprocess_HCP/code/subs/subs_HCP_A.txt', delim = '\t', col_names = FALSE)





####
# need to make sure we are doing simulations based on only subs we have data for
# as using leo's subs need to filter these out first
subs_leo <- tibble(read_csv('/home/mgell/Work/Prediction_HCP/text_files/subs_leo.csv'))
d_leo = filter(d, src_subject_id %in% c(subs_leo$src_subject_id))
subs = d_leo %>% select(src_subject_id)


#### movement
FD = tibble(read_csv('/home/mgell/Work/Prediction_HCP/text_files/HCP_A_FD_mean.csv'))
FD$FDz = (FD$FD - mean(FD$FD))/sd(FD$FD)
subs_lowmo = FD[FD$FDz < 3,]

subs = filter(subs, src_subject_id %in% c(subs_lowmo$eid))



#### All subs with age ####
d <- filter(d, !is.na(interview_age)) # all have age recorded
d <- d %>% select(src_subject_id, sex, interview_age)
age_imaging_only <- filter(d, src_subject_id %in% c(subs$src_subject_id))
write_csv(age_imaging_only, '/home/mgell/Work/Prediction_HCP/text_files/HCP_A_age.csv')



#### Subs with motor behaviour ####
mt <- tibble(read_delim('/home/mgell/Work/reliability/text_files/HCP_A_beh/tlbx_motor01.txt', delim = '\t'))
mt <- mt[-1,]
mt <- filter(mt, !is.na(nih_tlbx_agecsc_dominant))
mt <- mt %>% select(src_subject_id, sex, interview_age, nih_tlbx_domsc, nih_tlbx_nondomsc, nih_tlbx_agecsc_dominant, nih_tlbx_agecsc_nondom, grip_standardsc_dom, grip_standardsc_nondom)
mt_imaging_only <- filter(mt, src_subject_id %in% c(subs$src_subject_id))
write_csv(mt_imaging_only, '/home/mgell/Work/Prediction_HCP/text_files/beh_HCP_A_motor.csv')
# nih_tlbx_agecsc_dominant has NaN!!!!!!!!!!!!!!!


#### Subs with cryst intelligence ####
cryst <- tibble(read_delim('/home/mgell/Work/reliability/text_files/HCP_A_beh/cogcomp01.txt', delim = '\t'))
cryst <- cryst[-1,]
cryst <- filter(cryst, !is.na(nih_crycogcomp_ageadjusted))
cryst <- cryst %>% select(src_subject_id, sex, interview_age, nih_crycogcomp_ageadjusted, nih_crycogcomp_unadjusted, nih_crystalcogcomp_np)
cryst_imaging_only <- filter(cryst, src_subject_id %in% c(subs$src_subject_id))
write_csv(cryst_imaging_only, '/home/mgell/Work/Prediction_HCP/text_files/beh_HCP_A_cryst.csv')
#overlap_cryst <- filter(cryst, src_subject_id %in% c(d$src_subject_id))


#### Subs with total intelligence ####
cryst <- tibble(read_delim('/home/mgell/Work/reliability/text_files/HCP_A_beh/cogcomp01.txt', delim = '\t'))
cryst <- cryst[-1,]
cryst <- filter(cryst, !is.na(nih_totalcogcomp_ageadjusted))
cryst <- cryst %>% select(src_subject_id, sex, interview_age, nih_totalcogcomp_ageadjusted, nih_totalcogcomp_unadjusted, nih_totalcogcomp_np)
tot_imaging_only <- filter(cryst, src_subject_id %in% c(subs$src_subject_id))
write_csv(cryst_imaging_only, '/home/mgell/Work/Prediction_HCP/text_files/beh_HCP_A_total.csv')
#overlap_cryst <- filter(cryst, src_subject_id %in% c(d$src_subject_id))


#### Reading English ####
reading <- tibble(read_delim('/home/mgell/Work/reliability/text_files/HCP_A_beh/orrt01.txt', delim = '\t')) #reading decoding
reading <- reading[-1,]
reading <- filter(reading, !is.na(tbx_reading_score))
reading <- reading %>% select(src_subject_id, sex, interview_age, tbx_reading_score, read_acss, nih_tlbx_theta, nih_tlbx_se, wcst_ni)
reading_imaging_only <- filter(reading, src_subject_id %in% c(subs$src_subject_id))
write_csv(reading_imaging_only, '/home/mgell/Work/Prediction_HCP/text_files/beh_HCP_A_reading.csv')

#### Subs with reading & vocab ####
cryst <- tibble(read_delim('/home/mgell/Work/reliability/text_files/HCP_A_beh/tpvt01.txt', delim = '\t')) #reading
cryst <- tibble(read_delim('/home/mgell/Work/reliability/text_files/HCP_A_beh/orrt01.txt', delim = '\t')) #vocab compren

cryst <- cryst[-1,]
cryst <- filter(cryst, !is.na(nih_crycogcomp_ageadjusted))
cryst <- cryst %>% select(src_subject_id, sex, interview_age, nih_crycogcomp_ageadjusted, nih_crycogcomp_unadjusted, nih_crystalcogcomp_np)
write_csv(cryst, '/home/mgell/Work/Prediction_HCP/text_files/beh_HCP_A_cryst.csv')
overlap_cryst <- filter(cryst, src_subject_id %in% c(d$src_subject_id))


#### Subs with List sorting ####
lswm <- tibble(read_delim('/home/mgell/Work/reliability/text_files/HCP_A_beh/lswmt01.txt', delim = '\t'))
lswm <- lswm[-1,]
lswm <- filter(lswm, !is.na(tbx_ls))
lswm <- lswm %>% select(src_subject_id, sex, interview_age, tbx_ls, age_corrected_standard_score)
write_csv(lswm, '/home/mgell/Work/Prediction_HCP/text_files/beh_HCP_A_lswm.csv')
overlap_cryst <- filter(lswm, src_subject_id %in% c(d$src_subject_id))

# create subs.csv for extracting FD
files <- tibble(read.delim('/home/mgell/Work/FC/text_files/available_preproc_data.txt',sep = '_'))
available_subs <- paste0(files$sub, '-', files$n)
d <- filter(d_available, participant_id %in% c(available_subs))

sub_dirs <- paste0(d$participant_id, '/', d$session,'/')
subIDs <-   paste0(d$participant_id, '_', d$session, '_task-rest_acq-1400_desc-confounds_regressors.tsv')

subs = tibble('sub' = sub_dirs, 'file' = subIDs)

write_csv(subs, '/home/mgell/Work/FC/text_files/subs.csv')





# extra
df = read_delim('/home/mgell/Work/t.txt', delim = '.', col_names = FALSE)
#s = read_delim('/home/mgell/Work/Preprocess_HCP/code/subs/subs_HCP_A.txt', delim = '\t', col_names = FALSE)
#s = read_delim('/home/mgell/Work/Preprocess_HCP/code/subs/subs_HCP_A_CARIT.txt', delim = '\t', col_names = FALSE)
s = read_delim('/home/mgell/Work/Preprocess_HCP/code/subs/subs_HCP_A_missing.txt', delim = '\t', col_names = FALSE)
missing = df$X2 + 1 # adjust for 0 base indexing on juseless
missinsubs = s[c(missing),1]
write_delim(missinsubs, '/home/mgell/Work/Preprocess_HCP/code/subs/subs_HCP_A_missing.txt', delim = '.', col_names = FALSE)
