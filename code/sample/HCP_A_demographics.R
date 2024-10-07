

subs = tibble(read_csv('/home/mgell/Work/Preprocess_HCP/code/subs/subs_HCP_A.txt', col_names = c('src_subject_id')))



age_prediction = tibble(read_csv('/home/mgell/Work/Prediction_HCP/text_files/rel/interview_age.csv')) # after removing super old subjects
age_prediction = filter(age_prediction, src_subject_id %in% c(subs$src_subject_id))

age_prediction$interview_age = as.numeric(age_prediction$interview_age)/12
age_prediction$sex = as.factor(age_prediction$sex)
summary(age_prediction)
sd(age_prediction$interview_age)



crycog = tibble(read_csv('/home/mgell/Work/Prediction_HCP/text_files/rel/HCP_A_cryst.csv'))
crycog = tibble(read_csv('/home/mgell/Work/Prediction_HCP/text_files/beh_HCP_A_cryst.csv'))

crycog = filter(crycog, src_subject_id %in% c(subs$src_subject_id))

crycog$interview_age = as.numeric(crycog$interview_age)/12
crycog$sex = as.factor(crycog$sex)
summary(crycog)




mot = tibble(read_csv('/home/mgell/Work/Prediction_HCP/text_files/rel/HCP_A_motor.csv'))
mot = filter(mot, src_subject_id %in% c(subs$src_subject_id))

mot$interview_age = as.numeric(mot$interview_age)/12
mot$sex = as.factor(mot$sex)
summary(mot)




tot = tibble(read_csv('/home/mgell/Work/Prediction_HCP/text_files/rel/HCP_A_total.csv'))
tot = filter(tot, src_subject_id %in% c(subs$src_subject_id))

tot$interview_age = as.numeric(tot$interview_age)/12
tot$sex = as.factor(tot$sex)
summary(tot)



all = Reduce(intersect, list(crycog$src_subject_id,tot$src_subject_id,mot$src_subject_id))
