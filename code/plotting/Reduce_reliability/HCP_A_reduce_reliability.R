
library(psych)
library(ggplot2)
library(dplyr)
library(tidyr)
library(faux)


# create perfectly correlated vectors with a given vector (i.e. sample cor.)
# Credits to: 
# https://stats.stackexchange.com/questions/15011/generate-a-random-variable-with-a-defined-correlation-to-an-existing-variables



### SET UP ###
#set.seed(654321)
#reliability <- c(0.85, 0.75, 0.65) # desired correlation with empirical measure
true_reliability <- 0.94  # actual reliability of measure
new_reliability <- c(0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5)   # desired correlation with empirical measure
n <- 100  # how many simulated noisy datasets to make
designator <- '' # '_true_score' or ''

#sim_mean <- 100 # exact mean or comment out to use mean from empirical data
#sim_sd <- 15 # exact sd or comment out to use sd from empirical data
custom_scale = TRUE # if custom_scale is false, need to specify scale
#scale = 1.4 # 1.25??? 1.5 for cryst cog 1.4 for motor???

subsample = FALSE
if (subsample == TRUE) {
  m = 250 # full sample 551 for beh and 650 for age
  k = 10 # number of subsamples 
  outdir <- '/home/mgell/Work/Prediction_HCP/text_files/rel/subsamples/'
  #???????????????????????
  dat = read.csv('/home/mgell/Work/reliability/text_files/subs_motor_and_FC.csv')
  #???????????????????????
} else {
  k = 1
}

# maximum and minimum value for behavioural measurement
# below is based on scoring and interpretation manual for NIH Toolbox
# These only make sense when using age adjusted scores => mean 100 and SD 15 SD
maximum <- 160 # 4SD 
minimum <- 40  # e.g.: < 30 is motor dysfunction (more than 4SD)

outdir <- '/home/mgell/Work/Prediction_HCP/text_files/rel/'
beh <- 'nih_tlbx_agecsc_dominant'  #'nih_tlbx_agecsc_dominant' #'nih_crycogcomp_ageadjusted' nih_totalcogcomp_ageadjusted
plt_out <- '/home/mgell/Work/reliability/plots/'
fname <- 'HCP_A_motor' # HCP_A_motor or HCP_A_cryst or beh_HCP_A_total

tab <- read.csv(paste0('/home/mgell/Work/Prediction_HCP/text_files/', fname, '.csv'))
##############



# Function for creating vectors with exact correlation
noisier_beh <- function(y, rho, x) {
  ### input:
  #   - y:   'real' vector we want to be correlated with x
  #   - rho: correlation/reliability between y and x
  #   - x:   random vector (this is your noise) that is optimised to correlate
  #          with y
  
  y_res <- residuals(lm(x ~ y)) # residuals of x against y (removing y from x).
  # y_res is orthogonal to y
  rho * sd(y_res) * y + y_res * sd(y) * sqrt(1 - rho^2) # optimise for specific r
}


# Function for making sure that sampled data dont go past some max and min
min_max <- function(vec, min = 0, max = Inf) {
  vec[vec < min] <- min
  vec[vec > max] <- max
  return(vec)
}




# Load real data to add noise to
#d <- tab %>% select(Subject, Age, Gender, all_of(beh))
d <- tab %>% select(src_subject_id, interview_age, sex, all_of(beh))
d <- d %>% filter(!is.na(d[,beh]))

# remove outliers from behaviour
print('removing outliers!!!!')
d = d[d[,beh] <= (mean(d[,beh]) + 3*sd(d[,beh])),]
d = d[d[,beh] >= (mean(d[,beh]) - 3*sd(d[,beh])),]
df <- d


hist(df[,beh],breaks = 50,xlim = c(40,160),ylim = c(0,40))


# n subsamples
beh_label <- beh
for (k_i in seq(k)) {

# subsample
if (subsample == TRUE) {
  print('Subsampling...')
  d <- df %>% filter(src_subject_id %in% c(dat$src_subject_id))
  d <- sample_n(d,m)
  hist(d[,beh],breaks = 50,xlim = c(40,160),ylim = c(0,40))
  T1 <- d[,beh]
  beh <- paste0(beh,'_',m)
} else {
  #d <- df
  T1 <- d[,beh]
}


# Where to save plots
plt_out <- paste0(plt_out,beh,'/')
if (!file.exists(plt_out)){
  dir.create(file.path(plt_out))
}


# plot hist of empirical data and save
# png(file=paste0(plt_out,'hist_',beh,'.png'), width=400, height=400)
# hist(df[,beh_label],breaks = 50,xlim = c(40,160),ylim = c(0,40))
# dev.off()


# simulate noisier data
if (!exists('sim_mean')) {
  empirical_mean <- mean(T1)
  empirical_sd <- sd(T1)
}


# save actual sampled data
if (subsample == TRUE) {
  beh = paste0(beh,'_',k_i)
}

#write.csv(d, paste0(outdir,fname,designator,'.csv'),row.names = FALSE)

all_ICC = numeric(0)

# create datasets
for (rel_i in new_reliability) {

  writeLines(sprintf('\n\n Setting up for %f', rel_i))

  # First run 1000 times to get offset from sample mean.
  # This increases due to sampling+noise and increases inversely to noise.
  # Rather than adjusting for each new noisy behaviour this solution assures
  # Some variability of the mean across noisy samples
  # => it will not be exactly 100 on for each noisy behaviour sample.
  # This could also be overcome by zscoring - but given certain variables
  # are already normalised and adjusted by age this seems a better solution
  # Test variables
  all_cor_test <- numeric(1000)
  all_scalers <- numeric(1000)
  all_offsets <- numeric(1000)

  for (i in 1:1000) {
    # create a random vector that will be manipulated to have specific cor with beh
    x <- rnorm(n = length(T1), mean = empirical_mean, sd = empirical_sd)
    # now make this correlate with T1 at defined reliability (rel_i)
    T1_noisy <- noisier_beh(T1, rel_i, x) # Correlated but different scale
    scaler <- sd(T1_noisy)/empirical_sd   # For scaling data to same as T1
    T1_noisy_scaled <- round(T1_noisy/scaler, digits = 0) # Round as age is int
    mean_offset <- empirical_mean - mean(T1_noisy_scaled) # diff between means
    T1_noisy_ok <- T1_noisy_scaled + mean_offset # shift mean by diff in means
    # Now: 
    #   empirical_mean == mean(T1_noisy_ok)
    #   empirical_sd   == sd(T1_noisy_ok)
    
    # Save
    all_scalers[i] <- scaler
    all_offsets[i] <- mean_offset
    all_cor_test[i] <- cor(T1_noisy_ok,T1)
    
  }
  remove(scaler,mean_offset)
  
  #all_noise_test_mean = mean(all_noise_test)
  #noise_offset = round(mean(T1) - all_noise_test_mean,digits = 0)
  mean_offset = mean(all_offsets)
  scaler = mean(all_scalers)
  
  print('mean correlation between T1_noisy and T1 before offset adjustment')
  print(mean(all_cor_test))
  print('scaler size:')
  print(scaler)
  print('offset size:')
  print(mean_offset)

  # Now add noise and adjust for offset
  print(sprintf('Creating noisy datasets with reliability %f ...', rel_i))

  # save
  all_noisy <- matrix(0, nrow = length(T1), ncol = n)
  all_cor <- numeric(n)
  all_mean <- numeric(n)
  all_sd <- numeric(n)

  for (i in 1:n) {
    # create a random vector that will be manipulated to have specific cor with beh
    x <- rnorm(n = length(T1), mean = empirical_mean, sd = empirical_sd)

    #to be sligthly smaller as it gets inflated while adding noise. Now is the same as beh
    T1_noisy <- noisier_beh(T1, rel_i, x)
    ###T1_noisy_ok <- min_max(T1_noisy/10, min = minimum, max = maximum)
    T1_noisy_ok <- round((T1_noisy/scaler) + mean_offset, digits = 0)
    # values end up 10*higher than original values in T1
    # when adding noise. Dividing by 10 adjusts for it. Linear scaling doesnt
    # affect correlations so this makes no difference.

    # Save
    all_noisy[,i] <- T1_noisy_ok
    all_cor[i] <- cor(T1_noisy_ok,T1)
    all_mean[i] <- mean(T1_noisy_ok)
    all_sd[i] <- sd(T1_noisy_ok)

    beh_noise = data.frame('src_subject_id' = d$src_subject_id, 'placeholder' = T1_noisy_ok)
    names(beh_noise)[names(beh_noise) == "placeholder"] <- beh_label
    rel <- unlist(strsplit(as.character(rel_i), '[.]'))
    rel <- rel[2]
    # write.csv(beh_noise, paste0(outdir,fname,designator,'_wnoise_rel_0',rel,'_',i,'.csv'),row.names = FALSE)
  }

  # ICC
  # calculate this in case you want/need to use ot for plotting later
  all_noisy = data.frame(all_noisy)
  
  n_cols <- 100
  ICC2 <- matrix(NA, nrow = n_cols, ncol = n_cols)
  
  for (col_i in 1:(n_cols - 1)) {
    for (col_j in (col_i + 1):n_cols) {
      x <- tryCatch(ICC(data.frame(all_noisy[, col_i], all_noisy[, col_j])), error = function(e) NA)
      ICC <- tryCatch(x$results$ICC[2], error = function(e) NA)
      ICC2[col_i, col_j] <- ICC
      ICC2[col_j, col_i] <- ICC  # Symmetrically assign the value
    }
  }
  
  # Fill the diagonal with 1 since ICC of a variable with itself is always 1
  diag(ICC2) <- 1
  
  rm(ICC)
  
  ICC_ind = lower.tri(ICC2, diag = FALSE)
  ICC = ICC2[ICC_ind]
  ICC = mean(ICC,na.rm = TRUE)
  all_ICC = rbind(all_ICC,ICC)
  
  # Print stuff
  print(sprintf('Group mean and SD for %s', beh))
  print(empirical_mean)
  print(empirical_sd)

  print(sprintf('Average mean and SD for all simulated datasets %s', beh))
  print(mean(all_noisy))
  print(mean(apply(all_noisy,2,sd)))

  # Check a few noisy examples
  print('example correlation:')
  print(cor(all_noisy[,1],T1))

  # correlation between T1_noisy and T1
  print('correlation of T1_noisy and T1 after offset adjustment')
  print(mean(all_cor))

  # Create and save plots
  eg1 = 50
  eg2 = 5

  png(file=paste0(plt_out,'hist_example1_',beh,designator,'_wnoise_rel_',rel_i,'.png'), width=400, height=400)
  hist(all_noisy[,eg1],breaks = 50,xlim = c(40,160),ylim = c(0,40))
  dev.off()

  png(file=paste0(plt_out,'hist_example2_',beh,designator,'_wnoise_rel_',rel_i,'.png'), width=400, height=400)
  hist(all_noisy[,eg2],breaks = 50,xlim = c(40,160),ylim = c(0,40))
  dev.off()

  noisy_data = data.frame('Noisy_means' = all_mean, 'Noisy_sd' = all_sd)
  plt1 = ggplot(noisy_data, aes(x = Noisy_means)) + geom_histogram(col = "black", fill = "skyblue3", alpha=0.5, bins = 15) +
    geom_segment(size= 2, aes(x = mean(T1), y = 0, xend = mean(T1), yend = 5, colour= "red")) +
    theme_classic() + theme(legend.position="none") + ylab('count') + xlab('mean of noisy datasets') +
    coord_cartesian(xlim = c(empirical_mean - 15, empirical_mean + 15))
  ggsave(paste0(plt_out,'hist_all_means_',beh,designator,'_wnoise_rel_',rel_i,'.png'), plt1, width=4, height=4)

  plt2 = ggplot(noisy_data, aes(x = Noisy_sd)) + geom_histogram(col = "black", fill = "skyblue3", alpha=0.5, bins = 15) +
    geom_segment(size= 2, aes(x = sd(T1), y = 0, xend = sd(T1), yend = 5, colour= "red")) +
    theme_classic() + theme(legend.position="none") + ylab('count') + xlab('mean of noisy datasets') +
    coord_cartesian(xlim = c(empirical_sd - 3, empirical_sd + 3))
  ggsave(paste0(plt_out,'hist_all_SD_',beh,designator,'_wnoise_rel_',rel_i,'.png'), plt2, width=4, height=4)

  all_noisy_d = data.frame('Actual_data' = T1, 'Example_1_noisy_data' = all_noisy[,eg1], 'Example_2_noisy_data' = all_noisy[,eg2])
  plt3 = ggplot(all_noisy_d,aes(x=Actual_data,y=Example_1_noisy_data)) +
    geom_point(size=2.5, alpha=0.6, colour="skyblue3") + theme_classic() +
    annotate('text', x = 85, y = 135, label=paste("Correlation:", round(cor(all_noisy_d$Actual_data, all_noisy_d$Example_1_noisy_data),digits = 2)), size=5)
  ggsave(paste0(plt_out,'plt_example1_',beh,designator,'_wnoise_rel_',rel_i,'.png'), plt3, width=4, height=4)

  plt4 = ggplot(all_noisy_d,aes(x=Actual_data,y=Example_2_noisy_data)) +
    geom_point(size=2.5, alpha=0.6, colour="skyblue3") + theme_classic() +
    annotate('text', x = 85, y = 135, label=paste("Correlation:", round(cor(all_noisy_d$Actual_data, all_noisy_d$Example_2_noisy_data),digits = 2)), size=5)
  ggsave(paste0(plt_out,'plt_example2_',beh,designator,'_wnoise_rel_',rel_i,'.png'), plt4, width=4, height=4)

  #plot(all_noisy[,50], T1)
  #plot(all_noisy[,5], T1)
  #plot(all_noisy[,78], T1)

  # Save whole noisy dataset for future reference
  write.csv(all_noisy, paste0('/home/mgell/Work/reliability/text_files/',fname,designator,'_wnoise_rel_',rel_i,'.csv'))

}

  beh <- beh_label
  ICCs = data.frame('reliability' = new_reliability, 'ICC2' = all_ICC)
  write.csv(ICCs, paste0('/home/mgell/Work/reliability/text_files/', fname, '_rel_ICC_simulations.csv'))
  
}

print('DONE!!')



