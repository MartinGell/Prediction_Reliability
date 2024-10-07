
library(ggplot2)
library(dplyr)
library(tidyr)
library(faux)


# create perfectly correlated vectors with a given vector (i.e. sample cor.)
# Credits to: 
# https://stats.stackexchange.com/questions/15011/generate-a-random-variable-with-a-defined-correlation-to-an-existing-variables



### SET UP ###
#reliability <- c(0.85, 0.75, 0.65) # desired correlation with empirical measure
true_reliability <- 1.0  # actual reliability of measure
new_reliability <- c(0.85) #c(0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5)    # desired correlation with empirical measure
n <- 100  # how many simulated noisy datasets to make

outdir <- '/data/project/ukb_reliability_in_prediction/text_files/rel/subsamples/'
beh <- 'Age_when_attended_assessment_centre'
plt_out <- '/data/project/ukb_reliability_in_prediction/plots/UKB/'
fname <- 'UKB_5000_subs_FC_all_cogs'
beh_label = paste0(beh,'.2.0')

tab <- read.csv(paste0('/data/project/ukb_reliability_in_prediction/text_files/', fname, '.csv'))
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


# function for removing too young or old participants to preserve similar data
# structure to original data
ok_age <- function(vec,extreme) {
  if (extreme == 'young') {
    new_vec = vec
    replacement = round(vec[vec < 40] + abs(rnorm(n = length(vec[vec < 40]), mean = 0, sd = 50)), digits = 0)
    new_vec[vec < 40] = replacement
    return(new_vec)
  } else if(extreme == 'old') {
    new_vec = vec
    replacement = round(vec[vec > 90] - abs(rnorm(n = length(vec[vec > 90]), mean = 0, sd = 50)), digits = 0)
    new_vec[vec > 90] = replacement
    return(new_vec)
  } else {
    print('WRONG OPTION, SKIPPING...')
  }
}



# Where to save plots
plt_out <- paste0(plt_out,beh,'/')
if (!file.exists(plt_out)){
  dir.create(file.path(plt_out))
}

# Load real data to add noise to
d <- tab %>% select(eid, sex, all_of(beh_label))
d <- d %>% filter(!is.na(d[,beh_label]))

# remove outliers from behaviour
#d = d[d[,beh_label] <= (mean(d[,beh_label]) + 3*sd(d[,beh_label])),]
#d = d[d[,beh_label] >= (mean(d[,beh_label]) - 3*sd(d[,beh_label])),]


# simulate noisier data
T1 <- d[,beh_label]

PDF = density(T1, bw = bw.SJ(T1, tol=0.9), kernel = 'gaussian', from = min(T1), to = max(T1))

PDF_plt = PDF
PDF_plt$y = PDF_plt$y * 5000

png(file=paste0(plt_out,'hist_',beh,'.png'), width=400, height=400)
hist(d[,beh_label],breaks = 30,xlim = c(40,90))
lines(PDF_plt, lwd = 2, col = 'red')
dev.off()

#ggplot(data.frame('T1' = T1),aes(T1)) + geom_histogram(bins = 50) + xlim(c(300,1300)) + ylim(c(0,40)) + geom_density()

empirical_mean <- mean(T1)
empirical_sd <- sd(T1)

names(d)[names(d) == beh_label] <- paste0(beh,'-2.0')
write.csv(d, paste0(outdir,beh,'.csv'),row.names = FALSE)

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
    x <- rnorm(n = length(T1), mean = sample(T1, size = length(T1), replace = TRUE), PDF$bw)
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
    x <- rnorm(n = length(T1), mean = sample(T1, size = length(T1), replace = TRUE), PDF$bw)
    #to be sligthly smaller as it gets inflated while adding noise. Now is the same as beh
    T1_noisy <- noisier_beh(T1, rel_i, x)
    T1_noisy_ok <- round((T1_noisy/scaler) + mean_offset, digits = 0)
    # values end up 10*higher than original values in T1
    # when adding noise. Dividing by 10 adjusts for it. Linear scaling doesnt
    # affect correlations so this makes no difference.
    
    # Make sure there are no extra young or extra old participants
    # (within a margin of acceptable variation outside the sample < 300, > 1200)
    while (min(T1_noisy_ok) < 40) {
      print('fixing too young')
      T1_noisy_ok = ok_age(T1_noisy_ok,'young')
    }
    while (max(T1_noisy_ok) > 90) {
      print('fixing too old')
      T1_noisy_ok = ok_age(T1_noisy_ok,'old')
    }
    
    # Save
    all_noisy[,i] <- T1_noisy_ok
    all_cor[i] <- cor(T1_noisy_ok,T1)
    all_mean[i] <- mean(T1_noisy_ok)
    all_sd[i] <- sd(T1_noisy_ok)
    
    rel <- unlist(strsplit(as.character(rel_i), '[.]'))
    rel <- rel[2]
    
    beh_noise = data.frame('src_subject_id' = d$eid, 'interview_age' = T1_noisy_ok)
    names(beh_noise)[names(beh_noise) == "interview_age"] <- paste0(beh,'-2.0')
    write.csv(beh_noise, paste0(outdir,beh,'_wnoise_rel_0',rel,'_',i,'.csv'),row.names = FALSE)
  }
  
  # Print stuff
  print(sprintf('Group mean and SD for %s', beh))
  print(empirical_mean)
  print(empirical_sd)
  
  print(sprintf('Average mean and SD for all simulated datasets %s', beh))
  print(mean(all_noisy))
  print(sd(all_noisy))
  
  # Check a few noisy examples
  print('example correlation:')
  print(cor(all_noisy[,1],T1))
  
  # correlation between T1_noisy and T1 
  print('correlation of T1_noisy and T1 after offset adjustment')
  print(mean(all_cor))
  
  # Create and save plots
  eg1 = 50
  eg2 = 5
  
  png(file=paste0(plt_out,'hist_example1_',beh,'_wnoise_rel_',rel_i,'.png'), width=400, height=400)
  hist(all_noisy[,eg1],breaks = 30,xlim = c(40,90))
  dev.off()
  
  png(file=paste0(plt_out,'hist_example2_',beh,'_wnoise_rel_',rel_i,'.png'), width=400, height=400)
  hist(all_noisy[,eg2],breaks = 30,xlim = c(40,90))
  dev.off()
  
  noisy_data = data.frame('Noisy_means' = all_mean, 'Noisy_sd' = all_sd) 
  plt1 = ggplot(noisy_data, aes(x = Noisy_means)) + geom_histogram(col = "black", fill = "skyblue3", alpha=0.5, bins = 15) +
    geom_segment(size= 2, aes(x = mean(T1), y = 0, xend = mean(T1), yend = 10, colour= "red")) + 
    theme_classic() + theme(legend.position="none") + ylab('count') + xlab('mean of noisy datasets') +
    coord_cartesian(xlim = c(empirical_mean - 3, empirical_mean + 3))
  ggsave(paste0(plt_out,'hist_all_means_',beh,'_wnoise_rel_',rel_i,'.png'), plt1, width=4, height=4)
  
  plt2 = ggplot(noisy_data, aes(x = Noisy_sd)) + geom_histogram(col = "black", fill = "skyblue3", alpha=0.5, bins = 15) +
    geom_segment(size= 2, aes(x = sd(T1), y = 0, xend = sd(T1), yend = 10, colour= "red")) + 
    theme_classic() + theme(legend.position="none") + ylab('count') + xlab('SD of noisy datasets') +
    coord_cartesian(xlim = c(empirical_sd - 0.5, empirical_sd + 0.5))
  ggsave(paste0(plt_out,'hist_all_SD_',beh,'_wnoise_rel_',rel_i,'.png'), plt2, width=4, height=4)
  
  all_noisy_d = data.frame('Actual_data' = T1, 'Example_1_noisy_data' = all_noisy[,eg1], 'Example_2_noisy_data' = all_noisy[,eg2])
  plt3 = ggplot(all_noisy_d,aes(x=Actual_data,y=Example_1_noisy_data)) + 
    geom_point(size=2.5, alpha=0.6, colour="skyblue3") + theme_classic() + 
    annotate('text', x = 55, y = 90, label=paste("Correlation:", round(cor(all_noisy_d$Actual_data, all_noisy_d$Example_1_noisy_data),digits = 2)), size=5)
  ggsave(paste0(plt_out,'plt_example1_',beh,'_wnoise_rel_',rel_i,'.png'), plt3, width=4, height=4)
  
  plt4 = ggplot(all_noisy_d,aes(x=Actual_data,y=Example_2_noisy_data)) + 
    geom_point(size=2.5, alpha=0.6, colour="skyblue3") + theme_classic() + 
    annotate('text', x = 55, y = 90, label=paste("Correlation:", round(cor(all_noisy_d$Actual_data, all_noisy_d$Example_2_noisy_data),digits = 2)), size=5)
  ggsave(paste0(plt_out,'plt_example2_',beh,'_wnoise_rel_',rel_i,'.png'), plt4, width=4, height=4)
  
  #plot(all_noisy[,50], T1)
  #plot(all_noisy[,5], T1)
  #plot(all_noisy[,78], T1)
  
  # Save whole noisy dataset for future reference
  write.csv(all_noisy, paste0('/data/project/ukb_reliability_in_prediction/text_files/',beh,'_wnoise_rel_',rel_i,'.csv'))
  
}

print('DONE!!')

