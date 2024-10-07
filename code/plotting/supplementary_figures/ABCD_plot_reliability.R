
library(tidyverse)

# Load data
d = read.csv('/home/mgell/Work/reliability/res/ABCD_reliability_accuracy.csv') 


# Plot Reliability and accuracy
plt = ggplot(data = d, aes(x = beh, y = reliability_r)) +
  geom_bar(stat = "identity", width = .75) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90),
        axis.title.x=element_blank()) +
  ylim(c(0,1.0)) +
  ylab('Reliability (r)')

ggsave('/home/mgell/Work/reliability/plots/ABCD/test_retest_beh_all.png', plot = plt)



plt = ggplot(data = d, aes(x = beh, y = reliability_icc3)) +
  geom_bar(stat = "identity", width = .75) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90),
        axis.title.x=element_blank()) +
  ylim(c(0,1.0)) +
  ylab('Reliability (ICC)')

ggsave('/home/mgell/Work/reliability/plots/ABCD/test_retest_beh_all_icc.png', plot = plt)



### Correlation of reliability and accuracy

# test-retest r

# test for correlation
test = cor.test(d$reliability_r,d$test_R2)

p_value <- format(test$p.value, digits = 2, scientific = TRUE)
conf_low <- round(test$conf.int[1], 2)
conf_high <- round(test$conf.int[2], 2)

result <- sprintf("r(%d) = %.2f, p = %s, 95%% CI [%.2f, %.2f]",
                  test$parameter, round(test$estimate, 2), 
                  p_value, 
                  conf_low, conf_high)

print(result)

result <- sprintf("r(%d) = %.2f, p < 0.001, 95%% CI [%.2f, %.2f]",
                  test$parameter, round(test$estimate, 2), 
                  conf_low, conf_high)

# plot
plt = ggplot() +
  geom_smooth(data=d, aes(reliability_r,test_R2), method = lm, se = FALSE, colour = 'lightgray', size = 1.5, alpha = 0.4) +
  geom_point(data=d, aes(reliability_r,test_R2), colour = '#56B4E9', size = 2) +
  theme_classic() +
  ylab('Prediction Accuracy (R2)') + xlab('Reliability (r)') +
  scale_x_continuous(limits = c(0,0.72), breaks = c(seq(0.0,0.7,0.1))) +
  scale_y_continuous(limits = c(-0.13,0.13), breaks = c(-0.12,-0.08,-0.04,0,0.04,0.08,0.12)) +
  annotate('text', x = 0.42, y = -0.112, label=result, size=4) +
  theme(axis.text = element_text(size = 10.5),
        axis.title = element_text(size = 10.5))

# Add dashed lines at 0.6 rel and 0
plt2 = plt + 
  #geom_segment(aes(x = 0.6, y = -0.1, xend = 0.6, yend = 0.2, colour= "red"), inherit.aes = FALSE, size=1, alpha=0.4, linetype= "dashed") +
  geom_segment(aes(x = 0, y = 0,    xend = max(d$reliability_r), yend = 0,  colour= "red"), inherit.aes = FALSE, size=1, alpha=0.4, linetype= "dashed") +
  theme(legend.position = "none")
ggsave('/home/mgell/Work/reliability/plots/ABCD/ABCD_all_behs_r2_r.png', plt2, width=4.5, height=3.5)





# ICC

# test for correlation
test = cor.test(d$reliability_icc3,d$test_R2)

p_value <- format(test$p.value, digits = 2, scientific = TRUE)
conf_low <- round(test$conf.int[1], 2)
conf_high <- round(test$conf.int[2], 2)

result <- sprintf("r(%d) = %.2f, p = %s, 95%% CI [%.2f, %.2f]",
                  test$parameter, round(test$estimate, 2), 
                  p_value, 
                  conf_low, conf_high)

print(result)

# adjust for printing in plot
result <- sprintf("r(%d) = %.2f, p < 0.001, 95%% CI [%.2f, %.2f]",
                  test$parameter, round(test$estimate, 2), 
                  conf_low, conf_high)

# plot
plt = ggplot() +
  geom_smooth(data=d, aes(reliability_icc3,test_R2), method = lm, se = FALSE, colour = 'lightgray', size = 1.5, alpha = 0.4) +
  geom_point(data=d, aes(reliability_icc3,test_R2), colour = '#56B4E9', size = 2) +
  theme_classic() +
  ylab('Prediction Accuracy (R2)') + xlab('Reliability (ICC)') +
  scale_x_continuous(limits = c(0,0.72), breaks = c(seq(0.0,0.7,0.1))) +
  scale_y_continuous(limits = c(-0.13,0.13), breaks = c(-0.12,-0.08,-0.04,0,0.04,0.08,0.12)) +
  annotate('text', x = 0.42, y = -0.112, label=result, size=4) +
  theme(axis.text = element_text(size = 10.5),
        axis.title = element_text(size = 10.5))

# Add dashed lines at 0.6 rel and 0
plt2 = plt + 
  #geom_segment(aes(x = 0.6, y = -0.1, xend = 0.6, yend = 0.2, colour= "red"), inherit.aes = FALSE, size=1, alpha=0.4, linetype= "dashed") +
  geom_segment(aes(x = 0, y = 0,    xend = max(d$reliability_r), yend = 0,  colour= "red"), inherit.aes = FALSE, size=1, alpha=0.4, linetype= "dashed") +
  theme(legend.position = "none")
ggsave('/home/mgell/Work/reliability/plots/ABCD/ABCD_all_behs_r2_icc3.png', plt2, width=4.5, height=3.5)


