
library(tidyverse)
library(ggExtra)


format_cor <- function(test) {
  
  p_value <- format(test$p.value, digits = 2, scientific = TRUE)
  conf_low <- round(test$conf.int[1], 2)
  conf_high <- round(test$conf.int[2], 2)
  
  result <- sprintf("r(%d) = %.2f, p = %s, 95%% CI [%.2f, %.2f]",
                    test$parameter, round(test$estimate, 2), 
                    p_value, 
                    conf_low, conf_high)
  
  return(result) 
}

### Figure 2A ###
# HCP-YA
plt_out = '/home/mgell/Work/reliability/plots/Figures/'

# load data
d = read_csv('/home/mgell/Work/reliability/res/HCP_YA_reliability_accuracy.csv')


# test for correlation
test = cor.test(d$reliability_icc2, d$test_R2)
res = format_cor(test)
print(res)

pdf(file = paste0(plt_out, "Fig2_A.pdf"), width = 3.5, height = 3, useDingbats = FALSE)
#p = 
  ggplot() +
  geom_smooth(data=d, aes(reliability_icc2,test_R2), method = lm, se = FALSE, colour = 'lightgray', size = 1.5, alpha = 0.4) +
  geom_point(data=d, aes(reliability_icc2,test_R2), colour = '#56B4E9', size = 2) +
  theme_classic() +
  ylab('Prediction Accuracy (R2)') + xlab('Reliability (ICC)') +
  scale_x_continuous(limits = c(0.2,1.0), breaks = c(seq(0.2,1.0,0.2))) +
  scale_y_continuous(limits = c(-0.13,0.12), breaks = c(0.12,0.08,0.04,0,-0.04,-0.08,-0.12)) +
  geom_segment(aes(x = 0.2, y = 0, xend = 1.0, yend = 0,  colour= "red"), 
               inherit.aes = FALSE, size=1, alpha=0.4, linetype= "dashed") +
  theme(legend.position = "none",
        axis.text = element_text(size = 10.5),
        axis.title = element_text(size = 10.5))

#pdf(file = paste0(plt_out, "Fig2_A.pdf"), width = 4, height = 3.5, useDingbats = FALSE)
#ggMarginal(p, type = "density")
dev.off()

# Additional correaltions:
test_icc_upper = cor.test(d$reliability_icc2_upper,d$test_R2)
test_icc_lower = cor.test(d$reliability_icc2_lower,d$test_R2)

res = format_cor(test_icc_upper)
print(res)

res = format_cor(test_icc_lower)
print(res)

d_pos = d
d_pos = d_pos[d$test_R2 > 0,]
test_pos_only = cor.test(d_pos$reliability_icc2,d_pos$test_R2)

res = format_cor(test_pos_only)
print(res)




### Figure 2B ###
# UKB
rm(list=setdiff(ls(), "format_cor"))


plt_out = '/data/project/ukb_reliability_in_prediction/plots/Figures/'

# load data
d = read_csv('/data/project/ukb_reliability_in_prediction/res/UKB_reliability_accuracy.csv')


# test for correlation
test = cor.test(d$reliability_icc2, d$test_R2)

p_value <- format(test$p.value, digits = 2, scientific = TRUE)
conf_low <- round(test$conf.int[1], 2)
conf_high <- round(test$conf.int[2], 2)

result <- sprintf("r(%d) = %.2f, p = %s, 95%% CI [%.2f, %.2f]",
                  test$parameter, round(test$estimate, 2), 
                  p_value, 
                  conf_low, conf_high)

print(result)

pdf(file = paste0(plt_out, "Fig2_B.pdf"), width = 3.5, height = 3, useDingbats = FALSE)
#p = 
ggplot() +
  geom_smooth(data=d, aes(reliability_icc2,test_R2), method = lm, se = FALSE, colour = 'lightgray', size = 1.5, alpha = 0.4) +
  geom_point(data=d, aes(reliability_icc2,test_R2), colour = '#56B4E9', size = 2) +
  theme_classic() +
  ylab('Prediction Accuracy (R2)') + xlab('Reliability (ICC)') +
  scale_x_continuous(limits = c(0.2,0.82), breaks = c(seq(0.2,0.8,0.2))) +
  scale_y_continuous(limits = c(-0.13,0.12), breaks = c(0.12,0.08,0.04,0,-0.04,-0.08,-0.12)) +
  geom_segment(aes(x = 0.2, y = 0, xend = 0.82, yend = 0,  colour= "red"), 
               inherit.aes = FALSE, size=1, alpha=0.4, linetype= "dashed") +
  theme(legend.position = "none",
        axis.text = element_text(size = 10.5),
        axis.title = element_text(size = 10.5))

#pdf(file = paste0(plt_out, "Fig2_A.pdf"), width = 4, height = 3.5, useDingbats = FALSE)
#ggMarginal(p, type = "density")
dev.off()



