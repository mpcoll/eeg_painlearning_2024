#-*- coding: utf-8  -*-

# Author: michel-pierre.coll
# Date: 2020-07-23 12:01:58
# Description:


# Load packages
library(lme4)

# Load data
data <- read.csv("derivatives/task-fearcond_alldata.csv")

# Remove reinforced trials
dfns <- data[data$"cond" != "CS++", ]

# Get rid of excluded participants with bad SCR or eeg
dfns2 <- dfns[!(dfns$"sub" %in% c("sub-31", "sub-35", "sub-42", "sub-55")), ]

# remove block1
dfns2 <- dfns2[!(dfns2$block == 1), ]

model <- lmer(scr ~ cond + block + cond:block + (1 | sub), data = dfns2, REML = FALSE)
drop1(model, scope = "cond", test = "Chisq")
drop1(model, test = "Chisq")

# Test for each condition
dfns_test <- dfns2[(dfns2$cond == "CS+"), ]
dfns_test$block <- as.numeric(dfns_test$block)
model <- lmer(scr ~ block + (1 | sub), data = dfns_test, REML = FALSE)
drop1(model, test = "Chisq")

dfns_test <- dfns2[(dfns2$cond == "CS-1"), ]
dfns_test$block <- as.numeric(dfns_test$block)
model <- lmer(scr ~ block + (1 | sub), data = dfns_test, REML = FALSE)
drop1(model, test = "Chisq")

dfns_test <- dfns2[(dfns2$cond == "CS-2"), ]
dfns_test$block <- as.numeric(dfns_test$block)
model <- lmer(scr ~ block + (1 | sub), data = dfns_test, REML = FALSE)
drop1(model, test = "Chisq")

dfns_test <- dfns2[!(dfns2$cond == "CS-E"), ]
dfns_test$block <- as.numeric(dfns_test$block)
model <- lmer(scr ~ block + (1 | sub), data = dfns_test, REML = FALSE)
drop1(model, test = "Chisq")
