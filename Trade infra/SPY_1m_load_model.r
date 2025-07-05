library("caret")

#--------------------------------------------load models
lm_1_3 <- readRDS("./models/SPY_1m/SPY_1m_lm_1_3.rds")

lm_3_5 <- readRDS("./models/SPY_1m/SPY_1m_lm_3_5.rds")
glm_3_5 <- readRDS("./models/SPY_1m/SPY_1m_glm_3_5.rds")

glm_5_15 <- readRDS("./models/SPY_1m/SPY_1m_glm_5_15.rds")

#-----------------------------------------------Get input
input <- read.csv("./SPY_1m_current.csv")

pred_1_3 <- predict(lm_1_3,input,type = "response")

pred_lm_3_5 <- predict(lm_3_5,input,type = "response")
pred_glm_3_5 <- as.double(as.character(predict(glm_3_5,input,type = "response")))

pred_glm_5_15 <- as.double(as.character(predict(glm_5_15,input,type = "response")))

bullish_1_3 <- ifelse(pred_1_3>0.8634, 1, 0)
bullish_3_5 <- ifelse(pred_glm_3_5 > 0.5913871 & pred_lm_3_5 > 1.558466, 1, 0)
bullish_5_15 <- ifelse(pred_glm_5_15 > 0.5146253,1,0)

#--------------------------------------------Write out the predictions
write.csv(data.frame(bullish_1_3),"./SPY_1m_1_3_out.csv")
write.csv(data.frame(bullish_3_5),"./SPY_1m_3_5_out.csv")
write.csv(data.frame(bullish_5_15),"./SPY_1m_5_15_out.csv")