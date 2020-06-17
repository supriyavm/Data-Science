##########Random Forest###########
install.packages("csv")
install.packages("rsample")
install.packages("randomForest")
install.packages("caret")
library(randomForest)
library("csv")
library("rsample")
original_data <- read.csv("C:\\Users\\murty\\Desktop\\final.csv", 1)

# fill the NA with 0
is.na(original_data)
original_data[is.na(original_data)] <- 0
#check the data set
str(original_data)
layout(matrix(c(1),2,2)) # optional 4 graphs/page 
set.seed(1000)



original_data$Nationality <- as.numeric(as.factor(original_data$Nationality))
original_data$Club <- as.numeric(as.factor(original_data$Club))
original_data$Acceleration <- as.numeric(as.factor(original_data$Acceleration))
original_data$Aggression <- as.numeric(as.factor(original_data$Aggression))
original_data$Agility <- as.numeric(as.factor(original_data$Agility))
original_data$Balance <- as.numeric(as.factor(original_data$Balance))
original_data$Ball_control <- as.numeric(as.factor(original_data$Ball_control))
original_data$Composure <- as.numeric(as.factor(original_data$Composure))
original_data$Curve <- as.numeric(as.factor(original_data$Curve))
original_data$Dribbling <- as.numeric(as.factor(original_data$Dribbling))
original_data$Finishing <- as.numeric(as.factor(original_data$Finishing))
original_data$Free_kick_accuracy <- as.numeric(as.factor(original_data$Free_kick_accuracy))
original_data$GK_diving <- as.numeric(as.factor(original_data$GK_diving))
original_data$GK_reflexes <- as.numeric(as.factor(original_data$GK_reflexes))
original_data$Heading_accuracy <- as.numeric(as.factor(original_data$Heading_accuracy))
original_data$Interceptions <- as.numeric(as.factor(original_data$Interceptions))
original_data$Jumping <- as.numeric(as.factor(original_data$Jumping))
original_data$Long_passing <- as.numeric(as.factor(original_data$Long_passing))
original_data$Long_shots <- as.numeric(as.factor(original_data$Long_shots))
original_data$Marking <- as.numeric(as.factor(original_data$Marking))
original_data$Penalties <- as.numeric(as.factor(original_data$Penalties))
original_data$Short_passing <- as.numeric(as.factor(original_data$Short_passing))
original_data$Shot_power <- as.numeric(as.factor(original_data$Shot_power))
original_data$Sliding_tackle <- as.numeric(as.factor(original_data$Sliding_tackle))
original_data$Sprint_speed <- as.numeric(as.factor(original_data$Sprint_speed))
original_data$Stamina <- as.numeric(as.factor(original_data$Stamina))
original_data$Standing_tackle <- as.numeric(as.factor(original_data$Standing_tackle))
original_data$Strength <- as.numeric(as.factor(original_data$Strength))
original_data$Vision <- as.numeric(as.factor(original_data$Vision))
original_data$Volleys <- as.numeric(as.factor(original_data$Volleys))
original_data$Preferred_Positions <- as.numeric(as.factor(original_data$Preferred_Positions))
original_data$Crossing <- as.numeric(as.factor(original_data$Crossing))
original_data$Positioning <- as.numeric(as.factor(original_data$Positioning))
original_data$Reactions <- as.numeric(as.factor(original_data$Reactions))
dim(original_data)
names(original_data)
class(original_data)
str(original_data)
summary(original_data)
plot(original_data[,c()],pch=3)
suppressMessages(library(caret))
cor(original_data,original_data$Value_Num)
set.seed(12345)
indexes= sample(1:nrow(original_data), size=0.8*nrow(original_data))
train_data<-original_data[indexes,]
test_data<-original_data[-indexes,]

fit.rf <- randomForest(formula = Value_Num ~ ., data = train_data)
pred.rf <- predict(fit.rf, test_data)
error<-(pred.rf) - test_data$Value_Num
rmse.rf <- sqrt(sum(((pred.rf) - test_data$Value_Num)^2)/
                  length(test_data$Value_Num))
c(RMSE = rmse.rf, pseudoR2 = mean(fit.rf$rsq))
plot(error,test_data$Value_Num, xlab = "Attribules", ylab = "Value of Player", pch = 3)
