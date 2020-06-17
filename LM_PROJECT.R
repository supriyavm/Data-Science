#load original
install.packages("csv")
library(csv)
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
original_data$GK_handling <- as.numeric(as.factor(original_data$GK_handling))
original_data$GK_kicking <- as.numeric(as.factor(original_data$GK_kicking))
original_data$GK_positioning <- as.numeric(as.factor(original_data$GK_positioning))
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

#check the data set
str(original_data)

#divide the original data into trainset and testset
train_rows <- sample(nrow(original_data), .8*nrow(original_data))
train <- original_data[train_rows, ]
test <- original_data[-train_rows, ]

str(original_data)


#remove the dependent(Value_Num/Value) and identifier(Name) variables
train_para <- subset(train, select = -c(Value_Num, Value))
test_para <- subset(test, select = -c(Value_Num, Value))
str(train_para)
str(test_para)
head(test_para)

##########################   principal component analysis    ####################################
prin_comp <- prcomp(train_para, center=TRUE,scale=TRUE)
names(prin_comp)
#outputs the mean of variables
prin_comp$center
#outputs the standard deviation of variables
prin_comp$scale
prin_comp$rotation
summary(prin_comp)
dim(prin_comp$x)
#biplot(prin_comp, scale = 0)
#compute standard deviation of each principal component
std_dev <- prin_comp$sdev
#compute variance
pr_var <- std_dev^2
#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
#scree plot
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")
#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

#This plot shows that 30 components results in variance close to ~ 98%. 
#Therefore, in this case, we'll select number of components as 30 [PC1 to PC30] and proceed to the modeling stage. 
#This completes the steps to implement PCA on train data. 
#For modeling, we'll use these 30 components as predictor variables and follow the normal procedures.

#add a training set with principal components
train.data <- data.frame(Value_Num = train$Value_Num, prin_comp$x)
#we are interested in first 30 PCAs
train.data <- train.data[,1:31]
train.data

#transform test into PCA
test.data <- predict(prin_comp, newdata = test_para)
test.data <- as.data.frame(test.data)
#select the first 30 components
test.data <- test.data[,1:30]






###########################  decision tree ################################
library(rpart)
library(rpart.plot)
rpart.model <- rpart(Value_Num ~ .,data = train.data, method = "anova")

# rpart.model <- rpart(Value_Num ~Age + Overall,data = train, method = "anova")

rpart.model
prp(rpart.model, main = "Value_Num")
rpart.plot(rpart.model)

#make prediction on test data
rpart.prediction <- predict(rpart.model, test.data)
summary(rpart.prediction)
str(rpart.prediction)
str(test$Value_Num)

err.rpart <- test$Value_Num - rpart.prediction
rmse.rpart <- sqrt(mean((err.rpart^2)))
rmse.rpart
#Errors histogram
hist(err.rpart, main="Value_Num", sub="(Actual-Predicted)", xlab="Error", breaks=10, col="darkred")






##########################   linear regression  ####################################
#####  not use PCA ######
#remove the Value(not Value_Num) and identifier(Name) variables
train1 <- subset(train, select = -c(Value))
test1 <- subset(test, select = -c(Value))
str(train1)

nP_linear.model1 <- lm(train1$Value_Num ~ .,data = train1)
layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(nP_linear.model1)
summary(nP_linear.model1)
# Coefficients: (16 not defined because of singularities) for highly correlated variables
alias(nP_linear.model1)
require(dplyr)
library(dplyr)
summary(lm(train1$Value_Num ~ .,data = train1))$coefficients
summary(lm(train1$Value_Num ~ .,data = train1))$r.squared

# choose the predictor variables by p-value
# only retain Age and Overall
nP_linear.model2 <- lm(Value_Num ~ Age + Overall, data = train1) #not use train1$****
layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(nP_linear.model2)
summary(nP_linear.model2)

#make prediction on test data
nP_linear.prediction <- predict(nP_linear.model2, newdata = test1)
nP_linear.prediction
summary(nP_linear.prediction)
str(nP_linear.prediction)

err.nP_linear <- test1$Value_Num - nP_linear.prediction
rmse.nP_linear <- sqrt(mean((err.nP_linear^2)))
rmse.nP_linear
#Errors histogram
layout(matrix(c(1),2,2))
hist(err.nP_linear, main="Value_Num", sub="(Actual-Predicted)", xlab="Error", breaks=10, col="darkred")

###################################Random Forest######################################


install.packages("rsample")
install.packages("randomForest")
install.packages("caret")
library(randomForest)
library("csv")
library("rsample")
library(tidyr)
fit.rf <- randomForest(Value_Num ~ Age + Overall, data = train1)
pred.rf <- predict(fit.rf, test1)

rmse.rf <- sqrt(sum(((pred.rf) - test1$Value_Nm)^2)/
                  length(test_data$Value_Num))
c(RMSE = rmse.rf, pseudoR2 = mean(fit.rf$rsq))
plot(pred.rf,test_data$Value_Num, xlab = "Attribules", ylab = "Value of Player", pch = 3)




#####  use PCA ######
P_linear.model1 <- lm(Value_Num ~ .,data = train.data)
layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(P_linear.model1)
summary(P_linear.model1)

