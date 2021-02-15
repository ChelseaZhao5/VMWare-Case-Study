library(tidyverse)
library(sqldf)
library(readxl)
library(dplyr)
library(mice)
library(caret)
library(car)
library(Matrix)
library(base)
library(data.table)
library(lattice)
library(ggplot2)
library(ggfortify)
library(forecast)
library(astsa)
library(xts)
library(tibble)
library(ggbiplot)
library(MASS)
library(ROCR)
library(lift)
library(glmnet)
library(e1071)
library(FactoMineR)
library(fastDummies)
library(caTools)
library(corrplot)

TrainData <- read.csv("Training.csv")
ValidationData<-read.csv("Validation.csv")
total<-rbind(TrainData,ValidationData)

#check if dependent variables have missing values, none of the these variables have zero
sum(is.na(total$tgt_hol)=="TRUE")
sum(is.na(total$tgt_eval)=="TRUE")
sum(is.na(total$tgt_webinar)=="TRUE")
sum(is.na(total$tgt_seminar)=="TRUE")
sum(is.na(total$tgt_download)=="TRUE")
sum(is.na(total$tgt_whitepaper)=="TRUE")

#check how many customers took the target action
sum(total$tgt_hol)  #1111
sum(total$tgt_eval)  #0
sum(total$tgt_first_date_eval_page_view)  #50
sum(total$tgt_webinar)  #43
sum(total$tgt_first_date_webinar_page_view)  #45
sum(total$tgt_seminar)  #0
sum(total$tgt_download)   #1776
sum(total$tgt_whitepaper)   #21
#create a group include target actions of webinar, seminar, evaluation, and whitepaper
tgt_group<-ifelse(total$tgt_webinar==1|total$tgt_first_date_eval_page_view==1|total$tgt_whitepaper==1,1,0)

num <- select_if(total, is.numeric)
cat<-total %>% select_if(negate(is.numeric))
for(i in 1:ncol(num)){num[is.na(num[,i]), i] <- mean(num[,i], na.rm = TRUE)}
#count(is.na(num$db_annualsales)=="TRUE") used this to check if NAs are fixed

#for all categorical varibales, change NA to "None"
for(i in 1:ncol(cat)){cat[is.na(cat[,i]), i] <- "None"}

#Check all columns dont have NA and combine the dataset
colnames(cat)[colSums(is.na(cat)) > 0]
colnames(num)[colSums(is.na(num)) > 0]

# deleting all-0 columns for numerical variables
num <- num[,-(which(colSums(num) == 0))]
#deleting numerical columns contain more than 90% of 0
num <- num[,!colSums(num != 0) < 90000]
#drop the ones with no varaince
num <-num[ , which(apply(num, 2, var) != 0)]
#remove column of list/order
num<-subset(num,select=-c(X))
cor(num)
sum(cor(num)>=0.9)
num1<-num[,-(42:95)]
num1<-num1[,-(2:7)]
sum(cor(num1)>=0.9)


#create dummy variables for some categorical variables
country_USdummy<-ifelse(cat$db_country=="US",1,0)
employeerange_Commercialdummy<-ifelse(cat$db_employeerange=="Commercial",1,0)
employeerange_Enterprisedummy<-ifelse(cat$db_employeerange=="Enterprise",1,0)
employeerange_Nonedummy<-ifelse(cat$db_employeerange=="None",1,0)
employeerange_Smalldummy<-ifelse(cat$db_employeerange=="Small",1,0)
employeerange_VerySmalldummy<-ifelse(cat$db_employeerange=="Very Small",1,0)
audience_Residentialdummy<-ifelse(cat$db_audience=="Residential",1,0)
audience_SOHOdummy<-ifelse(cat$db_audience=="SOHO",1,0)
audience_WMNdummy<-ifelse(cat$db_audience=="Wireless->Mobile Network",1,0)
isocountry_USdummy<-ifelse(cat$iso_country_dunssite=="UNITED STATES",1,0)


#combine the numerical variables, dummy variables created for categorical variables, and dependent variables
Fulldata<-cbind(num1, country_USdummy, employeerange_Commercialdummy, employeerange_Enterprisedummy,employeerange_Nonedummy,employeerange_Smalldummy,
                employeerange_VerySmalldummy,audience_Residentialdummy,audience_SOHOdummy,audience_WMNdummy,isocountry_USdummy,total$tgt_hol, 
                total$tgt_download, tgt_group)
                
#Split the dataset to Train/Test (90%) and Valication (10%)
TrainTest <- Fulldata[1:90011,]
Validation <- Fulldata[90010:100012,]

#Manually split TrainTest dataset to do perform cross-validation

########################################################################################
###################################Cross-Validation 1###################################
########################################################################################
set.seed(123)   
sample = sample.split(TrainTest,SplitRatio = 0.75) # splits the data in the ratio mentioned in SplitRatio. After splitting marks these rows as logical TRUE and the the remaining are marked as logical FALSE
train1 =subset(TrainTest,sample ==TRUE) # creates a training dataset named train1 with rows which are marked as TRUE
test1=subset(TrainTest, sample==FALSE)

drop <- c("total$tgt_hol","total$tgt_download","tgt_group")
train1forpca = train1[,!(names(train1) %in% drop)]
test1forpca = test1[,!(names(test1) %in% drop)]

pca1 <- prcomp(train1forpca,
              center = TRUE,
              scale. = TRUE) 
# which PC to retain
plot(pca1, type = "l")
summary(pca1)# chose 28 PCs to retain at least 90% of the variance

#created a matrix with 28 PCs for each of the observation for train and test dataset. 
PCAMatrix1<-as.matrix(pca1$rotation)
PCARetained1<-PCAMatrix1[,1:28]
train1afterPCA<-as.matrix(train1forpca)
test1afterPCA<-as.matrix(test1forpca)
train1afterPCA_keep<-train1afterPCA%*%PCARetained1
train1afterPCA_keep<-as.data.frame(train1afterPCA_keep)
test1afterPCA_keep<-test1afterPCA%*%PCARetained1
test1afterPCA_keep<-as.data.frame(test1afterPCA_keep)

######################################################
#####Run the model for target action - Drive HoL#####
train1afterPCA<-cbind(train1afterPCA_keep,train1$`total$tgt_hol`)
test1afterPCA<-cbind(test1afterPCA_keep,test1$`total$tgt_hol`)

#start the simple logistic model
colnames(train1afterPCA)[29] <- "targetHoL"
colnames(test1afterPCA)[29] <- "targetHoL"
model_logistic_HoL1<-glm(targetHoL ~ ., data=train1afterPCA, family="binomial"(link="logit")) 
model_logistic_HoL1 

#Try to use stepwise regression
model_logistic_stepwiseAIC_HoL1<-stepAIC(model_logistic_HoL1,direction = c("both"),trace = 1) 
summary(model_logistic_stepwiseAIC_HoL1) 

###Finding predicitons: probabilities and classification
logistic_probabilities_HoL1<-predict(model_logistic_stepwiseAIC_HoL1,newdata=test1afterPCA,type="response") #Predict probabilities

####ROC Curve
logistic_ROC_prediction_HoL1 <- prediction(logistic_probabilities_HoL1, test1afterPCA$targetHoL)
logistic_ROC_HoL1 <- performance(logistic_ROC_prediction_HoL1,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC_HoL1, main ="ROC Curve: Target Action_Drive HoL") #Plot ROC curve

####AUC (area under curve)
auc.tmp_HoL1 <- performance(logistic_ROC_prediction_HoL1,"auc") #Create AUC data
logistic_auc_testing_HoL1 <- as.numeric(auc.tmp_HoL1@y.values) #Calculate AUC
logistic_auc_testing_HoL1 #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

######################################################
#####Run the model for target action - Download#####
train1afterPCA<-cbind(train1afterPCA_keep,train1$`total$tgt_download`)
test1afterPCA<-cbind(test1afterPCA_keep,test1$`total$tgt_download`)

#start the simple logistic model
colnames(train1afterPCA)[29] <- "targetDownload"
colnames(test1afterPCA)[29] <- "targetDownload"
model_logistic_download1<-glm(targetDownload ~ ., data=train1afterPCA, family="binomial"(link="logit")) 
model_logistic_download1

#Try to use stepwise regression
model_logistic_stepwiseAIC_download1<-stepAIC(model_logistic_download1,direction = c("both"),trace = 1) 
summary(model_logistic_stepwiseAIC_download1) 

###Finding predicitons: probabilities and classification
logistic_probabilities_download1<-predict(model_logistic_stepwiseAIC_download1,newdata=test1afterPCA,type="response") #Predict probabilities

####ROC Curve
logistic_ROC_prediction_download1 <- prediction(logistic_probabilities_download1, test1afterPCA$targetDownload)
logistic_ROC_download1 <- performance(logistic_ROC_prediction_download1,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC_download1, main ="ROC Curve: Target Action_Download") #Plot ROC curve

####AUC (area under curve)
auc.tmp_download1 <- performance(logistic_ROC_prediction_download1,"auc") #Create AUC data
logistic_auc_testing_download1 <- as.numeric(auc.tmp_download1@y.values) #Calculate AUC
logistic_auc_testing_download1 


######################################################
#####Run the model for target action - Other (webinar, seminar, evaluation, and whitepaper)#####
train1afterPCA<-cbind(train1afterPCA_keep,train1$tgt_group)
test1afterPCA<-cbind(test1afterPCA_keep,test1$tgt_group)

#start the simple logistic model
colnames(train1afterPCA)[29] <- "targetOther"
colnames(test1afterPCA)[29] <- "targetOther"
model_logistic_Other1<-glm(targetOther ~ ., data=train1afterPCA, family="binomial"(link="logit")) 
model_logistic_Other1

#Try to use stepwise regression
model_logistic_stepwiseAIC_Other1<-stepAIC(model_logistic_Other1,direction = c("both"),trace = 1) 
summary(model_logistic_stepwiseAIC_Other1) 

###Finding predicitons: probabilities and classification
logistic_probabilities_Other1<-predict(model_logistic_stepwiseAIC_Other1,newdata=test1afterPCA,type="response") #Predict probabilities

####ROC Curve
logistic_ROC_prediction_Other1 <- prediction(logistic_probabilities_Other1, test1afterPCA$targetOther)
logistic_ROC_Other1 <- performance(logistic_ROC_prediction_Other1,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC_Other1, main ="ROC Curve: Target Action_Others") #Plot ROC curve

####AUC (area under curve)
auc.tmp_Other1 <- performance(logistic_ROC_prediction_Other1,"auc") #Create AUC data
logistic_auc_testing_Other1 <- as.numeric(auc.tmp_Other1@y.values) #Calculate AUC
logistic_auc_testing_Other1 

###Compare the results for predictions on four target actions using ROC curve
par(mfrow=c(1,3))
plot(logistic_ROC_HoL1, main ="ROC Curve: Target Action_Drive HoL") 
plot(logistic_ROC_download1, main ="ROC Curve: Target Action_Download")
plot(logistic_ROC_Other1, main ="ROC Curve: Target Action_Others")
par(mfrow=c(1,1))

###Compare the results for predictions on four target actions using AUC
logistic_auc_testing_HoL1  # 0.9860551
logistic_auc_testing_download1  # 0.9888771
logistic_auc_testing_Other1  # 0.9830715


########################################################################################
###################################Cross-Validation 2###################################
########################################################################################
set.seed(125)   
sample <- sample.split(TrainTest,SplitRatio = 0.75) 
train2 <- subset(TrainTest,sample ==TRUE) 
test2 <-subset(TrainTest, sample==FALSE)

drop <- c("total$tgt_hol","total$tgt_download","tgt_group")
train2forpca = train2[,!(names(train2) %in% drop)]
test2forpca = test2[,!(names(test2) %in% drop)]

pca2 <- prcomp(train2forpca,
               center = TRUE,
               scale. = TRUE) 
# which PC to retain
plot(pca2, type = "l")
summary(pca2)# chose 28 PCs to retain at least 90% of the variance

#created a matrix with 28 PCs for each of the observation for train and test dataset. 
PCAMatrix2<-as.matrix(pca2$rotation)
PCARetained2<-PCAMatrix2[,1:28] ####CHANGE
train2afterPCA<-as.matrix(train2forpca)
test2afterPCA<-as.matrix(test2forpca)
train2afterPCA_keep<-train2afterPCA%*%PCARetained2
train2afterPCA_keep<-as.data.frame(train2afterPCA_keep)
test2afterPCA_keep<-test2afterPCA%*%PCARetained2
test2afterPCA_keep<-as.data.frame(test2afterPCA_keep)

######################################################
#####Run the model for target action - Drive HoL#####
train2afterPCA<-cbind(train2afterPCA_keep,train2$`total$tgt_hol`)
test2afterPCA<-cbind(test2afterPCA_keep,test2$`total$tgt_hol`)

#start the simple logistic model
colnames(train2afterPCA)[29] <- "targetHoL"
colnames(test2afterPCA)[29] <- "targetHoL"
model_logistic_HoL2<-glm(targetHoL ~ ., data=train2afterPCA, family="binomial"(link="logit")) 
model_logistic_HoL2 

#Try to use stepwise regression
model_logistic_stepwiseAIC_HoL2<-stepAIC(model_logistic_HoL2,direction = c("both"),trace = 1) 
summary(model_logistic_stepwiseAIC_HoL2) 

###Finding predicitons: probabilities and classification
logistic_probabilities_HoL2<-predict(model_logistic_stepwiseAIC_HoL2,newdata=test2afterPCA,type="response") #Predict probabilities

####ROC Curve
logistic_ROC_prediction_HoL2 <- prediction(logistic_probabilities_HoL2, test2afterPCA$targetHoL)
logistic_ROC_HoL2 <- performance(logistic_ROC_prediction_HoL2,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC_HoL2, main ="ROC Curve: Target Action_Drive HoL") #Plot ROC curve

####AUC (area under curve)
auc.tmp_HoL2 <- performance(logistic_ROC_prediction_HoL2,"auc") #Create AUC data
logistic_auc_testing_HoL2 <- as.numeric(auc.tmp_HoL2@y.values) #Calculate AUC
logistic_auc_testing_HoL2 

######################################################
#####Run the model for target action - Download#####
train2afterPCA<-cbind(train2afterPCA_keep,train2$`total$tgt_download`)
test2afterPCA<-cbind(test2afterPCA_keep,test2$`total$tgt_download`)

#start the simple logistic model
colnames(train2afterPCA)[29] <- "targetDownload"
colnames(test2afterPCA)[29] <- "targetDownload"
model_logistic_download2<-glm(targetDownload ~ ., data=train2afterPCA, family="binomial"(link="logit")) 
model_logistic_download2

#Try to use stepwise regression
model_logistic_stepwiseAIC_download2<-stepAIC(model_logistic_download2,direction = c("both"),trace = 1) 
summary(model_logistic_stepwiseAIC_download2) 

###Finding predicitons: probabilities and classification
logistic_probabilities_download2<-predict(model_logistic_stepwiseAIC_download2,newdata=test2afterPCA,type="response") #Predict probabilities

####ROC Curve
logistic_ROC_prediction_download2 <- prediction(logistic_probabilities_download2, test2afterPCA$targetDownload)
logistic_ROC_download2 <- performance(logistic_ROC_prediction_download2,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC_download2, main ="ROC Curve: Target Action_Download") #Plot ROC curve

####AUC (area under curve)
auc.tmp_download2 <- performance(logistic_ROC_prediction_download2,"auc") #Create AUC data
logistic_auc_testing_download2 <- as.numeric(auc.tmp_download2@y.values) #Calculate AUC
logistic_auc_testing_download2 


######################################################
#####Run the model for target action - Other (webinar, seminar, evaluation, and whitepaper)#####
train2afterPCA<-cbind(train2afterPCA_keep,train2$tgt_group)
test2afterPCA<-cbind(test2afterPCA_keep,test2$tgt_group)

#start the simple logistic model
colnames(train2afterPCA)[29] <- "targetOther"
colnames(test2afterPCA)[29] <- "targetOther"
model_logistic_Other2<-glm(targetOther ~ ., data=train2afterPCA, family="binomial"(link="logit")) 
model_logistic_Other2

#Try to use stepwise regression
model_logistic_stepwiseAIC_Other2<-stepAIC(model_logistic_Other2,direction = c("both"),trace = 1) 
summary(model_logistic_stepwiseAIC_Other2) 

###Finding predicitons: probabilities and classification
logistic_probabilities_Other2<-predict(model_logistic_stepwiseAIC_Other2,newdata=test2afterPCA,type="response") #Predict probabilities

####ROC Curve
logistic_ROC_prediction_Other2 <- prediction(logistic_probabilities_Other2, test2afterPCA$targetOther)
logistic_ROC_Other2 <- performance(logistic_ROC_prediction_Other2,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC_Other2, main ="ROC Curve: Target Action_Others") #Plot ROC curve

####AUC (area under curve)
auc.tmp_Other2 <- performance(logistic_ROC_prediction_Other2,"auc") #Create AUC data
logistic_auc_testing_Other2 <- as.numeric(auc.tmp_Other2@y.values) #Calculate AUC
logistic_auc_testing_Other2 


###Compare the results for predictions on four target actions using ROC curve
par(mfrow=c(1,3))
plot(logistic_ROC_HoL2, main ="ROC Curve: Target Action_Drive HoL") 
plot(logistic_ROC_download2, main ="ROC Curve: Target Action_Download")
plot(logistic_ROC_Other2, main ="ROC Curve: Target Action_Others")
par(mfrow=c(1,1))

###Compare the results for predictions on four target actions using AUC
logistic_auc_testing_HoL2  # 0.9853075
logistic_auc_testing_download2  # 0.9893886
logistic_auc_testing_Other2  # 0.9764798


########################################################################################
###################################Cross-Validation 3###################################
########################################################################################
set.seed(127)   
sample <- sample.split(TrainTest,SplitRatio = 0.75) 
train3 <- subset(TrainTest,sample ==TRUE) 
test3 <-subset(TrainTest, sample==FALSE)

drop <- c("total$tgt_hol","total$tgt_download","tgt_group")
train3forpca = train3[,!(names(train3) %in% drop)]
test3forpca = test3[,!(names(test3) %in% drop)]

pca3 <- prcomp(train3forpca,
               center = TRUE,
               scale. = TRUE) 
# which PC to retain
plot(pca3, type = "l")
summary(pca3)# chose 28 PCs to retain at least 90% of the variance

#created a matrix with 28 PCs for each of the observation for train and test dataset. 
PCAMatrix3<-as.matrix(pca3$rotation)
PCARetained3<-PCAMatrix3[,1:28] ####CHANGE
train3afterPCA<-as.matrix(train3forpca)
test3afterPCA<-as.matrix(test3forpca)
train3afterPCA_keep<-train3afterPCA%*%PCARetained3
train3afterPCA_keep<-as.data.frame(train3afterPCA_keep)
test3afterPCA_keep<-test3afterPCA%*%PCARetained3
test3afterPCA_keep<-as.data.frame(test3afterPCA_keep)

######################################################
#####Run the model for target action - Drive HoL#####
train3afterPCA<-cbind(train3afterPCA_keep,train3$`total$tgt_hol`)
test3afterPCA<-cbind(test3afterPCA_keep,test3$`total$tgt_hol`)

#start the simple logistic model
colnames(train3afterPCA)[29] <- "targetHoL"
colnames(test3afterPCA)[29] <- "targetHoL"
model_logistic_HoL3<-glm(targetHoL ~ ., data=train3afterPCA, family="binomial"(link="logit")) 
model_logistic_HoL3 

#Try to use stepwise regression
model_logistic_stepwiseAIC_HoL3<-stepAIC(model_logistic_HoL3,direction = c("both"),trace = 1) 
summary(model_logistic_stepwiseAIC_HoL3) 

###Finding predicitons: probabilities and classification
logistic_probabilities_HoL3<-predict(model_logistic_stepwiseAIC_HoL3,newdata=test3afterPCA,type="response") #Predict probabilities

####ROC Curve
logistic_ROC_prediction_HoL3 <- prediction(logistic_probabilities_HoL3, test3afterPCA$targetHoL)
logistic_ROC_HoL3 <- performance(logistic_ROC_prediction_HoL3,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC_HoL3, main ="ROC Curve: Target Action_Drive HoL") #Plot ROC curve

####AUC (area under curve)
auc.tmp_HoL3 <- performance(logistic_ROC_prediction_HoL3,"auc") #Create AUC data
logistic_auc_testing_HoL3 <- as.numeric(auc.tmp_HoL3@y.values) #Calculate AUC
logistic_auc_testing_HoL3 

######################################################
#####Run the model for target action - Download#####
train3afterPCA<-cbind(train3afterPCA_keep,train3$`total$tgt_download`)
test3afterPCA<-cbind(test3afterPCA_keep,test3$`total$tgt_download`)

#start the simple logistic model
colnames(train3afterPCA)[29] <- "targetDownload"
colnames(test3afterPCA)[29] <- "targetDownload"
model_logistic_download3<-glm(targetDownload ~ ., data=train3afterPCA, family="binomial"(link="logit")) 
model_logistic_download3

#Try to use stepwise regression
model_logistic_stepwiseAIC_download3<-stepAIC(model_logistic_download3,direction = c("both"),trace = 1) 
summary(model_logistic_stepwiseAIC_download3) 

###Finding predicitons: probabilities and classification
logistic_probabilities_download3<-predict(model_logistic_stepwiseAIC_download3,newdata=test3afterPCA,type="response") #Predict probabilities

####ROC Curve
logistic_ROC_prediction_download3 <- prediction(logistic_probabilities_download3, test3afterPCA$targetDownload)
logistic_ROC_download3 <- performance(logistic_ROC_prediction_download3,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC_download3, main ="ROC Curve: Target Action_Download") #Plot ROC curve

####AUC (area under curve)
auc.tmp_download3 <- performance(logistic_ROC_prediction_download3,"auc") #Create AUC data
logistic_auc_testing_download3 <- as.numeric(auc.tmp_download3@y.values) #Calculate AUC
logistic_auc_testing_download3 


######################################################
#####Run the model for target action - Other (webinar, seminar, evaluation, and whitepaper)#####
train3afterPCA<-cbind(train3afterPCA_keep,train3$tgt_group)
test3afterPCA<-cbind(test3afterPCA_keep,test3$tgt_group)

#start the simple logistic model
colnames(train3afterPCA)[29] <- "targetOther"
colnames(test3afterPCA)[29] <- "targetOther"
model_logistic_Other3<-glm(targetOther ~ ., data=train3afterPCA, family="binomial"(link="logit")) 
model_logistic_Other3

#Try to use stepwise regression
model_logistic_stepwiseAIC_Other3<-stepAIC(model_logistic_Other3,direction = c("both"),trace = 1) 
summary(model_logistic_stepwiseAIC_Other3) 

###Finding predicitons: probabilities and classification
logistic_probabilities_Other3<-predict(model_logistic_stepwiseAIC_Other3,newdata=test3afterPCA,type="response") #Predict probabilities

####ROC Curve
logistic_ROC_prediction_Other3 <- prediction(logistic_probabilities_Other3, test3afterPCA$targetOther)
logistic_ROC_Other3 <- performance(logistic_ROC_prediction_Other3,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC_Other3, main ="ROC Curve: Target Action_Others") #Plot ROC curve

####AUC (area under curve)
auc.tmp_Other3 <- performance(logistic_ROC_prediction_Other3,"auc") #Create AUC data
logistic_auc_testing_Other3 <- as.numeric(auc.tmp_Other3@y.values) #Calculate AUC
logistic_auc_testing_Other3 


###Compare the results for predictions on four target actions using ROC curve
par(mfrow=c(1,3))
plot(logistic_ROC_HoL3, main ="ROC Curve: Target Action_Drive HoL") 
plot(logistic_ROC_download3, main ="ROC Curve: Target Action_Download")
plot(logistic_ROC_Other3, main ="ROC Curve: Target Action_Others")
par(mfrow=c(1,1))

###Compare the results for predictions on four target actions using AUC
logistic_auc_testing_HoL3  # 0.9852378
logistic_auc_testing_download3  # 0.9887834
logistic_auc_testing_Other3  # 0.9562005



#compare models
sum(logistic_auc_testing_HoL1+logistic_auc_testing_download1+logistic_auc_testing_Other1)/3 #0.9860013
sum(logistic_auc_testing_HoL2+logistic_auc_testing_download2+logistic_auc_testing_Other2)/3 #0.9837253
sum(logistic_auc_testing_HoL3+logistic_auc_testing_download3+logistic_auc_testing_Other3)/3 #0.9767406
#Based on average AUC, the following models preforms better

summary(pca2)
summary(model_logistic_stepwiseAIC_HoL2) 
summary(model_logistic_stepwiseAIC_download2) 
summary(model_logistic_stepwiseAIC_Other2)

par(mfrow=c(1,3))
plot(logistic_ROC_HoL2, main ="ROC Curve: Target Action_Drive HoL") 
plot(logistic_ROC_download2, main ="ROC Curve: Target Action_Download")
plot(logistic_ROC_Other2, main ="ROC Curve: Target Action_Others")
par(mfrow=c(1,1))


############################################################
#use Validation dataset for testing
drop <- c("total$tgt_hol","total$tgt_download","tgt_group")
Validationforpca = Validation[,!(names(Validation) %in% drop)]

PCAMatrix2<-as.matrix(pca2$rotation)
PCARetained2<-PCAMatrix2[,1:28] 
ValidationafterPCA<-as.matrix(Validationforpca)
ValidationafterPCA_keep<-ValidationafterPCA%*%PCARetained2
ValidationafterPCA_keep<-as.data.frame(ValidationafterPCA_keep)

######################################################
#####Run the model for target action - Drive HoL#####
ValidationafterPCA<-cbind(ValidationafterPCA_keep,Validation$`total$tgt_hol`)

#start the simple logistic model
colnames(ValidationafterPCA)[29] <- "targetHoL"

###Finding predicitons: probabilities and classification
logistic_probabilities_HoLValidation<-predict(model_logistic_stepwiseAIC_HoL2,newdata=ValidationafterPCA,type="response") #Predict probabilities

####ROC Curve
logistic_ROC_prediction_HoLValidation <- prediction(logistic_probabilities_HoLValidation, ValidationafterPCA$targetHoL)
logistic_ROC_HoLValidation <- performance(logistic_ROC_prediction_HoLValidation,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC_HoLValidation, main ="ROC Curve: Target Action_Drive HoL") #Plot ROC curve

####AUC (area under curve)
auc.tmp_HoLValidation <- performance(logistic_ROC_prediction_HoLValidation,"auc") #Create AUC data
logistic_auc_testing_HoLValidation <- as.numeric(auc.tmp_HoLValidation@y.values) #Calculate AUC
logistic_auc_testing_HoLValidation 

######################################################
#####Run the model for target action - Download#####
ValidationafterPCA<-cbind(ValidationafterPCA_keep,Validation$`total$tgt_download`)

#start the simple logistic model
colnames(ValidationafterPCA)[29] <- "targetDownload"

summary(model_logistic_stepwiseAIC_download2) 

###Finding predicitons: probabilities and classification
logistic_probabilities_downloadValidation<-predict(model_logistic_stepwiseAIC_download2,newdata=ValidationafterPCA,type="response") #Predict probabilities

####ROC Curve
logistic_ROC_prediction_downloadValidation <- prediction(logistic_probabilities_downloadValidation, ValidationafterPCA$targetDownload)
logistic_ROC_downloadValidation <- performance(logistic_ROC_prediction_downloadValidation,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC_downloadValidation, main ="ROC Curve: Target Action_Download") #Plot ROC curve

####AUC (area under curve)
auc.tmp_downloadValidation <- performance(logistic_ROC_prediction_downloadValidation,"auc") #Create AUC data
logistic_auc_testing_downloadValidation <- as.numeric(auc.tmp_downloadValidation@y.values) #Calculate AUC
logistic_auc_testing_downloadValidation 


######################################################
#####Run the model for target action - Other (webinar, seminar, evaluation, and whitepaper)#####
ValidationafterPCA<-cbind(ValidationafterPCA_keep,Validation$tgt_group)

#start the simple logistic model
colnames(ValidationafterPCA)[29] <- "targetOther"

#Try to use stepwise regression
summary(model_logistic_stepwiseAIC_Other2) 

###Finding predicitons: probabilities and classification
logistic_probabilities_OtherValidation<-predict(model_logistic_stepwiseAIC_Other2,newdata=ValidationafterPCA,type="response") #Predict probabilities

####ROC Curve
logistic_ROC_prediction_OtherValidation <- prediction(logistic_probabilities_OtherValidation, ValidationafterPCA$targetOther)
logistic_ROC_OtherValidation <- performance(logistic_ROC_prediction_OtherValidation,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC_OtherValidation, main ="ROC Curve: Target Action_Others") #Plot ROC curve

####AUC (area under curve)
auc.tmp_OtherValidation <- performance(logistic_ROC_prediction_OtherValidation,"auc") #Create AUC data
logistic_auc_testing_OtherValidation <- as.numeric(auc.tmp_OtherValidation@y.values) #Calculate AUC
logistic_auc_testing_OtherValidation 


###Compare the results for predictions on four target actions using ROC curve
par(mfrow=c(1,3))
plot(logistic_ROC_HoLValidation, main ="ROC Curve: Target Action_Drive HoL") 
plot(logistic_ROC_downloadValidation, main ="ROC Curve: Target Action_Download")
plot(logistic_ROC_OtherValidation, main ="ROC Curve: Target Action_Others")
par(mfrow=c(1,1))

###Compare the results for predictions on four target actions using AUC
logistic_auc_testing_HoLValidation  #0.9827796
logistic_auc_testing_downloadValidation  # 0.988237
logistic_auc_testing_OtherValidation  # 0.9739053

