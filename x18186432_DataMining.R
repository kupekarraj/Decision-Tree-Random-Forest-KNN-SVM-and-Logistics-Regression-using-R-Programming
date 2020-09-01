#Importing dataset
chess<-read.csv("C:\\Users\\kupekarraj\\Desktop\\NciReports&Assignments\\Datamining\\Chess_Dataset\\Chess_Dataset.csv",header = FALSE, na.strings = "")
#Giving Column names for the data frame
colnames(chess)<-c("bkblk","bknwy","bkon8","bkona","bkspr","bkxbq","bkxcr","bkxwp","blxwp","bxqsq","cntxt","dsopp","dwipd",
                   "hdchk","katri","mulch","qxmsq","r2ar8","reskd","reskr","rimmx","rkxwp","rxmsq","simpl","skach","skewr",
                   "skrxp","spcop","stlmt","thrsk","wkcti","wkna8","wknck","wkovl","wkpos","wtoeg","status")
#Structure of dataframe
str(chess)
summary(chess)
#Checking for any missing values in dataframe
sapply(chess, function(x)sum(is.na(x)))
#library(Amelia)
#missmap(chess,main = "Missing Data")
#Checking for any outliers in the dataframe cannot be performed because all the columns are in the factor format
#Giving our own reference for the class with maximum enteries 
chess$dwipd<-relevel(chess$dwipd,ref = "l")
chess$katri<-relevel(chess$katri,ref = "n")
chess$r2ar8<-relevel(chess$r2ar8,ref = "t")
chess$skewr<-relevel(chess$skewr,ref = "t")
chess$wkovl<-relevel(chess$wkovl,ref = "t")
chess$wkpos<-relevel(chess$wkpos,ref="t")
#Recoding the status variable
table(chess$status)
chess$status<-factor(chess$status,levels = c("won","nowin"),labels = c(0,1))
chess$status<-relevel(chess$status,ref = "0")
#Dividing the dataframe into training and testing dataset
library(caret)
Train<-createDataPartition(chess$status,p=0.7,list=FALSE)
training<-chess[Train,]
testing<-chess[-Train,]
#Implementing Logistics Regression 
lmodel<-glm(status~.,family = "binomial",data=training)
summary(lmodel)
#performing stepwise selection to remove the multicolinearity in model alog with insignificant variables
lmodel2<-step(glm(status~.,family = "binomial",data=training),direction="both")
summary(lmodel2)
#checking the presence of multicolinearity
library(car)
vif(lmodel2)
#Prediction 
testing$predict<-predict(lmodel2,testing,type="response")
testing$Predict<-as.factor(ifelse(testing$predict>0.70,"1","0"))
#Accuracy
library(e1071)
confusionMatrix(testing$Predict,testing$status)
#ROC Curve 
library(ROCR)
predictTrain = predict(lmodel2,testing, type="response")
#prediction function
ROCRpred = prediction(testing$predict, testing$status)
##performance function is to fetch 
ROCRperf = performance(ROCRpred, "tpr", "fpr")
#plot ROC curve
plot(ROCRperf)
pred = prediction(testing$predict,testing$status)
as.numeric(performance(pred,"auc")@y.values)
#Thus, around 99% of the area in under curve


#Decision Tree Classifier
library(tree)
decimod<-tree(status~.,data = training)
plot(decimod)
text(decimod,pretty=0)
decimod
tree.pred=predict(decimod,testing,type="class")
table(tree.pred,testing$status)
#Displaying the accuracy of model using confusion matrix
library(caret)
confusionMatrix(tree.pred,testing$status)
#Applying cross validation using the missclassified data points
cv.decimod<-cv.tree(decimod,FUN=prune.misclass)
cv.decimod
#ploting the line graph in order to determine the best parameter for which the miss classification is minimum
plot(cv.decimod$size,cv.decimod$dev,type = "b")
#Using 15 as the best parameter
prune.decimod<-prune.misclass(decimod,best = 15)
plot(prune.decimod)
text(prune.decimod,pretty=0)
prunePredict<-predict(prune.decimod,testing,type = "class")
#checking the accuracy of the model
confusionMatrix(prunePredict,testing$status)

#Random Forest Classification
library(randomForest)
ranmod<-randomForest(status~.,data = testing)
summary(ranmod)
#prediction
testing$ran<-predict(ranmod,testing,type = "class")
confusionMatrix(testing$ran,testing$status)
#Cross Validation 
trctrl<-trainControl(method = "repeatedcv",number = 10, repeats = 3, search="grid")
tunegrid<-expand.grid(.mtry=c(1:10))
rancv<-train(status~.,data = training,method="rf",tuneGrid=tunegrid,trControl=trctrl)
rancv
#ploting the cross validated random forest model
plot(rancv)
testing$rancv<-predict(rancv,testing)
confusionMatrix(testing$rancv,testing$status)

##2nd dataset
#Importing the Letter Recognization dataset
letter<-read.csv("C:\\Users\\kupekarraj\\Desktop\\NciReports&Assignments\\Datamining\\LetterRecognition\\letter.csv",header = FALSE,na.strings = "",stringsAsFactors = FALSE)
#Giving the Colnames for dataframe
colnames(letter)<-c("letter","xbox","ybox","width","high","onpix","xbar","ybar",
                    "x2bar","y2bar","xybar","x2ybr","xy2bar","xege","xegvy","yege","yegvx")
summary(letter)
str(letter)
#Checking for any missing values
sapply(letter,function(x)sum(is.na(letter)))

#NOrmalizing the Numeric attributes 
normalize<-function(x){
  return((x-min(x))/(max(x)-min(x)))
}
#In our case only the first column is a non-numeric column.So eliminating the first column and normalizing the rest
letter.n<-as.data.frame(lapply(letter[,2:17],normalize))

#data partition 
Train<-sample(1:nrow(letter.n),size=nrow(letter.n)*0.7,replace = FALSE)
training<-letter.n[Train,]
testing<-letter.n[-Train,]

#Getting the train and test labels inorder to evaluate the model accuracy
train_labels<-letter[Train,1]
test_labels<-letter[-Train,1]

#Importing the KNN library 'class'
library(class)
knnmodel<-knn(train = training,test = testing,cl=train_labels,k=118)
#Here we selected the k value as the square root of the total number of observations in the dataset.
#In our case it approximately around 118
summary(knnmodel)
#Checking the accuracy using confusion matrix
confusionMatrix(table(knnmodel,test_labels))

#finding the optimal value for k 
i=1
k.optm=1
for (i in 1:120) {
  knnmodel<-knn(train = training,test = testing,cl=train_labels,k=i)
  k.optm[i]<-100*sum(test_labels==knnmodel)/NROW(test_labels)
  k=i
  cat(k,"=",k.optm[i],"\n")
}
#Thus from the output it is clear that the k value with 1 gives us the maximum accuracy.
#Final model
knnmod<-knn(train = training,test = testing,cl=train_labels,k=1)
summary(knnmod)
library(caret)
confusionMatrix(table(test_labels,knnmod))
#Additional Stuff
install.packages("gmodels")
#Building a cross table to get an overview of the probability of correct and incorrect classification
library(gmodels)
CrossTable(x=test_labels,y=knnmod)

#Support vector Machine Classification model
#As the letter varaible is having a char data type, so we need to convert the data type to factor data type
letter$letter<-as.factor(letter$letter)
#Data Slicing 
library(caret)
Train1<-createDataPartition(letter$letter,p=0.7,list = FALSE)
train1<-letter[Train1,]
test1<-letter[-Train1,]
#Implementing the SVM model
library(e1071)
svmmodel<-svm(letter~.,data=train1)
summary(svmmodel)
test1$result<-predict(svmmodel,test1)
table(test1$result,test1$letter)
confusionMatrix(test1$letter,test1$result)

#tuning the svm and setting the cost parameter to 10 
trcrtl<-trainControl(method = "repeatedcv",number = 10,repeats = 3)
svm_linear<-svm(letter~.,data = train1,trControl=trcrtl,cost=10)
summary(svm_linear)
test1$result1<-predict(svm_linear,test1)
table(test1$result1,test1$letter)
confusionMatrix(test1$letter,test1$result1)
#Thus we can see that by altering the cost parameter the accuracy of the model is increased by 2% 

##third dataset
#Importing the Tesco Marketing data
tesco<-read.csv("C:\\Users\\kupekarraj\\Desktop\\Tesco_train.csv")
str(tesco)
#Converting the data type to factor
tesco$content_1<-as.factor(tesco$content_1)
tesco$content_2<-as.factor(tesco$content_2)
tesco$content_3<-as.factor(tesco$content_3)
tesco$content_4<-as.factor(tesco$content_4)
tesco$content_5<-as.factor(tesco$content_5)
tesco$content_6<-as.factor(tesco$content_6)
tesco$content_7<-as.factor(tesco$content_7)
tesco$content_8<-as.factor(tesco$content_8)
tesco$content_9<-as.factor(tesco$content_9)

#imputing the 'NA' level to '1'
sapply(tesco, function(x)sum(is.na(x)))
tesco$content_1[is.na(tesco$content_1)]<-1
tesco$content_2[is.na(tesco$content_2)]<-1
tesco$content_3[is.na(tesco$content_3)]<-1
tesco$content_4[is.na(tesco$content_4)]<-1
tesco$content_5[is.na(tesco$content_5)]<-1
tesco$content_6[is.na(tesco$content_6)]<-1
tesco$content_7[is.na(tesco$content_7)]<-1
tesco$content_8[is.na(tesco$content_8)]<-1
tesco$content_9[is.na(tesco$content_9)]<-1
#removing the county variable as it is irrelvant for the model
tesco<-tesco[-27]
#recoding the affluency variable to two levels 
library(car)
tesco$affluency<-recode(tesco$affluency,'c("High","Very High","Low","Very Low")="0";"Mid"="1"')
#Data partition
library(caret)
Train<-createDataPartition(tesco$content_1,p=0.7,list = FALSE)
training<-tesco[Train,]
testing<-tesco[-Train,]

#library(e1071)
#nbmodel<-naiveBayes(content_1~.,data = training)
#summary(nbmodel)
#testing$predict<-predict(nbmodel,testing,type = "class")
#confusionMatrix(testing$predict,testing$content_1)
#Logistics Regression
lmodel<-glm(content_1~.,data = testing,family = "binomial")
#prediction
testing$predict<-predict(lmodel,testing,type = "response")
testing$prediction<-as.factor(ifelse(testing$predict>0.58,"0","1"))
confusionMatrix(testing$prediction,testing$content_1)
#Evaluation using Roc Curve to measure the performance of classification at various threshold values.
library(ROCR)
#prediction function
ROCRpred = prediction(testing$predict, testing$content_1)
##performance function is to fetch 
ROCRperf = performance(ROCRpred, "tpr", "fpr")
#plot ROC curve
plot(ROCRperf)
pred = prediction(testing$predict,testing$content_1)
as.numeric(performance(pred,"auc")@y.values)
#Thus, around 60% predictions lies under the Roc Curve
