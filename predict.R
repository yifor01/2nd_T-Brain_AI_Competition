library(tidyverse);library(glmnet);library(ncvreg);library(Ohit);library(xgboost);library(lubridate)
library(tseries);library(randomForest);library(e1071);library(MASS);library(caret);library(DrugClust)
library(beepr);library(iterators);library(parallel);library(glmnet);library(Metrics)

setwd("C:\\Users\\yifor\\Desktop\\台灣ETF價格預測競賽")
dat1 <- read.csv("tetfp.csv");dat2 <- read.csv("tsharep.csv")
#-------------------------------------------------------------------------------------------
#資料處理
head(dat1)
colnames(dat1)=colnames(dat2) =c("id","date","name","op","maxp","minp","cp","trading")
dat1$trading = as.numeric( gsub(",","",dat1$trading));dat2$trading = as.numeric( gsub(",","",dat2$trading))
dat2$op = as.numeric(as.character(dat2$op));dat2$cp = as.numeric(as.character( dat2$cp))
dat2$maxp = as.numeric(as.character( dat2$maxp));dat2$minp = as.numeric(as.character( dat2$minp))
dat1$date =  ymd(dat1$date);dat2$date =  ymd(dat2$date)
dat1 %>%  mutate(day = as.POSIXlt(dat1$date)$wday)

newdata =  function(data){
  data = data[!duplicated(data),]
  w1 = data[,c(1,2,7)] %>% group_by(date) %>% spread(id,cp)
  w2 = data[,c(1,2,4)] %>% group_by(date) %>% spread(id,op)
  w3 = data[,c(1,2,5)] %>% group_by(date) %>% spread(id,maxp)
  w4 = data[,c(1,2,6)] %>% group_by(date) %>% spread(id,minp)
  w5 = data[,c(1,2,8)] %>% group_by(date) %>% spread(id,trading)
  w6 = left_join(w1,w2,by="date")
  w7 = left_join(w6,w3,by="date")
  w8 = left_join(w7,w4,by="date")
  w9 = left_join(w8,w5,by="date")
  return(w9)
}
a1 = newdata(dat1)
a2 = newdata(dat2)
a1 = a1 %>%  mutate(day = as.POSIXlt(date)$wday)
ab = left_join(a1,a2,by="date")

#--------------------------------------------------------------------------------------------
# 建立預測函數  id = 基金序號 , method =  1(OGA+HDIC)  2(XGboost) 3(O+H+T) 4 SVM 5 RF
pred =  function(a,id,method){
  n = nrow(a)
  y = na.omit( diff( as.matrix(a[,(id+1)]) ) )
  ny = length( y )
  nn = min(ny,n-5)
  x = as.matrix( a[ (n-nn-4):n ,-c(1,(id+1)) ] )
  y = y[(ny-nn+1):ny]
  n = nrow(x)
  xt = scale(x)*sqrt(nrow(x)/(nrow(x)-1))
  yt = y - mean(y)
  
  xtrain = xt[1:(n-5),];  ytrain = yt;  xtest = xt[(n-4):n,]
  score = function(pred,test){
    mean(as.numeric(sign( diff(pred) ) == sign( diff(test) ) ))- sum(abs(pred-test))*0.5
  }
  params = list(colsample_bytree = 0.8, subsample = 0.7,booster = "gbtree",max_depth = 3,
                eta = 0.03,eval_metric = "rmse",objective = "reg:linear",gamma = 0) 

  if(method==1){
    fit = Ohit(X = xtrain , y = ytrain ,HDIC_Type = "HDAIC" )
    num = predict_Ohit(fit, xtest  )$pred_HDIC + mean(y)
  }
  if(method==2){
    fit = xgboost(xtrain,label = ytrain,params,nrounds=71,verbose = 0)
    num = predict( fit , xtest ) + mean(y)
  }   
  if(method==3){
    fit = xgboost(xtrain,label = ytrain,params,nfolds=5,nrounds=71,verbose = 0)
    num = predict( fit , xtest ) + mean(y)
  } 
  if(method==4){
    xtt = xt[,complete.cases(t(xt))]
    xtrain = xtt[1:(n-5),];  ytrain = yt;  xtest = xtt[(n-4):n,]
    data = as.data.frame( cbind(ytrain,xtrain)  )
    fit = svm( ytrain~.,data  )
    num = predict(fit, xtest  ) + mean(y)
  }
  if(method==5){
    xtt = xt[,complete.cases(t(xt))]
    xtrain = xtt[1:(n-5),];  ytrain = yt;  xtest = xtt[(n-4):n,]
    data = as.data.frame( cbind(ytrain,xtrain)  )
    fit = randomForest( ytrain~.,data  ,ntree=300,mytry=400)
    num = predict(fit, xtest  ) + mean(y)
  }
  return(num)
}
#############################################
# 預測值(method 1-4)
k1 = t(sapply(1:18,function(i) pred(ab,i,1)))
k2 = t(sapply(1:18,function(i) pred(ab,i,2)))
k3 = t(sapply(1:18,function(i) pred(ab,i,3)))
k4 = t(sapply(1:18,function(i) pred(ab,i,4)))
#########################################################################
E1 = list()
for(i in 0:21){
  nq=nrow(ab)
  data = ab[1:(nq-5*i),]
  e1 = t(sapply(1:18,function(j) pred(data,j,1)))
  e2 = t(sapply(1:18,function(j) pred(data,j,2)))
  e3 = t(sapply(1:18,function(j) pred(data,j,3)))
  e4 = t(sapply(1:18,function(j) pred(data,j,4)))
  E1[[i+1]] = cbind(e1,e2,e3,e4)
  print(i)
}

bb=list()
for(i in 0:(length(E1)-1) ){
  data.star = ab[c( (nrow(ab)-5*i-5):(nrow(ab)-5*i) )  ,2:19]
  data.diff = matrix(NA,nrow=18,ncol=5)
  for(j in 1:18){
    data.diff[j,] = as.numeric(diff( as.matrix( data.star[,j] ))  )
  }
  bb[[i+1]] = data.diff
}

blr = function(id){
  true = as.numeric(sapply(1:length(E1),function(x) bb[[x]][id,]) )
  pred1 = c(as.numeric( sapply(1:length(E1),function(x) E1[[x]][id,1:5] )   ) ,k1[id,] )
  pred2 = c(as.numeric( sapply(1:length(E1),function(x) E1[[x]][id,6:10] )   ) ,k2[id,] )
  pred3 = c(as.numeric( sapply(1:length(E1),function(x) E1[[x]][id,11:15] )   ) ,k3[id,] )
  pred4 = c(as.numeric( sapply(1:length(E1),function(x) E1[[x]][id,16:20] )   ) ,k4[id,] )
  true = true - mean(true)
  xx =  scale( cbind(pred1,pred2,pred3,pred4) )
  n = nrow(xx)
  params = list(booster = "gbtree",max_depth = 3,
                eta = 0.03,eval_metric = "rmse",objective = "reg:linear",gamma = 0) 
  
  fit1 =  xgboost(data = xx[1:(n-5),] ,label = as.numeric(true),params,
                  nrounds = 15,verbose = 0)
  w1 =  predict(fit1,xx[1:(n-5),]   )
  ww1 =  predict(fit1,xx[(n-4):n, ]   )
  
  data1 = data.frame(cbind(true,xx[1:(n-5),]  )  )
  
  fit2 = svm(true~ . ,data = data1  )
  w2 = predict(fit2,  data1[1:(n-5),-1]  )
  ww2 = predict(fit2,data.frame( xx[(n-4):n,]  ))
  fit3 = glm(true~ . ,data = data1  )
  w3 = predict(fit3,  data1[1:(n-5),-1]  )
  ww3 = predict(fit3,data.frame( xx[(n-4):n,]  ))
  ab12 = cbind(w1,w2,w3 )
  ab13 = cbind(ww1,ww2,ww3)
  colnames(ab12)=  colnames(ab13)
  fit = xgboost(data = ab12 ,label = as.numeric(true),params,
                nrounds = 30,verbose = 0)
  ww = predict(fit, ab13  )
  return(ww + mean(true))
}
#                      
t(sapply(1:18, function(x) blr(x)) )
