library(tidyverse);library(glmnet);library(ncvreg);library(Ohit);library(xgboost);library(lubridate)
library(tseries);library(randomForest);library(e1071);library(MASS);library(caret);library(DrugClust);library(rpart)
library(beepr);library(iterators);library(parallel);library(glmnet);library(Metrics)
setwd("C:\\Users\\yifor\\Desktop\\台灣ETF價格預測競賽")
dat1 <- read.csv("tetfp.csv");dat2 <- read.csv("tsharep.csv")
#testid =c(50,51,52,53,54,55,56,57,58,59,6201,6203,6204,6208,690,692,701,713 )
#-------------------------------------------------------------------------------------------
# 資料處理
head(dat1)
colnames(dat1)=colnames(dat2) =c("id","date","name","op","maxp","minp","cp","trading")
dat1$trading = as.numeric( gsub(",","",dat1$trading));dat2$trading = as.numeric( gsub(",","",dat2$trading))
dat2$op = as.numeric(as.character(dat2$op));dat2$cp = as.numeric(as.character( dat2$cp))
dat2$maxp = as.numeric(as.character( dat2$maxp));dat2$minp = as.numeric(as.character( dat2$minp))
dat1$date =  ymd(dat1$date);dat2$date =  ymd(dat2$date)


dat1 %>%  mutate(day = as.POSIXlt(dat1$date)$wday)

# 只適用dat1,dat2
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
# [Y] a1 = dat1(cp,....)  [X] a2=dat2(cp,.....)
a1 = newdata(dat1)
a2 = newdata(dat2)

a1 = a1 %>%  mutate(day = as.POSIXlt(date)$wday)

# 外部資料=================
# 台灣大盤 
#out1 = read.csv("Y9999(5_18).txt",sep="\t")
#out1 = out1[,c(3,4,5,6,7,8,9)]
#colnames(out1)=c("date","op","maxp","minp","cp","trading","tradingv")
#out1$date = ymd(out1$date)
# 美股
out2 = read.csv("DJI(6_1).csv");out2$Date = ymd(out2$Date)
out3 = read.csv("GSPC(6_1).csv");out3$Date = ymd(out3$Date)
out4 = read.csv("NDX(6_1).csv");out4$Date = ymd(out4$Date)
out5 = read.csv("SOX(6_1).csv");out5$Date = ymd(out5$Date)
colnames(out2) = colnames(out3) = colnames(out4) = colnames(out5) =
  c("date","op","maxp","minp","cp","adcp","trading")
#out6 = read.csv("日月光.csv");out6$date = ymd(as.factor(out6$date))
#out7 = read.csv("美金即期匯率(5_24).txt",sep="\t");
#out7 = out7[,c(3,4,5)];colnames(out7)=c("date","buy","sell")
#out7$date = ymd(as.factor(out7$date))
#out8 = read.csv("TLHDR(5_24).txt",sep = "\t");out8 = out8[,c(3,4,5,6)]
#colnames(out8) = c("date","cp","trading","tradingv");out8$date = ymd(out8$date)
#out9 = read.csv("TLHD(5_24).txt",sep = "\t");out9 = out9[,c(3,4,5,6)]
#colnames(out9) = c("date","cp","trading","tradingv");out9$date = ymd(out9$date)
#out10 = read.csv("M2300(5_24).txt",sep="\t");
#out10 = out10[,c(3,4,5,6,7,8,9)]
#colnames(out10)=c("date","op","maxp","minp","cp","trading","tradingv")
#out10$date = ymd(out10$date)
#out11 = read.csv("Y9997(5_24).txt",sep="\t");
#out11 = out11[,c(3,4)]
#colnames(out11)=c("date","cp")
#out11$date = ymd(out11$date)

# 整合外部資料
#out.1 = left_join(out1,out2,by="date");out.2 = left_join(out.1,out3,by="date")
#out.3 = left_join(out.2,out4,by="date");out.4 = left_join(out.3,out4,by="date")
#out.5 = left_join(out.4,out6[,c(3,4,7)],by="date")
#out.6 = left_join(out.5,out7,by="date")
#out.7 = left_join(out.6,out8,by="date")
#out.8 = left_join(out.7,out9,by="date")
#out.9 = left_join(out.8,out10,by="date")
#out.10 = left_join(out.9,out11,by="date")
# 建立X+Y大矩陣
#x1 = left_join(a2,out.10,by="date")
#aa = left_join(a1,x1,by="date")
#aa # 資料處理
###############################################
ab = left_join(a1,a2,by="date")
ab1 = left_join(ab,out2,by="date")
ab2 = left_join(ab1,out3,by="date")
ab3 = left_join(ab2,out4,by="date")
ab4 = left_join(ab3,out5,by="date")

#--------------------------------------------------------------------------------------------
# 建立預測函數  id = 基金序號 , method =  1(OGA+HDIC)  2(XGboost) 3(O+H+T) 4 SVM
pred =  function(a,id,method){
  n = nrow(a)
  y = na.omit( diff( as.matrix(a[,(id+1)]) ) )
  ny = length( y )
  nn = min(ny,n-5)
  x = as.matrix( a[ (n-nn-4):n ,-c(1,(id+1)) ] )
  y = y[(ny-nn+1):ny]
  n = nrow(x)
  # 標準化
  xt = scale(x)*sqrt(nrow(x)/(nrow(x)-1))
  yt = y - mean(y)
  set.seed(123)
  
  xtrain = xt[1:(n-5),];  ytrain = yt;  xtest = xt[(n-4):n,]
  score = function(pred,test){
    mean(as.numeric(sign( diff(pred) ) == sign( diff(test) ) ))- sum(abs(pred-test))*0.5
  }
  
  params = list(colsample_bytree = 0.8, subsample = 0.7,booster = "gbtree",max_depth = 3,
                eta = 0.03,eval_metric = "rmse",objective = "reg:linear",gamma = 0)      
  set.seed(123)
  xindex =  CreateFolds(xtrain,5)
  # CV
  xtrain1=xtrain[-c(which(xindex==0)),];    xtrain2=xtrain[-c(which(xindex==1)),]
  xtrain3=xtrain[-c(which(xindex==2)),];    xtrain4=xtrain[-c(which(xindex==3)),]
  xtrain5=xtrain[-c(which(xindex==4)),];    ytrain1=ytrain[-c(which(xindex==0))]
  ytrain2=ytrain[-c(which(xindex==1))];    ytrain3=ytrain[-c(which(xindex==2))]
  ytrain4=ytrain[-c(which(xindex==3))];    ytrain5=ytrain[-c(which(xindex==4))]
  # method 1 = OGA + HDIC  
  if(method==1){
    # cv
    fit1 = Ohit(X = xtrain1 , y = ytrain1  )
    fit2 = Ohit(X = xtrain2 , y = ytrain2  )
    fit3 = Ohit(X = xtrain3 , y = ytrain3  )
    fit4 = Ohit(X = xtrain4 , y = ytrain4  )
    fit5 = Ohit(X = xtrain5 , y = ytrain5  )
    
    num1 = predict_Ohit(fit1, xtrain[which(xindex==0),]  )$pred_HDIC 
    num2 = predict_Ohit(fit2, xtrain[which(xindex==1),]  )$pred_HDIC 
    num3 = predict_Ohit(fit3, xtrain[which(xindex==2),]  )$pred_HDIC 
    num4 = predict_Ohit(fit4, xtrain[which(xindex==3),]  )$pred_HDIC 
    num5 = predict_Ohit(fit5, xtrain[which(xindex==4),]  )$pred_HDIC 
    
    score1 = score(num1,ytrain[which(xindex==0)])
    score2 = score(num2,ytrain[which(xindex==1)])
    score3 = score(num3,ytrain[which(xindex==2)])
    score4 = score(num4,ytrain[which(xindex==3)])
    score5 = score(num5,ytrain[which(xindex==4)])
    
    yi = which.max(c(score1,score2,score3,score4,score5))
    xtraintt = xtrain[-c(which(xindex==(yi-1)) ),]
    ytraintt = ytrain[-c(which(xindex==(yi-1)) )]
    fit = Ohit(X = xtraintt , y = ytraintt  )
    num = predict_Ohit(fit, xtest  )$pred_HDIC + mean(y)
  }
  # method 2 XGboostCVS  
  if(method==2){
    fit1 = xgboost(xtrain1,label = ytrain1,params,nrounds=71,verbose = 0)
    fit2 = xgboost(xtrain2,label = ytrain2,params,nrounds=71,verbose = 0)
    fit3 = xgboost(xtrain3,label = ytrain3,params,nrounds=71,verbose = 0)
    fit4 = xgboost(xtrain4,label = ytrain4,params,nrounds=71,verbose = 0)
    fit5 = xgboost(xtrain5,label = ytrain5,params,nrounds=71,verbose = 0)
    num1 = predict( fit1 , xtrain[which(xindex==0),] )
    num2 = predict( fit2 , xtrain[which(xindex==1),] )
    num3 = predict( fit3 , xtrain[which(xindex==2),] )
    num4 = predict( fit4 , xtrain[which(xindex==3),] )
    num5 = predict( fit5 , xtrain[which(xindex==4),] )
    
    score1 = score(num1,ytrain[which(xindex==0)])
    score2 = score(num2,ytrain[which(xindex==1)])
    score3 = score(num3,ytrain[which(xindex==2)])
    score4 = score(num4,ytrain[which(xindex==3)])
    score5 = score(num5,ytrain[which(xindex==4)])
    
    yi = which.max(c(score1,score2,score3,score4,score5))
    xtraintt = xtrain[-c(which(xindex==(yi-1)) ),]
    ytraintt = ytrain[-c(which(xindex==(yi-1)) ) ]
    fit = xgboost( xtraintt , label = ytraintt  ,params,nrounds=71,verbose = 0)
    num = predict(fit, xtest  ) + mean(y)
    
  }   
  # method 3 XGboostCV  
  if(method==3){
    fit = xgboost(xtrain,label = ytrain,params,nfolds=5,nrounds=71,verbose = 0)
    num = predict( fit , xtest ) + mean(y)
  } 
  # method 4 SVM 
  if(method==4){
    # 變換成no na資料
    xtt = xt[,complete.cases(t(xt))]
    xtrain = xtt[1:(n-5),];  ytrain = yt;  xtest = xtt[(n-4):n,]
    index =  CreateFolds(xtrain,3)
    
    data = as.data.frame( cbind(ytrain,xtrain)  )
    fit1 = svm( ytrain~.,data[-c(which(index==0)),]  )
    fit2 = svm( ytrain~.,data[-c(which(index==1)),]  )
    fit3 = svm( ytrain~.,data[-c(which(index==2)),]  )
    
    score1 = score( predict(fit1,xtrain[which(index==0),] ) ,ytrain[which(index==0)])
    score2 = score( predict(fit2,xtrain[which(index==1),] ) ,ytrain[which(index==1)])
    score3 = score( predict(fit3,xtrain[which(index==2),] ) ,ytrain[which(index==2)])
    
    yi = which.max(c(score1,score2,score3))
    fit = svm( ytrain~.,data[-c(which(index==(yi-1))),]  )
    num = predict(fit, xtest  ) + mean(y)
  }
  # method 5 RF
  if(method==5){
    # 變換成no na資料
    xtt = xt[,complete.cases(t(xt))]
    xtrain = xtt[1:(n-5),];  ytrain = yt;  xtest = xtt[(n-4):n,]
    index =  CreateFolds(xtrain,3)
    # RF
    fit1 = randomForest(xtrain[-c(which(index==0)),],ytrain[-c(which(index==0))],ntree=200,mytry = 4676)
    fit2 = randomForest(xtrain[-c(which(index==1)),],ytrain[-c(which(index==1))],ntree=200,mytry = 4676)
    fit3 = randomForest(xtrain[-c(which(index==2)),],ytrain[-c(which(index==2))],ntree=200,mytry = 4676)
    
    score1 = score( predict(fit1,xtrain[which(index==0),] ) ,ytrain[which(index==0)])
    score2 = score( predict(fit2,xtrain[which(index==1),] ) ,ytrain[which(index==1)])
    score3 = score( predict(fit3,xtrain[which(index==2),] ) ,ytrain[which(index==2)])
    
    yi = which.max(c(score1,score2,score3))
    fit = randomForest(xtrain[-c(which(xindex==(yi-1))),], ytrain[-c(which(xindex==(yi-1)) )] ,ntree=200,mytry = 4676) 
    
    num = predict(fit, xtest  ) + mean(y)
  }
  
  return(num)
}

sipred =  function(a,id,method){
  n = nrow(a)
  y = na.omit( diff( as.matrix(a[,(id+1)]) ) )
  ny = length( y )
  nn = min(ny,n-5)
  x = as.matrix( a[ (n-nn-4):n ,-c(1,(id+1)) ] )
  y = as.numeric( y[(ny-nn+1):ny] >0)
  x = x[,complete.cases(t(x) )]
  n=nrow(x)
  # 標準化
  xt = scale(x)*sqrt(nrow(x)/(nrow(x)-1))
  yt = y 
  set.seed(123)
  
  xtrain = xt[1:(n-5),];  ytrain = yt;  xpred = xt[(n-4):n,]
  testid = sample(1:nrow(xtrain),100,replace = F )
  train = cbind(ytrain[-testid],xtrain[-testid,])
  test = cbind(ytrain[testid],xtrain[testid,])
  params = list(colsample_bytree = 0.7,subsample = 0.7  ,  booster = "gbtree",                   
                max_depth = 3, eta = 0.03, eval_metric = "rmse", objective = "binary:logistic",gamma = 0) 
  ty = CreateFolds(train,3)
  ##############################################################
  if(method==1){
    model_1 = randomForest(x= train[-which(ty==0),-1],y=factor(train[-which(ty==0),1]),importance = T,ntree=500)
    model_2 = randomForest(x= train[-which(ty==1),-1],y=factor(train[-which(ty==1),1]),importance = T,ntree=500)
    model_3 = randomForest(x= train[-which(ty==2),-1],y=factor(train[-which(ty==2),1]),importance = T,ntree=500)
    
    score1 = mean( predict(model_1 , train[which(ty==0),-1]) == train[which(ty==0),1])
    score2 = mean( predict(model_2 , train[which(ty==1),-1]) == train[which(ty==1),1])
    score3 = mean( predict(model_3 , train[which(ty==2),-1]) == train[which(ty==2),1])
    bp = which.max(c(score1,score2,score3))-1
    model = randomForest(x=train[-which(ty==bp),-1],y=factor(train[-which(ty==bp),1]),importance = T,ntree=5)
    num = as.numeric(predict(model , xpred))-1
  }
  if(method==2){
    model_1 = svm( factor(V1) ~.,train[-which(ty==0),],scale=F)
    model_2 = svm( factor(V1) ~.,train[-which(ty==1),],scale=F)
    model_3 = svm( factor(V1) ~.,train[-which(ty==2),],scale=F)
    
    score1 = mean( predict(model_1 ,train[which(ty==0),-1]) == train[which(ty==0),1])
    score2 = mean( predict(model_2 ,train[which(ty==1),-1]) == train[which(ty==1),1])
    score3 = mean( predict(model_3 ,train[which(ty==2),-1]) == train[which(ty==2),1])
    
    bp = which.max(c(score1,score2,score3))-1
    model =  svm( factor(V1) ~.,train[-which(ty==bp),],scale=F)
    num = as.numeric(predict(model , xpred))-1
  }
  if(method==3){
    model_1 = xgboost(params=params,data = train[-which(ty==0),-1],label = train[-which(ty==0),1],
                      nfold=5,nrounds=3,verbose = 0)
    
    model_2 = xgboost(params=params,data = train[-which(ty==1),-1],label = train[-which(ty==1),1],
                      nfold=5, nrounds=3,verbose = 0)
    
    model_3 = xgboost(params=params,data = train[-which(ty==2),-1],label = train[-which(ty==2),1],
                      nfold=5,nrounds=3,verbose = 0)
    
    score1 = mean( as.numeric(predict(model_1 ,train[which(ty==0),-1])>=0.5) == train[which(ty==0),1])
    score2 = mean( as.numeric(predict(model_2 ,train[which(ty==1),-1])>=0.5) == train[which(ty==1),1])
    score3 = mean( as.numeric(predict(model_3 ,train[which(ty==2),-1])>=0.5) == train[which(ty==2),1])
    
    bp = which.max(c(score1,score2,score3))-1
    model =   xgboost(params=params,data = train[-which(ty==bp),-1],label = train[-which(ty==bp),1],
                      nfold=5,nrounds=3,verbose = 0)
    num = as.numeric(predict(model ,xpred)>=0.5)
  }
  qqnum = 2*num-1
  return(qqnum)
}

##################################
s1 = s2 = s3 =  matrix(NA,nrow=18,ncol=5)
#1
for (i in 1:18) {
  s1[i,] = sipred(aa,i,1) 
  print( s1[i,] )
}
#2
for (i in 1:18) {
  s2[i,] = sipred(aa,i,2) 
  print( s2[i,] )
}
#3
for (i in 1:18) {
  s3[i,] = sipred(aa,i,3) 
  print( s3[i,] )
} # sipred (要修改)
#############################################
k1 = k2 = k3 = k4 = k5 =  matrix(NA,nrow=18,ncol=5)

k1 = t(sapply(1:18,function(i) pred(ab4,i,1) ));k1
k2 = t(sapply(1:18,function(i) pred(ab4,i,2) ));k2
k3 = t(sapply(1:18,function(i) pred(ab4,i,3) ));k3
k4 = t(sapply(1:18,function(i) pred(ab4,i,4) ));k4

sum(sign(qqq)==sign(k1))
sum(sign(qqq)==sign(k4))

sum((sign(qqq)==sign(k1))%*%c(0.1,0.15,0.2,0.25,0.3))
sum((sign(qqq)==sign(k2))%*%c(0.1,0.15,0.2,0.25,0.3))
sum((sign(qqq)==sign(k3))%*%c(0.1,0.15,0.2,0.25,0.3))
sum((sign(qqq)==sign(k4))%*%c(0.1,0.15,0.2,0.25,0.3))


############################################### 
# Stacking 
pred2 =  function(a,id){
  n = nrow(a)
  y = na.omit( diff( as.matrix(a[,(id+1)]) ) )
  ny = length( y )
  nn = min(ny,n-5)
  x = as.matrix( a[ (n-nn-4):n ,-c(1,(id+1)) ] )
  y = y[(ny-nn+1):ny]
  # 標準化
  xt = scale(x)*sqrt(nrow(x)/(nrow(x)-1))
  yt = y - mean(y)
  
  set.seed(123)
  xtrain = xt[1:nn,]
  ytrain = yt
  xpred = xt[(nn+1):(nn+5),]
  
  xt.com = xt[  ,complete.cases( t(xt) )  ]
  xpred.com = xt.com[(nn+1):(nn+5),]
  
  testid = sample(1:nn,100,replace = F)
  test = cbind(ytrain[testid],xtrain[testid,])
  train = cbind(ytrain[-testid],xtrain[-testid,])
  
  xindex =  CreateFolds(train,3)
  # CV
  train1=train[c(which(xindex==0)),];    train2=train[c(which(xindex==1)),]
  train3=train[c(which(xindex==2)),]
  ################################ 1 ###################
  meta.x = vector()
  meta.y = list()
  
  stacking.train = rbind(train1,train2)
  stacking.valid = train3
  stacking.test = test
  
  model_1 =  Ohit(X= stacking.train[,-1],y= stacking.train[,1])
  
  tmp.meta.x = predict_Ohit(model_1, stacking.valid[,-1])$pred_HDIC
  tmp.meta.y = predict_Ohit(model_1, stacking.test[,-1])$pred_HDIC
  star1_1 = predict_Ohit(model_1, xpred )$pred_HDIC
  #print(star1_1+mean(y))
  
  meta.x = c(meta.x, tmp.meta.x)
  meta.y[[1]] = tmp.meta.y
  
  # 2nd fold for validation
  stacking.train = rbind(train1,train3)
  stacking.valid = train2
  stacking.test = test
  
  model_1 =  Ohit(X= stacking.train[,-1],y= stacking.train[,1])
  
  tmp.meta.x = predict_Ohit(model_1, stacking.valid[,-1])$pred_HDIC
  tmp.meta.y = predict_Ohit(model_1, stacking.test[,-1])$pred_HDIC
  star1_2 = predict_Ohit(model_1, xpred )$pred_HDIC
  #print(star1_2+mean(y))
  meta.x = c(meta.x, tmp.meta.x)
  meta.y[[2]] = tmp.meta.y
  
  # 3nd fold for validation
  stacking.train = rbind(train2,train3)
  stacking.valid = train1
  stacking.test = test
  
  model_1 =  Ohit(X= stacking.train[,-1],y= stacking.train[,1])
  star1_3 = predict_Ohit(model_1, xpred )$pred_HDIC
  #print(star1_3+mean(y))
  tmp.meta.x = predict_Ohit(model_1, stacking.valid[,-1])$pred_HDIC
  tmp.meta.y = predict_Ohit(model_1, stacking.test[,-1])$pred_HDIC
  
  meta.x = c(meta.x, tmp.meta.x)
  meta.y[[3]] = tmp.meta.y
  ################
  mean.meta.y = (meta.y[[1]] + meta.y[[2]] + meta.y[[3]]) / 3
  
  meta.train.1 = data.frame(`meta.x` = meta.x,  y=train[,1])
  
  meta.test.1 = data.frame(`mete.y` = mean.meta.y, y = test[,1])
  
  ########################### (2) ##################
  meta.x = vector()
  meta.y = list()
  
  xtrain.com =  xtrain[,complete.cases( t(xtrain))]
  test.com = cbind(ytrain[testid],xtrain.com[testid,])
  train.com = cbind(ytrain[-testid],xtrain.com [-testid,])
  
  train1.com=train.com[c(which(xindex==0)),];    train2.com=train.com[c(which(xindex==1)),]
  train3.com=train.com[c(which(xindex==2)),]
  
  # 1st fold for validation
  stacking.train = rbind(train1,train2)
  stacking.valid = train3
  stacking.test = test
  xgb.params = list(colsample_bytree = 0.8, subsample = 0.7,booster = "gbtree",max_depth = 3,
                    eta = 0.03,eval_metric = "rmse",objective = "reg:linear",gamma = 0)      
  
  dtrain.f1 = xgb.DMatrix(data = stacking.train[,-1], label = stacking.train[,1])
  
  model_2 = xgb.train(paras = xgb.params,data = dtrain.f1,nrounds = 71,verbose = 0)
  
  
  tmp.meta.x = predict(model_2, stacking.valid[,-1] )
  tmp.meta.y = predict(model_2, stacking.test[,-1] )
  star2_1 = predict(model_2,xpred)
  #print(star2_1+mean(y) )
  meta.x = c(meta.x, tmp.meta.x)
  meta.y[[1]] = tmp.meta.y
  
  # 2nd fold for validation
  stacking.train = rbind(train1,train3)
  stacking.valid = train2
  stacking.test = test
  
  dtrain.f1 = xgb.DMatrix(data = stacking.train[,-1], label = stacking.train[,1])
  model_2 = xgb.train(paras = xgb.params,data = dtrain.f1,nrounds = 71,verbose = 0)
  
  tmp.meta.x = predict(model_2, stacking.valid[,-1])
  tmp.meta.y = predict(model_2, stacking.test[,-1])
  star2_2 = predict(model_2,xpred)
  #print(star2_2+mean(y))
  meta.x = c(meta.x, tmp.meta.x)
  meta.y[[2]] = tmp.meta.y
  
  # 3rd fold for validation
  stacking.train = rbind(train2,train3)
  stacking.valid = train1
  stacking.test = test
  
  dtrain.f1 = xgb.DMatrix(data = stacking.train[,-1], label = stacking.train[,1])
  model_2 = xgb.train(paras = xgb.params,data = dtrain.f1,nrounds = 71,verbose = 0)  
  
  tmp.meta.x = predict(model_2, stacking.valid[,-1])
  tmp.meta.y = predict(model_2, stacking.test[,-1])
  star2_3 = predict(model_2,xpred)
  #print(star2_3+mean(y))
  meta.x = c(meta.x, tmp.meta.x)
  meta.y[[3]] = tmp.meta.y
  
  # Average Meta.X of Test
  mean.meta.y = (meta.y[[1]] + meta.y[[2]] + meta.y[[3]] ) / 3
  meta.train.2 = data.frame(`meta.x` = meta.x,  y=train.com[,1])
  meta.test.2 = data.frame(`mete.y` = mean.meta.y, y = test.com[,1])
  
  
  ###################################################################    
  c(dim(meta.train.1), dim(meta.test.1))
  # 所以現在要建構第二階段的 Meta-Model，我這裡拿xgboost的模型來做：
  ### Meta- Model Construction 
  # 先把三個 Meta-Train合併一起：
  big.meta.train = rbind(meta.train.1, meta.train.2)
  # 轉換成 xgboost 的格式
  dtrain = xgb.DMatrix(data = as.matrix(big.meta.train[,1]), label = big.meta.train[, 2])
  #xgb.params = list(colsample_bytree = 0.8, subsample = 0.7,booster = "gbtree",max_depth = 3,
  #                  eta = 0.03,eval_metric = "rmse",objective = "reg:linear",gamma = 0)      
  
  xgb.model = xgb.train(paras = xgb.params, data = dtrain, nrounds = 71) 
  
  # 對三組 Meta-Test進行預測：
  dtest.1 = xgb.DMatrix(data = as.matrix(meta.test.1[,1]), label = meta.test.1[, 2])
  final_1 = predict(xgb.model, dtest.1)
  
  dtest.2 = xgb.DMatrix(data = as.matrix(meta.test.2[,1]), label = meta.test.2[, 2])
  final_2 = predict(xgb.model, dtest.2)
  predfinal1 = predict(xgb.model ,  as.matrix( star1_1) )
  predfinal2 = predict(xgb.model ,as.matrix(  star1_2))
  #print(predfinal2)
  predfinal3 = predict(xgb.model , as.matrix( star1_3))
  predfinal4 = predict(xgb.model , as.matrix(star2_1))
  predfinal5 = predict(xgb.model , as.matrix(star2_2))
  predfinal6 = predict(xgb.model , as.matrix(star2_3))
  
  predfinal = rbind(predfinal1,predfinal2,predfinal3,predfinal4,predfinal5,predfinal6)
  #print(predfinal)
  finalp =  apply(predfinal,2,mean)  + mean(y)
  # 把三組結果平均起來，然後算 MSE
  final_y = (final_1 + final_2 )/2
  mse = mean((final_y - test[,1])^2) # MSE
  return(list(pred = finalp ,mse = mse) )
}
u=123
#########################################################################
# 從歷史資料抽出 k1-k5   (找最優化個別公司模型混合比例 )
E1 = list()
ptm <- proc.time()
for(i in 0:22){
  nq=nrow(ab)
  data = ab[1:(nq-5*i),]
  e1 = t(sapply(1:18, function(j) pred(data,j,1)  ))
  e2 = t(sapply(1:18, function(j) pred(data,j,2)  ))
  e3 = t(sapply(1:18, function(j) pred(data,j,3)  ))
  e4 = t(sapply(1:18, function(j) pred(data,j,4)  ))
  E1[[i+1]] = cbind(e1,e2,e3,e4)
  print(i)
}
proc.time() - ptm

## 抽出歷史資料的 真實漲幅
bb=list()
for(i in 0:(length(E1)-1) ){
  data.star = ab[c( (nrow(ab)-5*i-5):(nrow(ab)-5*i) )  ,2:19]
  data.diff = matrix(NA,nrow=18,ncol=5)
  for(j in 1:18){
    data.diff[j,] = as.numeric(diff( as.matrix( data.star[,j] ))  )
  }
  bb[[i+1]] = data.diff
}

######  比例  #####################################
# 方法一 (stacking)
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
  
  fit1 =  xgboost(data = xx[1:(n-5),] ,label = as.numeric( true),params,
                 nrounds = 15,verbose = 0)
  w1 =  predict(fit1,xx[1:(n-5),]   )
  ww1 = predict(fit1,xx[(n-4):n, ]   )
  
  
  data1 = as.data.frame(cbind(true,xx[1:(n-5),]  )  )

  fit2 = svm(true~ . ,data = data1  )
  w2 = predict(fit2,as.data.frame( xx[1:(n-5), ]  )   )
  ww2 = predict(fit2,as.data.frame( xx[(n-4):n, ]  ) )
  
  fit3 = glm(true~ . ,data = data1  )
  w3 = predict(fit3,as.data.frame( xx[1:(n-5), ]  )   )
  ww3 = predict(fit3,as.data.frame( xx[(n-4):n, ]  ) )
  
  fit = xgboost(data = cbind(w1,w2,w3 ) ,label = as.numeric( true),params,
               nrounds = 30,verbose = 0)
  ww = predict(fit,cbind(ww1,ww2,ww3)   )

  
  return(ww + mean(true))
}
t(sapply(1:18, function(x) blr(x)) )


#-------1----------------------------------------------------------------
# 方法二
# 計算最優比例 (0-100):(0-100) 依序OHCV+XGCV1+XGCV2  
# day 擺資料距今時間 , id 擺公司編號 

# 4個
cscore = function(id){   # predt 擺 E1[[i]]  i=1,2,3,4,5
  true = as.numeric(sapply(1:length(E1),function(x) bb[[x]][id,]) )
  pred1 = as.numeric( sapply(1:length(E1),function(x) E1[[x]][id,1:5] )   ) 
  pred2 = as.numeric( sapply(1:length(E1),function(x) E1[[x]][id,6:10] )   ) 
  pred3 = as.numeric( sapply(1:length(E1),function(x) E1[[x]][id,11:15] )   ) 
  pred4 = as.numeric( sapply(1:length(E1),function(x) E1[[x]][id,16:20] )   ) 
  
  ddd = list()
  for (i in 0:100) {
    mm1 = matrix(0,ncol=101,nrow=101)
    for (j in 0:(100-i) ){
      for(k in 0:(100-j-i) ){
        mix = 0.01*i*pred1 +0.01*j*pred2 + 0.01*k*pred3 +(1-0.01*i-0.01*j-0.01*k)*pred4
        sc = sum( c(sign(true)==sign(mix))*rep(c(0.1,0.15,0.2,0.25,0.3),20) ) - 
               sum(abs(mix-true))*0.0005
        mm1[j+1,k+1] = sc
        #print(c(i,j,k,sc))
        #sc =  sum(as.matrix(sign(true)==sign(mix10))%*%c(0.1,0.15,0.2,0.25,0.3))*0.5 +
        #  sum( 1-(abs(true-mix)/t(as.matrix(aa[(nrow(aa)-5*day-4 ):(nrow(aa)-5*day),2:19])))
        #%*%c(0.1,0.15,0.2,0.25,0.3) )*0.5
      } 
    }
    ddd[[i+1]] = mm1
  }
  return( ddd  )
}

gold = function(id){
  abc = cscore(id)
  besti = which.max(  lapply(abc, function(x) max(x) )  )
  bestj = which.max( apply(abc[[besti]], 1,max  ) )
  bestk = which.max( apply(abc[[besti]], 2,max  ) )
  besti = besti -1;  bestj = bestj -1 
  bestk = bestk -1;  bestl = 100 - besti-bestj-bestk
  return(c(besti,bestj,bestk,bestl))
}

t(sapply(1:18,function(i) cbind( k1[i,],k2[i,],k3[i,],k4[i,])%*%gold(i)*0.01   ))

############
# 3個
cscore1 = function(id){   # predt 擺 E1[[i]]  i=1,2,3,4,5
  true = c();pred1=c();pred2=c();pred3=c();pred4=c()
  for(q in 1:length(E1)){
    true = c(true,as.numeric( bb[[q]][id,]) )
    pred1 = c(pred1, as.numeric( E1[[q]][id,1:5] )   )
    pred2 = c(pred2, as.numeric( E1[[q]][id,6:10] )   )
    pred3 = c(pred3, as.numeric( E1[[q]][id,11:15] )   )
  }
  mm1 = matrix(0,ncol=101,nrow=101)
  for (i in 0:100) {
    for (j in 0:(100-i) ){
        mix = 0.01*i*pred1 +0.01*j*pred2 + (1-0.01*j-0.01*i)*pred3 
        sc = sum( c(sign(true)==sign(mix))*rep(c(0.1,0.15,0.2,0.25,0.3),20) ) - 
          sum(abs(mix-true))*0.0005
        mm1[i+1,j+1] = sc
        #print(c(i,j,k,sc))
        #sc =  sum(as.matrix(sign(true)==sign(mix10))%*%c(0.1,0.15,0.2,0.25,0.3))*0.5 +
        #  sum( 1-(abs(true-mix)/t(as.matrix(aa[(nrow(aa)-5*day-4 ):(nrow(aa)-5*day),2:19])))
        #%*%c(0.1,0.15,0.2,0.25,0.3) )*0.5
    }
  }
  return( mm1  )
}
wtf = function(id){
  besti = which.max( apply(cscore1(id), 1,max  ) ) -1
  bestj = which.max( apply(cscore1(id), 2,max  ) ) -1
  bestk = 100 - besti -bestj
  return(c(besti,bestj,bestk)  )
}


t(sapply(1:18,function(i) cbind( k1[i,],k2[i,],k3[i,])%*%wtf(i)*0.01   ))





#as.numeric(   E1[[1]][,1:5] )
#as.numeric(  E1[[1]][,6:10] )
#as.numeric( E1[[1]][,11:15] )
#cbind(as.numeric(   E1[[1]][,1:5] ),as.numeric(  E1[[1]][,6:10] ),as.numeric( E1[[1]][,11:15] ))
#as.numeric( bb[[1]] )
#xgboost(data = cbind(as.numeric(   E1[[1]][,1:5] ),as.numeric(  E1[[1]][,6:10] ),as.numeric( E1[[1]][,11:15] )),
#label = as.numeric( bb[[1]] ) ,eta = 0.03, nthread = 8,nrounds = 200,
#             gamma = 0.0468, max_depth = 6)


#write.csv(qw,file = "cbind.csv")
#--------------------------------------------------------------------------------------------
# k1,k2 漲幅預測




##=======================================================
# 總分數計算 (輸入一階差分)   5/25
qq = scan()
0.35	-0.35	-1.45	0.35	0.6
0.21	0.1	-0.5	0.24	0.23
-0.1	-0.15	-1	0.05	0.45
0.09	-0.13	-0.27	0.01	0.26
0.27	0	-0.33	-0.01	0.1
0.02	-0.06	-0.24	0.02	0.06
0.06	-0.02	-0.36	0.18	0.21
0.05	-0.4	-0.66	0.29	0.22
0.27	0.2	-0.67	0.23	-0.03
0.17	-0.05	-1.09	0.23	0.15
0.22	-0.13	-0.07	0.09	0
0.06	-0.11	-0.55	-0.02	0.16
0.25	-0.25	-0.8	0.25	0.45
0.41	-0.47	-0.64	-0.06	0.1
0.05	-0.07	-0.3	-0.04	0.27
0.11	-0.1	-0.23	0.1	0.07
0.03	-0.16	-0.3	0.09	0.16
0.16	-0.04	-0.37	0.06	0.15

# 符號分數計算
qqq = matrix(qq ,nrow=18,ncol=5 ,by=1)

sum(s1==sign(qqq))
sum(s2==sign(qqq))
sum(s3==sign(qqq))

sum(sign(k1)==sign(qqq))
sum(sign(k2)==sign(qqq))
sum(sign((k1+k2)/2)==sign(qqq))

for(k in 0:100){
  for(j in 0:100){
    for (i in 0:100) {
      value = 0.01*i*k1+0.01*j*k2+0.01*k*k3+(1-0.01*i-0.01*j-0.01*k)*k4
      xd = sum((sign(value)==sign(qqq))%*%c(0.1,0.15,0.2,0.25,0.3))*0.5
      if(xd >5.2){    print(c(0.01*i,0.01*j,0.01*k,xd)) }
    }
  }
}


print(sum(sign(0.3*k1+0.7*k2)==sign(qqq)))
sum(sign(0.2*k1+0.8*k2)==sign(qqq))

sum(sign(0*k1+1*k2) ==sign(0.5*k1+0.5*k2)  )
sum(sign(0*k1+1*k2) ==sign(1*k1+0*k2)  )

sum(sign(0.3*k1+0.7*k2) ==sign(0.5*k1+0.5*k2)  )
sum(sign(0.4*k1+0.6*k2) ==sign(0.5*k1+0.5*k2)  )

#=--------------------------------------------------------------------------------
### time series 
acf(ts(a[,2]))
pacf(ts(a[,2]))

acf(diff( ts(a[,2])))
pacf(diff( ts(a[,2])))

arima(ts(a[,2]),order=c(2,1,0))
ts.plot(a[,2])



#########################################################################################################################
# 測試區
## Set up caret model training parameters
CARET.TRAIN.CTRL <- trainControl(method = "repeatedcv", number = 5, repeats = 5, 
                                 verboseIter = FALSE, allowParallel = TRUE)
gbmFit <- train(SalePrice ~ ., method = "gbm", metric = "RMSE", maximize = FALSE, 
                trControl = CARET.TRAIN.CTRL, tuneGrid = expand.grid(n.trees = (4:10) * 
                                                                       50, interaction.depth = c(5), shrinkage = c(0.05), n.minobsinnode = c(10)), 
                data = Training, verbose = FALSE)
## print(gbmFit)

## Predictions
preds1 <- predict(gbmFit, newdata = Validation)
rmse(Validation$SalePrice, preds1)


cv_lasso = cv.glmnet(x = as.matrix(qwer[1:1314,c(-1,-2)]), y = as.matrix(qwer[1:1314,2]) ,nfolds = 5  )
predict(cv_lasso, newx = as.matrix(qwer[1315:1317,c(-1,-2)]), s = "lambda.min")
plot(cv_lasso)

## Predictions
preds <- predict(cv_lasso, newx = as.matrix(Validation[, -59]), s = "lambda.min")
rmse(Validation$SalePrice, preds)






dtrainww <- xgb.DMatrix(data =scale( as.matrix(  qwer[2:1290,c(-1,-2)])) , label = ffff[1:1289] )
ffffff = xgboost(data = dtrainww,max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")


predict(ffffff,  scale(as.matrix( qwer[1291:1317,c(-1,-2)]) ) )
ffff[1290:1316]
prediction <- as.numeric(predict(ffffff,  scale(as.matrix( qwer[1291:1317,c(-1,-2)]) ) ) > 0.5)


table(prediction, ffff[1290:1316])





