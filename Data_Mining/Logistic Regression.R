# 필요한 라이브러리 설치
library(dplyr)
library(gsubfn)
library(proto)
library(RSQLite)
library(sqldf)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(psych)
library(dummies)
library(caret)

perf_eval <- function(cm){
  # true positive rate
  TPR = Recall = cm[2,2]/sum(cm[2,])
  # precision
  Precision = cm[2,2]/sum(cm[,2])
  # true negative rate
  TNR = cm[1,1]/sum(cm[1,])
  # accuracy
  ACC = sum(diag(cm)) / sum(cm)
  # balance corrected accuracy (geometric mean)
  BCR = sqrt(TPR*TNR)
  # f1 measure
  F1 = 2 * Recall * Precision / (Recall + Precision)
  
  re <- data.frame(TPR = TPR,
                   Precision = Precision,
                   TNR = TNR,
                   ACC = ACC,
                   BCR = BCR,
                   F1 = F1)
  return(re)
}

nor_minmax = function(x){
  result = (x - min(x)) / (max(x) - min(x))
  return(result)
}

# 로지스틱 회귀 데이터 로드 및 탐색  
data <- read.csv("LogisticRegression/train.csv")
summary(data)
str(data)

# 변수 자료형이 chr일 경우 summary 시 정보를 제대로 보여주지 않아 factor 변수로 처리
# 결측치 공백'' 은 NA 로 변환 후 제거 
data$workclass <-ifelse(data$workclass=='',NA,as.character(data$workclass))
data$workclass <- as.factor(data$workclass)

data$education <- as.factor(data$education)
data$marital.status <- as.factor(data$marital.status)

data$occupation <-ifelse(data$occupation =='',NA,as.character(data$occupation))
data$occupation <- as.factor(data$occupation)

data$relationship <- as.factor(data$relationship)
data$race<- as.factor(data$race)
data$sex <- as.factor(data$sex)

data$native.country <-ifelse(data$native.country =='',NA,as.factor(data$native.country))
data$native.country <- as.factor(data$native.country)
summary(data)

data <- na.omit(data)
sum(is.na(data))
# -------------이하 결측치 처리 및 데이터 형변환---------------


Work_class<-sqldf('SELECT workclass, count(workclass) as Count 
                  ,sum(target) as Above from data group by workclass')
Work_class$Below<-Work_class$Count-Work_class$Above
table<-data.frame(Class=Work_class$workclass, Proportion=Work_class$Above/Work_class$Count)
Work_class<-Work_class[,c(1,3,4)]
Workclass<-melt(Work_class,id.vars = 'workclass')
gg<-ggplot(Workclass,aes(x=workclass,y=value,fill=variable))+
  geom_bar(stat = 'identity',position = 'stack')+theme_bw()+
  scale_fill_manual(values = c('red','green'))+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle('Proprtions of above-paid within different classes')
tbl <- tableGrob(t(table), rows=NULL)
grid.arrange(tbl, gg,
             nrow=2,
             as.table=TRUE,
             heights=c(1,4))

# 소득분위에  따른 target 비율 시각화
education<-sqldf('SELECT education, count(education) as Count 
                  ,sum(target) as Above from data group by education')
education$Below<-education$Count-education$Above
table<-data.frame(Class=education$education, Proportion=education$Above/education$Count)
education<-education[,c(1,3,4)]
edu<-melt(education,id.vars = 'education')
gg<-ggplot(edu,aes(x=education,y=value,fill=variable))+
  geom_bar(stat = 'identity',position = 'stack')+theme_bw()+
  scale_fill_manual(values = c('red','green'))+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle('Proportions of above-paid within different education level')
tbl <- tableGrob(t(table), rows=NULL)
grid.arrange(tbl, gg,
             nrow=2,
             as.table=TRUE,
             heights=c(1,4))

colnames(data)[7]<-'Marital'
marital<-sqldf('SELECT Marital, count(Marital) as Count 
                  ,sum(target) as Above from data group by Marital')
marital$Below<-marital$Count-marital$Above
table<-data.frame(Marital=marital$Marital, Proportion=marital$Above/marital$Count)
marital<-marital[,c(1,3,4)]
mar<-melt(marital,id.vars = 'Marital')
gg<-ggplot(mar,aes(x=Marital,y=value,fill=variable))+
  geom_bar(stat = 'identity',position = 'stack')+
  theme_bw()+scale_fill_manual(values = c('red','green'))+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle('Proportions of above-paid within different marital status')
tbl <- tableGrob(t(table), rows=NULL)
grid.arrange(tbl, gg,
             nrow=2,
             as.table=TRUE,
             heights=c(1,4))

race<-sqldf('SELECT race, count(race) as Count 
                  ,sum(target) as Above from data group by race')
race$Below<-race$Count-race$Above
table<-data.frame(race=race$race, Proportion=race$Above/race$Count)
race<-race[,c(1,3,4)]
rac<-melt(race,id.vars = 'race')
gg<-ggplot(rac,aes(x=race,y=value,fill=variable))+geom_bar(stat = 'identity',position = 'stack')+theme_bw()+scale_fill_manual(values = c('red','green'))+theme(axis.text.x = element_text(angle = 45, hjust = 1))+ggtitle('Proportions of above-paid within different races')
tbl <- tableGrob(t(table), rows=NULL)
grid.arrange(tbl, gg,
             nrow=2,
             as.table=TRUE,
             heights=c(1,4))

# 성별에 따른 50k이상 소득분위 비율
sex<-sqldf('SELECT sex, count(sex) as Count 
                  ,sum(target) as Above from data group by sex')
sex$Below<-sex$Count-sex$Above
table<-data.frame(sex=sex$sex, Proportion=sex$Above/sex$Count)
sex<-sex[,c(1,3,4)]
se<-melt(sex,id.vars = 'sex')
gg<-ggplot(se,aes(x=sex,y=value,fill=variable))+
  geom_bar(stat = 'identity',position = 'stack')+
  theme_bw()+scale_fill_manual(values = c('red','green'))+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle('Proportions of above-paid within different sexes')
tbl <- tableGrob(t(table), rows=NULL)
grid.arrange(tbl, gg,
             nrow=2,
             as.table=TRUE,
             heights=c(1,4))

colnames(data)[15]<-'country'
country<-sqldf('SELECT country, count(country) as Count 
                  ,sum(target) as Above from data group by country')
country$Below<-country$Count-country$Above
table<-data.frame(country=country$country, Proportion=country$Above/country$Count)
country<-country[,c(1,3,4)]
se<-melt(country,id.vars = 'country')
gg<-ggplot(se,aes(x=country,y=value,fill=variable))+
  geom_bar(stat = 'identity',position = 'stack')+
  theme_bw()+scale_fill_manual(values = c('red','green'))+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle('Proportions of above-paid within different country')
tbl <- tableGrob(t(table), rows=NULL)
grid.arrange(tbl, gg,
             nrow=2,
             as.table=TRUE,
             heights=c(1,4))

gg<-qplot(fnlwgt, data=data, geom="histogram")+
  theme_bw()+ggtitle('Histogram of fnlwgt')
gg

colnames(data)[12]<-'CapitalGain'
gg<-qplot(CapitalGain, data=data, geom="histogram")+
  theme_bw()+ggtitle('Histogram of Capital Gain')
gg

colnames(data)[13]<-'CapitalLoss'
gg<-qplot(CapitalLoss, data=data, geom="histogram")+
  theme_bw()+ggtitle('Histogram of Capital Loss')
gg
# Brief Conclusion: The capital loss may not be useful for classification 
# as it is quite skewed and mostly concentrated at zero value. ## Working Hours

colnames(data)[14]<-'Hours'
gg<-qplot(Hours, data=data, geom="histogram")+
  theme_bw()+ggtitle('Histogram of Working Hours')
gg

df2<-data[,-c(1)]
df2$HourJ<-ifelse(df2$Hours<=40,'NormalWorkLoad','HugeWorkLoad')
wl<-sqldf('SELECT HourJ as WorkLoad, count(HourJ) as Count, sum(target) 
          as Above from df2 group by HourJ')
wl$Below<-wl$Count-wl$Above
Percentage<-wl$Above/wl$Count
wl<-wl[,c(1,3,4)]
wlt<-melt(wl,id.vars = 'WorkLoad')
wl<-cbind(wl,Percentage)
gg<-ggplot(wlt,aes(x=WorkLoad,y=value,fill=variable))+
  geom_bar(stat = 'identity',position = 'stack')+
  theme_bw()+scale_fill_manual(values = c('red','green'))+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle('Proportions of above-paid with different Work Load')
tbl <- tableGrob(t(wl[,c(1,4)]), rows=NULL)
grid.arrange(tbl, gg,
             nrow=2,
             as.table=TRUE,
             heights=c(1,4))
# Brief Conclusion: Work harder, life better, isn’t it?

summary(data)
df3<-data[,-c(1,12,13,15)]
summary(df3)

# 연속형 변수 정규화 진행
str(df3)
df3$age <- nor_minmax(df3$age)
df3$fnlwgt <- nor_minmax(df3$fnlwgt)
df3$education.num <- nor_minmax(df3$education.num)
df3$hours.per.week <- nor_minmax(df3$hours.per.week)
summary(df3)

# 범주형 변수(categorical variable) -> 더미 변수(Dummy variable; 0 or 1) 변환
dummy <- dummyVars(" ~ .", data=df3)
df3 <- data.frame(predict(dummy,newdata=df3))
head(df3)

# split train & test set
set.seed(2022)
test_id <- sample(1:nrow(df3), round(nrow(df3)*0.7))
dat_train <- df3[-test_id, ]
dat_test <- df3[test_id, ]

head(dat_train)
print("Training: ", str(nrow(dat_train)))

model <- glm(target ~ . , family = binomial(link='logit'), dat_train )
summary(model)

pred_income <- predict(model, dat_test, type="response")
pred_class <- rep(0, nrow(dat_test))
pred_class[pred_income > 0.5] <- 1
cm <- table(pred=pred_class, actual=dat_test$target)
perf_eval(cm)

# 변수선택법 - 휴리스틱 p-value 높은 변수 없애기
df4<-data[,-c(1,3,12,13,15)]
summary(df4)
# 범주형 변수(categorical variable) -> 더미 변수(Dummy variable; 0 or 1) 변환
dummy <- dummyVars(" ~ .", data=df4)
df4 <- data.frame(predict(dummy,newdata=df4))
df4<-as.numeric(df4)
head(df4)
# split train & test set
set.seed(2022)
test_id2 <- sample(1:nrow(df4), round(nrow(df4)*0.7))
dat_train2 <- df4[-test_id2, ]
dat_test2 <- df4[test_id2, ]

head(dat_train2)
print("Training: ", str(nrow(dat_train2)))

model2 <- glm(target ~ . , family = binomial(link='logit'), dat_train2 )
summary(model2)

pred_income2 <- predict(model2, dat_test2, type="response")
pred_class2 <- rep(0, nrow(dat_test2))
pred_class2[pred_income2 > 0.5] <- 1
cm2 <- table(pred=pred_class2, actual=dat_test2$target)
perf_eval(cm2)

#두 모델 비교
perf_eval(cm)
perf_eval(cm2)

# 변수선택법 - 전진선택법
model_fwd <- step(glm(target ~ 1, dat_train, 
                      family = binomial()), 
                  direction = "forward", trace = 0,
                  scope = formula(model))
pred_income3 <- predict(model_fwd, dat_test, type="response")
pred_class3 <- rep(0, nrow(dat_test))
pred_class3[pred_income3 > 0.5] <- 1
cm3 <- table(pred=pred_class3, actual=dat_test$target)
perf_eval(cm3)

perf_eval(cm)
perf_eval(cm2)
perf_eval(cm3)
