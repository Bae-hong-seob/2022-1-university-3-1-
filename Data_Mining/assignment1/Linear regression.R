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

normailzation_minmax <- function(x){
  result <- (x-min(x)) / (max(x) - min(x))
  return(result)
}

# 목적 : 어떤 변수가 차 값에 가장 많은 영향을 줄까?
# - Which variables are significant in predicting the price of a car
# - How well those variables describe the price of a car

# 데이터 로드 및 탐색
df <- read.csv("LinearRegression/CarPrice_Assignment.csv")
sum(is.na(df))
str(df)
summary(df)

# 데이터 형변환
df$CarName <- as.factor(df$CarName)
df$fueltype <- as.factor(df$fueltype)
df$aspiration <- as.factor(df$aspiration)
df$doornumber <- as.factor(df$doornumber)
df$carbody <- as.factor(df$carbody)
df$drivewheel <- as.factor(df$drivewheel)
df$enginelocation <- as.factor(df$enginelocation)
df$enginetype <- as.factor(df$enginetype)
df$cylindernumber <- as.factor(df$cylindernumber)
df$fuelsystem <- as.factor(df$fuelsystem)

summary(df)
str(df)
sum(is.na(df))

# 변수선택 
# 차 값(price)를 예측하는데 있어 중요한 지표만 추출하고자 한다. 
# 이때 브랜드가 어디냐에 따라가 아닌 차의 객관적인 상태에 따라 차값을 예측하고자
# 차 이름에 대한, 브랜드가 설명되어있는 칼럼은 삭제하였다.
# id 또한 설명력이 없는 변수이므로 삭제
names(df)
df <- df[,-c(1, 3)]
summary(df)


# 연속형 변수에 대해서 정규화를 진행한 다음 변수 탐색 -> 이후 상관관계 탐색
df$symboling <- normailzation_minmax(df$symboling)
df$wheelbase <- normailzation_minmax(df$wheelbase)
df$carlength <- normailzation_minmax(df$carlength)
df$carwidth <- normailzation_minmax(df$carwidth)
df$carheight <- normailzation_minmax(df$carheight)
df$enginesize <- normailzation_minmax(df$enginesize)
df$boreratio <- normailzation_minmax(df$boreratio)
df$stroke <- normailzation_minmax(df$stroke)
df$compressionratio <- normailzation_minmax(df$compressionratio)
df$horespower <- normailzation_minmax(df$horsepower)
df$peakrpm <- normailzation_minmax(df$peakrpm)
df$citympg <- normailzation_minmax(df$citympg)
df$highwaympg <- normailzation_minmax(df$highwaympg)
head(df)

# 범주형 변수 탐색
summary(df)

#각 변수의 카디널리티가 크지 않으므로 one-hot encoding 진행
df2 <- df
dummy <- dummyVars(" ~ .", data=df2)
df2 <- data.frame(predict(dummy,newdata=df2))
summary(df2)
head(df2)

# symboling 변수 탐색
gg<-qplot(symboling, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of symboling')
gg

car.symboling <- df$symboling
car.price <- df$price
subset <- cbind(df$symboling, df$price)
plot(car.price ~ car.symboling, data = subset, main ="Scatter plot")
# 제거해도 될듯

# wheelbase 변수 탐색
gg<-qplot(wheelbase, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of wheelbase')
gg
# wheelbase : 차량 앞바퀴 차축과 뒷바퀴 차축간의 거리.
# 길수록 회전반경이 커져서 직진성,주행,안정성,승차감이 좋아지지만 운전하기가 어렵다

# carlength 변수 탐색
gg<-qplot(carlength, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of carlength')
gg

# carwidth 변수 탐색
gg<-qplot(carwidth, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of carwidth')
gg

# carheight 변수 탐색
gg<-qplot(carheight, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of carheight')
gg

# 차량의 크기는 length, width, hegith인데 셋은 강한 상관관계가 존재할것으로 예상됨.
car.length <- df$carlength
car.width <- df$carwidth
car.height <- df$carheight
car.price <- df$price

# 우선 length와 Y값(pirce)와의 관계성 탐색
subset <- cbind(df$carlength, df$price)
plot(car.price ~ car.length, data = subset, main ="Scatter plot")
# ㄴ 어느정도 상관성이 존재해 보인다.

# length와 width 관계성 탐색
subset <- cbind(df$carlength, df$carwidth)
plot(car.width ~ car.length, data = subset, main ="Scatter plot")

# length와 height 관계성 탐색
subset <- cbind(df$carlength, df$carheight)
plot(car.length ~ car.height, data = subset, main ="Scatter plot")

# height와 price 관계성 탐색
subset <- cbind(df$carheight, df$price)
plot(car.price ~ car.height, data = subset, main ="Scatter plot")

# 따라서 length는 price에 대한 설명력이 있는것으로 관측됨
# 이때 length 와 width는 관계성이 관측되었으므로 다중공선성 문제가 예상됨.
# height는 price에 대한 설명력이 없는것으로 관측됨.


# curbweight 변수 탐색
gg<-qplot(curbweight, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of curbweight')
gg

# enginesize 변수 탐색
gg<-qplot(enginesize, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of enginesize')
gg

# boreratio 변수 탐색
gg<-qplot(boreratio, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of boreratio')
gg
# 실린더 행정 안지름비. 1.0보다 클때 장행정 엔진, 작을때 단행정 엔진이이라 한다.

# stroke 변수 탐색
gg<-qplot(stroke, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of stroke')
gg

# compressionratio 변수 탐색
gg<-qplot(compressionratio, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of compressionratio')
gg
# 압축비 압축비가 높을 시 열 효율이 증가하지만 
# 너무 증가했을 때는 오히려 열 효율이 떨어질 수 있음. (노킹현상)
# 디젤의 경우 노킹현상이 일어나지 않아 압축비가 큰것도 존재함.

# horsepower 변수 탐색
gg<-qplot(horsepower, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of horsepower')
gg

# peakrpm 변수 탐색
gg<-qplot(peakrpm, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of peakrpm')
gg

# citympg 변수 탐색 
gg<-qplot(citympg, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of citympg')
gg

# highwaympg 변수 탐색
gg<-qplot(highwaympg, data=df, geom="histogram")+
  theme_bw()+ggtitle('Histogram of highwaympg')
gg

# 학습모델 구축
multi_model <- lm(price ~ ., data = df2)
summary(multi_model)

# MSE 측정
num <- fitted(multi_model)
mean(((df$price-num)**2))

# MAE 측정
library(forecast)
accuracy(multi_model)

# p-value 관측 & 데이터 탐색시 의미없는 변수 제거 & 다중공선성 문제 발생 야기 변수 제거
# symboling, carlength, carheight, fuelsystem 제거
summary(df)
df3 <- df
df3 <- df3[,-c(1,9,11,16)]
summary(df3)

# 모델 구축
multi_model2 <- lm(price ~ ., data = df3)
summary(multi_model2)

# 두 모델 비교
anova(multi_model,multi_model2)

# Stepwise selcetion 방법 사용
model_step <- step(multi_model, direction = "backward")
summary(model_step)
