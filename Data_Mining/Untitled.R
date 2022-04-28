# 필요한 라이브러리 설치
library(dplyr)
library(sqldf)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(psych)
library(dummies)

# 로지스틱 회귀 데이터 로드 및 탐색  
data <- read.csv("input/train.csv")
str(data)
head(data)
summary(data)
# 변수 자료형이 chr일 경우 summary 시 정보를 제대로 보여주지 않아 factor 변수로 처리
# 결측치 공백'' 은 'Unknown'문자열로 처리 
data$workclass <-ifelse(data$workclass=='','Unknown',as.character(data$workclass))
data$workclass <- as.factor(data$workclass)
summary(data)

data$education <- as.factor(data$education)
summary(data)

data$marital.status <- as.factor(data$marital.status)

data$occupation <-ifelse(data$occupation =='','Unknown',as.character(data$occupation))
data$occupation <- as.factor(data$occupation)

data$relationship <- as.factor(data$relationship)
data$race<- as.factor(data$race)
data$sex <- as.factor(data$sex)

data$native.country <-ifelse(data$native.country =='','Unknown',as.character(data$native.country))
data$native.country <- as.factor(data$native.country)

summary(data)

# -------------이하 결측치 처리 및 데이터 형변환---------------

names(data)
is.na(data)
colSums(is.na(data))

# 상관계수 찍어주는 거
pairs.panels(data)


# 더미는 하나의 변수에 대해서만 encoding진행된 테이블을 반환하므로
# ordinal encoding 진행
#data2 <- dummy.data.frame(data, names=c("workclass"))
#data2 <- dummy.data.frame(data, names=c("education"))
#data2 <- dummy.data.frame(data, names=c("workclass"))

# 범주형 변수에 대해 
# nominal 데이터 -> label encoding
# ordinal 데이터 -> ordinal encoding 진행

# label encoding을 위해 데이터를 벡터로, 벡터로 숫자로 바꿔주는 연산을 진행한다.
workclass <- factor(data, level = c())

encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}
