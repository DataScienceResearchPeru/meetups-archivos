
rm(list=ls())
##########################################################################
### DESARROLLO E IMPLEMENTACION DE ALGORITMOS DE ML -- ###################
##########################################################################
######## Autores: Andre Chavez  ########################################## 
##########################################################################

# ######### 1) LIBRERIAS A UTILIZAR ################# 

library(sqldf)
library(ggvis)
library(party)
library(Boruta)
library(pROC)
library(randomForest)
library(e1071)
library(caret)
library(glmnet)
library(mboost)
library(adabag)
library(xgboost)
library(ROCR)
library(C50)
library(mlr)
library(lattice)
library(gmodels)
library(gplots)
library(DMwR)
library(UBL)
library(rminer)
library(polycor)
library(class)
library(neuralnet)
library(reticulate)


######### 2) EXTRAYENDO LA DATA ################# 

train<-read.csv("train.csv",na.strings = c(""," ",NA)) # leer la data de entrenamiento

names(train) # visualizar los nombres de la data
head(train)  # visualizar los 6 primeros registros
str(train)   # ver la estructura de la data

######### 3) EXPLORACION DE LA DATA ################# 

# tablas resumen
summary(train) # tabla comun de obtener
summarizeColumns(train) # tabla mas completa

resumen=data.frame(summarizeColumns(train))


######### 4) IMPUTACION DE LA DATA ################# 

# revisar valores perdidos

perdidos=data.frame(resumen$name,resumen$na,resumen$type); colnames(perdidos)=c("name","na","type")
perdidos

# recodificando Dependents
train$Dependents=ifelse(train$Dependents=="3+",3,
                        ifelse(train$Dependents=="0",0,
                               ifelse(train$Dependents=="1",1,
                                      ifelse(train$Dependents=="2",2,
                                             train$Dependents))))
train$Dependents=as.factor(train$Dependents)

# convirtiendo en factor Credit_History
train$Credit_History <- as.factor(train$Credit_History)

# recodificando Loan_Status
train$Loan_Status=ifelse(train$Loan_Status=="N",0,1)
train$Loan_Status=as.factor(train$Loan_Status)

# partcionando la data en numericos y factores

numericos <- sapply(train, is.numeric) # variables cuantitativas
factores <- sapply(train, is.factor)  # variables cualitativas

train_numericos <-  train[ , numericos]
train_factores <- train[ , factores]

# APLICAR LA FUNCION LAPPLY PARA DISTINTAS COLUMNAS CONVERTIR A FORMATO NUMERICO
n1=min(dim(train_factores))
train_factores[2:(n1-1)] <- lapply(train_factores[2:(n1-1)], as.numeric)
train_factores[2:(n1-1)] <- lapply(train_factores[2:(n1-1)], as.factor)

# Para train 

train=cbind(train_numericos,train_factores[,-1])

## Imputacion Parametrica

#Podemos imputar los valores perdidos por la media o la moda

# data train
train_parametrica <- impute(train, classes = list(factor = imputeMode(), 
                                                  integer = imputeMode(),
                                                  numeric = imputeMean()),
                            dummy.classes = c("integer","factor"), dummy.type = "numeric")
train_parametrica=train_parametrica$data[,1:min(dim(train))]

summary(train_parametrica)

#### -- 2) Modelo de Regresion Logistica

## supuestos

cor(train_parametrica[,1:4])

## Particion Muestral

set.seed(123)
training.samples <- train_parametrica$Loan_Status %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- train_parametrica[training.samples, ]
test.data <- train_parametrica[-training.samples, ]

## Modelado

modelo_logistica=glm(Loan_Status~.,data=train.data,family="binomial" )
summary(modelo_logistica)


## indicadores

proba1=predict(modelo_logistica, newdata=test.data,type="response")
AUC1 <- roc(test.data$Loan_Status, proba1)
## calcular el AUC
auc_modelo1=AUC1$auc
## calcular el GINI
gini1 <- 2*(AUC1$auc) -1
# Calcular los valores predichos
PRED <-predict(modelo_logistica,test.data,type="response")
PRED=ifelse(PRED<=0.5,0,1)
PRED=as.factor(PRED)
# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,test.data$Loan_Status,positive = "1")
# sensibilidad
Sensitivity1=as.numeric(tabla$byClass[1])
# Precision
Accuracy1=tabla$overall[1]
# Calcular el error de mala clasificaci?n
error1=mean(PRED!=test.data$Loan_Status)

# indicadores
auc_modelo1
gini1
Accuracy1
error1
Sensitivity1


#### -- 2) Random Forest

set.seed(1234)
modelo2 <- randomForest( Loan_Status~.,data = train.data,   # Datos a entrenar 
                          ntree=80,           # N?mero de ?rboles
                          mtry = 3,            # Cantidad de variables
                          importance = TRUE,   # Determina la importancia de las variables
                          replace=T) 

##probabilidades
proba2<-predict(modelo2, newdata=test.data,type="prob")
proba2=proba2[,2]

# curva ROC
AUC2 <- roc(test.data$Loan_Status, proba2) 
auc_modelo2=AUC2$auc

# Gini
gini2 <- 2*(AUC2$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo2,test.data,type="class")

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,test.data$Loan_Status,positive = "1")

# sensibilidad
Sensitivity2=as.numeric(tabla$byClass[1])

# Precision
Accuracy2=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error2=mean(PRED!=test.data$Loan_Status)

# indicadores
auc_modelo2
gini2
Accuracy2
error2
Sensitivity2

## --Tabla De Resultados ####

AUC=rbind(auc_modelo1,
          auc_modelo2
)
GINI=rbind(gini1,
           gini2
)
Accuracy=rbind(Accuracy1,
               Accuracy2
)

ERROR= rbind(error1,
             error2
)
SENSIBILIDAD=rbind(Sensitivity1,
                   Sensitivity2
)

resultado=data.frame(AUC,GINI,Accuracy,ERROR,SENSIBILIDAD)
rownames(resultado)=c('Logistico',
                      'RF'
)
resultado=round(resultado,2)
resultado

## Resultado Ordenado #####

# ordenamos por el Indicador que deseamos, quiza Accuracy en forma decreciente
Resultado_ordenado <- resultado[order(-Accuracy),] 
Resultado_ordenado

