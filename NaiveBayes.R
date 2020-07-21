library(e1071)

arquivo = read.csv(file.choose(), sep = ";", header = T)

amostra = sample(2, 1000, replace = T, prob =c(0.7,0.3))

creditotreino = arquivo[amostra== 1,]
creditoteste = arquivo[amostra ==2,]

modelo = naiveBayes(as.factor(CLASSE) ~ ., creditotreino )

predicao = predict(modelo, creditoteste, ) 

confusao = table(creditoteste$CLASSE, predicao)

taxaacerto = (confusao[1]+confusao[4]) /  sum(confusao)
taxaerro = (confusao[2] + confusao[3]) / sum(confusao)

