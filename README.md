# Modelos GLM
Este é um resumo básico de utilização de modelos GLM/GLMM, criado para referência e uso pessoal. Essas referências e seus pacotes são exclusivos para uso em Python.

## Sobre os pacotes
Em GLM trabalha-se com o pacote `statsmodels`. Com ele sozinho é possível realizar todos os modelos GLM/GLMM estudados ao longo do MBA em Data Science & Analytics da USP (da turma 241 ao menos).

Se o pacote não está instalado na máquina, simplesmente se comanda no terminal a instrução abaixo
> pip3 install statsmodels

Dentro dos scripts são necessárias duas importações básicas, descritas logo abaixo. A função `sm` é utilizada dentro da função `smf`, para definir a família da distribuição que será utilizada de acordo com o problema de pesquisa.

```
import statsmodels.api as sm
import statsmodels.formula.api as smf
```

## Resumo sobre os modelos e suas distribuições
|_**Modelo de Regressão**_      |_**Características da Variável Dependente (ou o target)**_     |_**Distribuição**_|
|-------------------------------|---------------------------------------------------------------|------------------|
|**Linear**                     |Quantitativa                                                   |_Normal_|
|**Com transformação Box-Cox**  |Quantitativa                                                   |_Normal após a transformação_|
|**Logística Binária**          |Qualitativa com **apenas duas categorias**                     |_Bernoulli_|
|**Logística Multinomial**      |Qualitativa com 3+ categorias                                  |_Binomial_|
|**Poisson/Zero-Inflated Poisson**                    |Quantitativa com valores inteiros e não-negativos (contagem)   |_Poisson_|
|**Binomial Negativo/Zero-Inflated Negative Binomial Poisson**          |Quantitativa com valores inteiros e não-negativos (contagem)   |_Poisson-Gama_|

## Modelos de Regressão
TO-DO: faz link de âncoras nos modelos acima com as explicações abaixo

### 1. Modelo Linear
Este é o modelo mais simples de todos, normalmente se utiliza OLS/MQO (ordinary least squares/mínimos quadrados ordinários).

#### Formulação
y = alfa + b*x<sub>1</sub> + b*x<sub>2</sub> + ... + b*x<sub>n</sub>
```
sm.OLS.from_formula()
```

#### Verificações para o modelo
Após rodar o modelo temos os resultados (_modelo.summary()_). Os pontos importantes são:
- p-value da variável: utiliza-se estatística T de Student




### 2. Modelo Linear c/ Transformação Box-Cox
A transformação se dá assim e assado, a regressão utiliza a mesma formulação do Modelo Linear


### 3. Modelo Logístico Binário
texto
```
sm.LOGIT.from_formula()
```

### 4. Modelo Logístico Multinomial
texto
```
MNLogit()
sm.discrete.discrete_model()
```

### 5. Modelo Poisson
texto
```
smf.glm(..., family=sm.families.Poisson())
sm.Poisson.from_formula()
```

### 6. Modelo Binomial Negativo
texto
```
smf.glm(..., family=sm.families.NegativeBinomial(alpha=N)
sm.NegativeBinomial.from_formula()
```

### 7. Modelo Binomial Negativo Zero-Inflated Poisson
texto
```
sm.ZeroInflatedPoisson()
```

### 8. Modelo Binomial Negativo
texto
```
sm.ZeroInflatedNegativeBinomialP()
```


### Facilitador para as fórmulas
Código para o script:
```
lista_colunas = list(df.drop(['dropa aqui as não-desejáveis e/ou a target'], axis=1)),
formula_modelo = ' + '.join(lista_colunas)
formula_modelo = "[target aqui] ~ " + formula__modelo
print("A forma funcional a ser utilizada é como segue abaixo:\n\n", formula_modelo)
```
