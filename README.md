# Modelos GLM
Este é um resumo básico de utilização de modelos GLM/GLMM, criado para referência e uso pessoal. Essas referências e seus pacotes são exclusivos para uso em Python.

## Sobre os pacotes
Em GLM trabalha-se com o pacote `statsmodels`. Com ele sozinho é possível realizar todos os modelos GLM/GLMM estudados ao longo do MBA em Data Science & Analytics da USP (da turma 241 ao menos).

Se o pacote não está instalado na máquina, simplesmente se comanda no terminal a instrução abaixo
> pip3 install statsmodels

Dentro dos scripts (e exclusivamente em relação à modelagem) são necessárias duas importações básicas, descritas logo abaixo. A função `sm` é utilizada dentro da função `smf`, para definir a família da distribuição que será utilizada de acordo com o problema de pesquisa.
```
import statsmodels.api as sm
import statsmodels.formula.api as smf
```

Para cálculos adicionais, caso necessário, como por exemplo estatística F ou estatística T, é utilizado o pacote `scipy` e que deve ser importado no início do script.

- Para a estatística F de Fischer (ou Snedecor):
```
from scipy.stats import f
...
f.ppf(0.95, df_modelo, df_residuos)
```
As mensurações de df_modelo e df_residuos são dadas no summary em `Df Model` e `Df Residuals`, respectivamente. Podendo ser obtidas com os códigos `modelo.df_model` e `modelo.df_residuals`, respectivamente.

- Para a estatística T de Student:
```
from scipy.stats import t
```
Se a regressão for uma regressão simples (apenas um parâmetro), então `t` é calculado simplesmente como a raiz quadrada da estatística F, com o auxílio do `numpy`:
```
t = np.sqrt(f)
```


A documentação básica pode ser encontrada no site do próprio pacote clicando [aqui](https://www.statsmodels.org/stable/glm.html).

## Resumo sobre os modelos e suas distribuições
|_**Modelo de Regressão**_      |_**Características da Variável Dependente (ou o target)**_     |_**Distribuição**_|
|-------------------------------|---------------------------------------------------------------|------------------|
|**Linear**                     |Quantitativa                                                   |_Normal_|
|**Com transformação Box-Cox**  |Quantitativa                                                   |_Normal após a transformação_|
|**Logística Binária**          |Qualitativa com **apenas duas categorias**                     |_Bernoulli_|
|**Logística Multinomial**      |Qualitativa com 3+ categorias                                  |_Binomial_|
|**Poisson / Zero-Inflated Poisson**                    |Quantitativa com valores inteiros e não-negativos (contagem)   |_Poisson_|
|**Binomial Negativo / Zero-Inflated Negative Binomial Poisson**          |Quantitativa com valores inteiros e não-negativos (contagem)   |_Poisson-Gama_|

## Modelos de Regressão
TO-DO: faz link de âncoras nos modelos acima com as explicações abaixo

### 1. Modelo Linear
Este é o modelo mais simples de todos, também chamado de OLS/MQO (ordinary least squares/mínimos quadrados ordinários). É um modelo que funciona muito bem para predição na interpolação dos dados, ou seja, dentro das dimensões mínimas e máximas dos dados em uso. Como característica, queremos que a soma dos erros de cada observação seja igual (ou muito próxima) a zero e a soma dos erros ao quadrado seja a mínima possível.

#### Formulação algébrica e no Python
y = alfa + b*x<sub>1</sub> + b*x<sub>2</sub> + ... + b*x<sub>n</sub>
```
sm.OLS.from_formula().fit()
```

#### Verificações para o modelo
Após rodar o modelo temos os resultados com `modelo.summary()`.

##### Estatística F (para validade ou não do modelo)
O p-value da estatística dado em `Prob (F-statistic)` no summary.
  - Se F<sub>calculado</sub> >= F<sub>crítico</sub>, então p-value <= 0,05, logo pelo menos um beta é diferente de zero e estatisticamente significante;
  - Se F<sub>calculado</sub> < F<sub>crítico</sub>, então p-value > 0,05, logo b<sub>1</sub> = b<sub>2</sub> = b<sub>3</sub> = ... = b<sub>n</sub> = 0, logo nenhum beta é estatisticamente significante e o modelo cai por terra, não podendo ser utilizada para fins preditivos.

O valor dessa estatística pode ser calculado por ABCDE.

##### Estatística T (para validade ou não dos parâmetros)
O p-value de cada variável é dado no summary (coluna p-value nas descrições dos parâmetros).
  - H<sub>0</sub>: p-value > 0,05, significando que o parâmetro **NÃO é estatisticamente significante**;
  - H<sub>1</sub>: p-value <= 0,05, significando que o parâmetro **É estatisticamente significante**;


#### Informações adicionais
Podem-se ser obtidos tanto o somatório dos quadrados do modelo ((y_observado - y_médio)<sup>2</sup> `modelo.ess`) como o somatório dos erros ao quadrado (`modelo.ssr`), que são utilizados para se calcular a estatística F do modelo.

É importante ressaltar que um modelo sobre um banco de dados horizontalizado (num features > num observações) não consegue explicar apropriadamente o comportamento desse banco de dados em função dessa discrepância.

O parâmetro alpha **sempre** se mantém na equação. Se porventura esse parâmetro der estatisticamente não significante é sinal apenas de pouca quantidade de observações no banco de dados. Logo, em se aumentando a quantidade de informações o parâmetro passa a ser significativo.

### 2. Modelo Linear c/ Transformação Box-Cox
A transformação se dá assim e assado, a regressão utiliza a mesma formulação do Modelo Linear


### 3. Modelo Logístico Binário
texto
```
sm.LOGIT.from_formula().fit()
```

### 4. Modelo Logístico Multinomial
texto
```
MNLogit()
sm.discrete.discrete_model().fit()
```

### 5. Modelo Poisson
texto
```
smf.glm(..., family=sm.families.Poisson()).fit()
sm.Poisson.from_formula().fit()
```

### 6. Modelo Binomial Negativo
texto
```
smf.glm(..., family=sm.families.NegativeBinomial(alpha=N).fit()
sm.NegativeBinomial.from_formula().fit()
```

### 7. Modelo Binomial Zero-Inflated Poisson
texto
```
sm.ZeroInflatedPoisson().fit()
```

### 8. Modelo Binomial Negativo Zero-Inflated Poisson
texto
```
sm.ZeroInflatedNegativeBinomialP().fit()
```

### Propriedades comuns aos modelos GLM/GLMM
Os modelos desenvolvidos a partir de GLM/GLMM possuem atributos que são disponibilizados de forma agrupada através da utilização de `.summary()` e que podem ser resgatados individualmente através de códigos no script. Alguns exemplos seguem abaixo.
- Parãmetros (alpha e betas) do modelo: `modelo.params`
- Estatística T dos parâmetros: `modelo.tvalues`
- p-values dos parâmetros: `modelo.pvalues`
- R<sup>2</sup> do modelo: `modelo.rsquared`
**Nota:** os parâmetros, as estatísticas e os p-values podem ser visualizados individualmente com utilização de `.iloc[]`

### Facilitador para as fórmulas
Código para o script:
```
lista_colunas = list(df.drop(['dropa aqui as não-desejáveis e/ou a target'], axis=1)),
formula_modelo = ' + '.join(lista_colunas)
formula_modelo = "[target aqui] ~ " + formula__modelo
print("A forma funcional a ser utilizada é como segue abaixo:\n\n", formula_modelo)
```
