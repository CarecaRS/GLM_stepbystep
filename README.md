# Modelos GLM
Este é um resumo básico de utilização de modelos GLM/GLMM, criado para referência e uso pessoal. Essas referências e seus pacotes são exclusivos para uso em Python.

## Sobre os pacotes
Em GLM trabalha-se com o pacote `statsmodels`. Com ele sozinho é possível realizar todos os modelos GLM/GLMM estudados ao longo do MBA em Data Science & Analytics da USP (da turma 241 ao menos).

Alguns outros pacotes estatísticos também são utilizados para os diversos diagnósticos a serem feitos de acordo com cada modelo estimado, como por exemplo o pacote `scipy` e o pacote `statstests`.

Caso algum pacote não esteja instalado na máquina, simplesmente se comanda no terminal instrução instrução semelhante aos exemplos abaixo
> pip3 install statsmodels

> pip3 install scipy

> pip3 install statstests

Dentro dos scripts (e exclusivamente em relação à modelagem) são necessárias duas importações básicas, descritas logo abaixo. A função `sm` é utilizada dentro da função `smf`, para definir a família da distribuição que será utilizada de acordo com o problema de pesquisa.
```
import statsmodels.api as sm
import statsmodels.formula.api as smf
```

A documentação básica pode ser encontrada no site do próprio pacote clicando [aqui](https://www.statsmodels.org/stable/glm.html).

## Resumo sobre os modelos e suas distribuições
|_**Modelo de Regressão**_      |_**Características da Variável Dependente (ou o target)**_     |_**Distribuição**_|
|-------------------------------|---------------------------------------------------------------|------------------|
|[**Linear**](#1-modelo-linear)                     |Quantitativa                                                   |_Normal_|
|[**Com transformação Box-Cox**](#2-modelo-linear-com-transformação-box-cox)  |Quantitativa                                                   |_Normal após a transformação_|
|[**Logística Binária**](#3-modelo-logístico-binário)          |Qualitativa com **apenas duas categorias**                     |_Bernoulli_|
|[**Logística Multinomial**](#4-modelo-logístico-multinomial)      |Qualitativa com 3+ categorias                                  |_Binomial_|
|[**Poisson**](#5-modelo-poisson) / [**Zero-Inflated Poisson**](#7-modelo-binomial-zero-inflated-poisson)                    |Quantitativa com valores inteiros e não-negativos (contagem)   |_Poisson_|
|[**Binomial Negativo**](#6-modelo-binomial-negativo) / [**Zero-Inflated Negative Binomial Poisson**](#8-modelo-binomial-negativo-zero-inflated-poisson)          |Quantitativa com valores inteiros e não-negativos (contagem)   |_Poisson-Gama_|

# Modelos de Regressão
TO-DO: faz link de âncoras nos modelos acima com as explicações abaixo

## 1. Modelo Linear
Este é o modelo mais simples de todos, também chamado de OLS/MQO (ordinary least squares/mínimos quadrados ordinários). É um modelo que funciona muito bem para predição na interpolação dos dados, ou seja, dentro das dimensões mínimas e máximas dos dados em uso. Como característica, queremos que a soma dos erros de cada observação seja igual (ou muito próxima) a zero e a soma dos erros ao quadrado seja a mínima possível.

### Formulação algébrica e no Python
y = alfa + b*x<sub>1</sub> + b*x<sub>2</sub> + ... + b*x<sub>n</sub>
```
sm.OLS.from_formula().fit()
```

### Verificações para o modelo
Após rodar o modelo temos os resultados com `modelo.summary()`.

#### Estatística F (para validade ou não do modelo)
O p-value da estatística dado em `Prob (F-statistic)` no summary.
  - H<sub>0</sub>: se F<sub>calculado</sub> < F<sub>crítico</sub>, então p-value > 0,05, logo b<sub>1</sub> = b<sub>2</sub> = b<sub>3</sub> = ... = b<sub>n</sub> = 0. Deste modo, nenhum beta é estatisticamente significante e o modelo cai por terra, não podendo ser utilizada para fins preditivos.
  - H<sub>1</sub>: se F<sub>calculado</sub> >= F<sub>crítico</sub>, então p-value <= 0,05, logo pelo menos um beta é diferente de zero e estatisticamente significante;

O valor dessa estatística pode ser calculado conforme disposto abaixo:
```
from scipy.stats import f
...
f.ppf(0.95, df_modelo, df_residuos)
```
As mensurações de df_modelo e df_residuos são dadas no summary em `Df Model` e `Df Residuals`, respectivamente. Podendo ser obtidas com os códigos `modelo.df_model` e `modelo.df_residuals`, respectivamente.

#### Estatística T (para validade ou não dos parâmetros)
O p-value de cada variável é dado no summary (coluna p-value nas descrições dos parâmetros).
  - H<sub>0</sub>: p-value > 0,05, significando que o parâmetro **NÃO é estatisticamente significante**;
  - H<sub>1</sub>: p-value <= 0,05, significando que o parâmetro **É estatisticamente significante**;

O valor dessa estatística pode ser calculado conform segue:
```
from scipy.stats import t
```
Se a regressão for uma regressão simples (apenas um parâmetro), então `t` é calculado simplesmente como a raiz quadrada da estatística F, com o auxílio do `numpy`:
```
t = np.sqrt(f)
```
Para cálculo das estatísticas T de cada um dos parâmetros precisa-se saber os graus de liberdade da amostra e também o score T da variável em questão.

TODO: verificar essa seção abaixo

'######

O cálculo do score T se dá por
```
t_score = (df[param].mean() - U) / (df[param].sd() / np.sqrt(len(df)))

tt = (sm-m)/np.sqrt(sv/float(n))  # t-statistic for mean
pval = stats.t.sf(np.abs(tt), n-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
```
'######

Sobre a comprovação do diagnóstico do parâmetro não ser estatisticamente significante quando de uma regressão múltipla, se na forma funcional geral ele já apresenta um p-value acima de 0,05 e se deseja comprovar essa descartabilidade do parâmetro, pode ser realizada uma regressão simples entre o parâmetro em questão e o target, a estatística t novamente vai indicar a não-rejeição de H<sub>0</sub>.

Se porventura uma regressão simples entre o parâmetro e o target indicar a rejeição de H<sub>0</sub> (ou seja, parâmetro sim estatísticamente significante) a conclusão é que o parâmetro em questão não é um bom preditor da variável target _na presença das demais variáveis_, o que pode ser confirmado através de uma matriz de correlação (é um caso de presença de multicolinearidade - alta correlação entre duas ou mais variáveis).

Para o estabelecimento final dos parâmetros que permanecem na forma funcional final da solução do problema proposto pode ser utilizada a função `stepwise`, disponível no pacote `statstests`:
```
from statstests.process import stepwise
...
stepwise(modelo, pvalue_limit=0.05)
```

#### Teste de verificação de aderência dos resíduos à normalidade (Shapiro-Francia)
Necessária a importação do pacote `statstests`:
```
from statstests.tests import shapiro_francia
...
shapiro_francia(modelo.resid).items()  # .items() utilizado apenas por questões estéticas
```

  - H<sub>0</sub>: p-value > 0,05, significando que a distribuição dos resíduos **NÃO é estatisticamente diferente** de uma distribução normal;
  - H<sub>1</sub>: p-value <= 0,05, significando que a distribuição dos resíduos **É estatisticamente diferente** (distribuição não aderente à normalidade, sendo necessária utilização de transformação Box-Cox _na variável dependente ou target_ - NORMALIZAÇÃO);

**Particularidade**: se o número de observações for inferior a 30 pode ser utilizado o teste de Shapiro-Wilk:
```
from scipy.stats import shapiro
...
shapiro(modelo.resid)
```

#### Diagnóstico de multicolinearidade
Multicolinearidae pode se dar em função de alguns pontos mais comuns:
- Existência de variáevis que apresentam a mesma tendência durante alguns períodos, em decorrência da seleção de uma amostra que inclua apenas observaçẽos referentes a estes períodos específicos;
- Utilização de amostras com reduzido número de observações;
- Utilização de valores defasados em algumas das variáveis explicativas como 'novas explicativas'

Alguma noção de multicolinearidade pode ser obtida através da matriz de correlação entre as variáveis. Dois parâmetros são utilizados para poder se diagnosticar indícios de multicolinearidade:
- Tolerance: `1 - R<sup>2</sup><sub>p</sub>` (onde `<sub>p</sub>` refere-se a cada uma das variáveis existentes no modelo - se 500 variáveis, então 500 valores de Tolerance)
- VIF: `1 / Tolerance` (Variance Inflation Factor)

Para um cálculo eficiente da VIF utiliza-se o pacote `statsmodels`:
```
from statsmodels.stats.outliers_influence import variance_inflation_factor
...
variance_inflation_factor(modelo.values, modelo.columns)  # função apenas demonstração, precisa ajuste de código
```

Temos então como diagnóstico:
- VIF baixo: indicador de ausência de multicolinearidade
- VIF alto: indicador de presença de multicolinearidade
Isso pode gerar a dúvida "e o que é VIF alto ou baixo?". Alguns autores utilizam como referência valores de 5 a 10, mas não existe concordância geral e cada caso é um caso.

Também pode ser indicativo de multicolinearidade a existência de sinais inesperados dos coeficientes dos parâmetros e testes t não significantes com teste F significante.

#### Diagnóstico de heterocedasticidade
Concetualmente, a existência de heterocedasticidade indica a omissão de feature/variável relevente para a explicação do comportamento da variável target.

Graficamente, a dispersão dos dados na relação target e features lembra a figura de um cone, que pode ser aberto tanto para cima como para baixo.

Após a modelagem, pode ser plotado o resultado dos resíduos e dos fitted values, também lembrando a figura de um cone como indicativo de heterocedasticidade (independente da adesão dos resíduos de erro serem aderentes à normalidade ou não).

Para cálculo de verificação da existência ou não de heterocedasticidade utiliza-se o teste de Breusch-Pagan, com função criada manualmente:
```
def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value
```
Em função do p_value obtido pelo teste de Breusch-Pagan, temos que
  - H<sub>0</sub>: p-value > 0,05, indica homocedasticidade (ausência de heterocedasticidade);
  - H<sub>1</sub>: p-value <= 0,05, indica presença de heterocedasticidade.


### Informações adicionais
É importante ressaltar que um modelo sobre um banco de dados horizontalizado (num features > num observações) não consegue explicar apropriadamente o comportamento desse banco de dados em função dessa discrepância.

O parâmetro alpha **sempre** se mantém na equação. Se porventura esse parâmetro der estatisticamente não significante é sinal apenas de pouca quantidade de observações no banco de dados. Logo, em se aumentando a quantidade de informações o parâmetro passa a ser significativo.

Para comparações de modelos OLS se utiliza o R<sup>2</sup><sub>adjusted</sub>, que leva em consideração as dimensões de cada modelo.

## 2. Modelo Linear com Transformação Box-Cox
A transformação se dá com a utilização do pacote `scipy` e na função descrita abaixo:
```
from scipy.stats import boxcox
...
y_chapeu, lbda = boxcox(df[target])
```
A execução da função acima retorna dois objetos: `y_chapeu` sendo os valores já transformados/normalizados da variável dependente (ou do target) e `lbda`, que é o lambda estabelecido para a transformação realizada (importante mencionar no desenvolvimento da solução do problema).

Utiliza-se `y_chapeu` como target da forma funcional do modelo em trabalho e roda-se novamente o modelo, refazendo todas as verificações pertinentes.

## 3. Modelo Logístico Binário
texto
```
sm.LOGIT.from_formula().fit()
```

## 4. Modelo Logístico Multinomial
texto
```
MNLogit()
sm.discrete.discrete_model().fit()
```

## 5. Modelo Poisson
texto
```
smf.glm(..., family=sm.families.Poisson()).fit()
sm.Poisson.from_formula().fit()
```

## 6. Modelo Binomial Negativo
texto
```
smf.glm(..., family=sm.families.NegativeBinomial(alpha=N).fit()
sm.NegativeBinomial.from_formula().fit()
```

## 7. Modelo Binomial Zero-Inflated Poisson
texto
```
sm.ZeroInflatedPoisson().fit()
```

## 8. Modelo Binomial Negativo Zero-Inflated Poisson
texto
```
sm.ZeroInflatedNegativeBinomialP().fit()
```

# Propriedades comuns aos modelos GLM/GLMM
Os modelos desenvolvidos a partir de GLM/GLMM possuem atributos que são disponibilizados de forma agrupada através da utilização de `.summary()` e que podem ser resgatados individualmente através de códigos no script. Alguns exemplos seguem abaixo.
- Fitted values (valores estimados) do modelo: `modelo.fittedvalues`
- Parãmetros (alpha e betas) do modelo: `modelo.params`
- Estatística T dos parâmetros: `modelo.tvalues`
- p-values dos parâmetros: `modelo.pvalues`
- R<sup>2</sup> do modelo: `modelo.rsquared`
- R<sup>2</sup><sub>adjusted</sub> do modelo: `modelo.rsquared_adj`
- Resíduos do modelo: `modelo.resid`
**Nota:** os valores pertinentes a cada observação, se aplicável, podem ser visualizados individualmente com utilização de `.iloc[]`

## Facilitador para as fórmulas
```
lista_colunas = list(df.drop(['dropa aqui as não-desejáveis e/ou a target'], axis=1)),
formula_modelo = ' + '.join(lista_colunas)
formula_modelo = "[target aqui] ~ " + formula__modelo
print("A forma funcional a ser utilizada é como segue abaixo:\n\n", formula_modelo)
```
