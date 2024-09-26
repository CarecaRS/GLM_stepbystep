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

Consideração importante: nesse documento explicativo sempre se utiliza intervalo de confiança de 95%. Caso seja necessária alteração na prática, prestar atenção nisso.
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
  - H<sub>0</sub>: se F<sub>calculado</sub> < F<sub>crítico</sub>, então p-value > 0,05, logo b<sub>1</sub> = b<sub>2</sub> = b<sub>3</sub> = ... = b<sub>n</sub> = 0, logo nenhum beta é estatisticamente significante e o modelo cai por terra, não podendo ser utilizada para fins preditivos.
  - H<sub>1</sub>: se F<sub>calculado</sub> >= F<sub>crítico</sub>, então p-value <= 0,05, logo pelo menos um beta é diferente de zero e estatisticamente significante;

O valor dessa estatística pode ser calculado conforme disposto abaixo:
```
from scipy.stats import f
...
f.ppf(0.95, df_modelo, df_residuos)
```
As mensurações de df_modelo e df_residuos são dadas no summary em `Df Model` e `Df Residuals`, respectivamente. Podendo ser obtidas com os códigos `modelo.df_model` e `modelo.df_residuals`, respectivamente.

##### Estatística T (para validade ou não dos parâmetros)
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
```
tt = (sm-m)/np.sqrt(sv/float(n))  # t-statistic for mean
pval = stats.t.sf(np.abs(tt), n-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
'######

Sobre a comprovação do diagnóstico do parâmetro não ser estatisticamente significante quando em uma regressão múltipla, se na forma funcional geral ele já apresenta um p-value acima de 0,05 e se deseja comprovar essa descartabilidade do parâmetro, pode ser realizada uma regressão simples entre o parâmetro em questão e o target, a estatística t novamente vai indicar a não-rejeição de H<sub>0</sub>.

##### Teste de verificação de aderência dos resíduos à normalidade (Shapiro-Francia)
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

##### Transformação de Box-Cox
```
from scipy.stats import boxcox
...
y_chapeu, lbda = boxcox(df[target])
```
A execução da função acima retorna dois objetos: `y_chapeu` sendo os valores já transformados/normalizados da variável dependente (ou do target) e `lbda`, que é o lambda estabelecido para a transformação realizada (importante mencionar no desenvolvimento da solução do problema).

#### Informações adicionais
É importante ressaltar que um modelo sobre um banco de dados horizontalizado (num features > num observações) não consegue explicar apropriadamente o comportamento desse banco de dados em função dessa discrepância.

O parâmetro alpha **sempre** se mantém na equação. Se porventura esse parâmetro der estatisticamente não significante é sinal apenas de pouca quantidade de observações no banco de dados. Logo, em se aumentando a quantidade de informações o parâmetro passa a ser significativo.

Para comparações de modelos OLS se utiliza o R<sup>2</sup><sub>adjusted</sub>, que leva em consideração as dimensões de cada modelo.

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
- R<sup>2</sup><sub>adjusted</sub> do modelo: `modelo.rsquared_adj`
- Resíduos do modelo: `modelo.resid`
**Nota:** os valores pertinentes a cada observação, se aplicável, podem ser visualizados individualmente com utilização de `.iloc[]`

### Facilitador para as fórmulas
Código para o script:
```
lista_colunas = list(df.drop(['dropa aqui as não-desejáveis e/ou a target'], axis=1)),
formula_modelo = ' + '.join(lista_colunas)
formula_modelo = "[target aqui] ~ " + formula__modelo
print("A forma funcional a ser utilizada é como segue abaixo:\n\n", formula_modelo)
```
