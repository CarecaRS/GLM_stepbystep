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
|[**Logística Binária**](#3-modelo-logístico-binário)          |Qualitativa com apenas duas categorias                     |_Bernoulli_|
|[**Logística Multinomial**](#4-modelo-logístico-multinomial)      |Qualitativa com 3+ categorias                                  |_Binomial_|
|[**Poisson**](#5-modelo-poisson) / [**Zero-Inflated Poisson**](#7-modelo-binomial-zero-inflated-poisson)                    |Quantitativa com valores inteiros e não-negativos (contagem)   |_Poisson_|
|[**Binomial Negativo**](#6-modelo-binomial-negativo) / [**Zero-Inflated Negative Binomial Poisson**](#8-modelo-binomial-negativo-zero-inflated-poisson)          |Quantitativa com valores inteiros e não-negativos (contagem)   |_Poisson-Gama_|

# Modelos de Regressão

## 1. Modelo Linear
Este é o modelo mais simples de todos, também chamado de OLS/MQO (ordinary least squares/mínimos quadrados ordinários). É um modelo que funciona muito bem para predição na interpolação dos dados, ou seja, dentro das dimensões mínimas e máximas dos dados em uso. Como característica, queremos que a soma dos erros de cada observação seja igual (ou muito próxima) a zero e a soma dos erros ao quadrado seja a mínima possível.

### Formulação algébrica e no Python
y = $a$ + $\beta$ x<sub>1</sub> + $\beta$ x<sub>2</sub> + ... + $\beta$ x<sub>n</sub>
```
sm.OLS.from_formula().fit()
```

### Verificações para o modelo
Após rodar o modelo temos os resultados com `modelo.summary()`.

#### Estatística F (para validade ou não do modelo)
O p-value da estatística dado em `Prob (F-statistic)` no summary.
- H<sub>0</sub>: se F<sub>calculado</sub> < F<sub>crítico</sub>, então p-value > 0,05, logo b<sub>1</sub> = b<sub>2</sub> = b<sub>3</sub> = ... = b<sub>n</sub> = 0. Deste modo, nenhum beta é estatisticamente significante e o modelo cai por terra, não podendo ser utilizada para fins preditivos;
- H<sub>1</sub>: se F<sub>calculado</sub> $\ge$ F<sub>crítico</sub>, então p-value $\le$ 0,05, logo pelo menos um beta é diferente de zero e estatisticamente significante.

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
  - H<sub>1</sub>: p-value $\le$ 0,05, significando que o parâmetro **É estatisticamente significante**;

O valor dessa estatística pode ser calculado conform segue:
```
from scipy.stats import t
```
Se a regressão for uma regressão simples (apenas um parâmetro), então `t` é calculado simplesmente como a raiz quadrada da estatística F, com o auxílio do `numpy`:
```
t = np.sqrt(f)
```
Para cálculo das estatísticas T de cada um dos parâmetros precisa-se saber os graus de liberdade da amostra e também o score T da variável em questão.

O cálculo do z-score (para utilização com a estatística T) se dá por:
```
from scipy.stats import zscore
...
zscore(df[param])
```
Também podendo ser calculado manualmente:
```
z_scores = (df[param] – df[param].mean()) / np.std(df[param])
```

Importante notar que este cálculo do z-score foi obtido nas entranhas da internet, a informação aqui contida pode estar equivocada ou incompleta.

Sobre a comprovação do diagnóstico do parâmetro não ser estatisticamente significante quando de uma regressão múltipla, se na forma funcional geral ele já apresenta um p-value acima de 0,05 e se deseja comprovar essa descartabilidade do parâmetro, pode ser realizada uma regressão simples entre o parâmetro em questão e o target, a estatística t novamente vai indicar a não-rejeição de H<sub>0</sub>.

Se porventura uma regressão simples entre o parâmetro e o target indicar a rejeição de H<sub>0</sub> (ou seja, parâmetro sim estatísticamente significante) a conclusão é que o parâmetro em questão não é um bom preditor da variável target _na presença das demais variáveis_, o que pode ser confirmado através de uma matriz de correlação (é um caso de presença de multicolinearidade - alta correlação entre duas ou mais variáveis).

#### Procedimento Stepwise
Para o estabelecimento final dos parâmetros que permanecem na forma funcional final da solução do problema proposto pode ser utilizada a função `stepwise`, disponível no pacote `statstests`:
```
from statstests.process import stepwise
...
stepwise(modelo, pvalue_limit=0.05)
```
Este procedimento é válido para qualquer modelo GLM que seja desenvolvido.

#### Teste de verificação de aderência dos resíduos à normalidade (Shapiro-Francia)
Necessária a importação do pacote `statstests`:
```
from statstests.tests import shapiro_francia
...
shapiro_francia(modelo.resid).items()  # .items() utilizado apenas por questões estéticas
```

  - H<sub>0</sub>: p-value > 0,05, significando que a distribuição dos resíduos **NÃO é estatisticamente diferente** de uma distribução normal;
  - H<sub>1</sub>: p-value $\le$ 0,05, significando que a distribuição dos resíduos **É estatisticamente diferente** (distribuição não aderente à normalidade, sendo necessária utilização de transformação Box-Cox _na variável dependente ou target_ - NORMALIZAÇÃO);

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
- Tolerance: 1 - R<sup>2</sup><sub>p</sub> (onde <sub>p</sub> refere-se a cada uma das variáveis existentes no modelo - se 500 variáveis, então 500 valores de Tolerance)
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
  - H<sub>1</sub>: p-value $\le$ 0,05, indica presença de heterocedasticidade.


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
Aqui variável target é dicotômica, o resultado será ou uma opção ou outra, tem ou não tem o evento. Por exemplo: no acidente do Titanic, a pessoa sobreviveu (evento) ou pereceu (não-evento); classificação de cogumelos entre não-comestíveis/venenosos (evento) ou comestíveis (não-evento); um arremesso de moedas vai resultar em cara (evento) ou coroa (não-evento). Como todo modelo logístico, retorna sempre uma probabilidade de ocorrer o evento ou o não-evento. A efetiva classificação de cada resultado de cada observação como evento ou não-evento se dá a partir de um _cutoff_, limite de corte que é definido pelo operador.

A distribuição de probabilidades de ocorrência de evento/não-evento da-se por:
- `p` sendo a probabilidade de ocorrer o evento; e
- `1 - p` sendo a probabilidade de ocorrer o não-evento.

'Chance' ou, em inglês, 'odds' é uma razão entre as duas probabilidades, que resulta em um número inteiro ou não ou em uma relação de proporções:
```
odds = evento / não-evento  # logo:
odds = p / (1 - p)
```
Se, por exemplo, p = 0,80, então temos que:
```
odds = p / (1 - p) = 0,80 / 0,20 = 4/1 (também pode ser escrito 4:1, lê-se 'chances de 4 para 1')
```
A estimação dos parâmetros é realizada através de um processo iterativo para maximizar o acerto da probabilidade de ocorrência de um evento à sua real ocorrência (Método de Máxima Verossimilhança). Em razão desse raciocínio, os resultados definidos para a variável target estarão sempre entre 0 (evento tende a não acontecer) e 1 (evento tende a acontecer com certeza). A análise de ajuste do modelo utiliza testes de significância dos parâmetros e vale-se da matriz de confusão dos resultados obtidos frente aos resultados reais.

### Formulação algébrica e no Python
De formulação semelhante à OLS, temos que a forma funcional da logística binária dá-se por:

ln(chance) = $a$ + $b$<sub>1</sub>x<sub>1<sub>i</sub></sub> + $b$<sub>2</sub>x<sub>2<sub>i</sub></sub> + ... + $b$<sub>n</sub>x<sub>n<sub>i</sub></sub>

Dentro da modelagem logística faz-se notar o conhecimento do constructo tido como logito (z), que é efetivamente toda a parte direita da igualdade da forma funcional acima:
- z = $a$ + $b$<sub>1</sub>x<sub>1<sub>i</sub></sub> + $b$<sub>2</sub>x<sub>2<sub>i</sub></sub> + ... + $b$<sub>n</sub>x<sub>n<sub>i</sub></sub>

Temos, por consequência, então:
- ln(p / (1 - p)) = z
- p / (1 - p) = e<sup>z</sup>
- p * (1 + e<sup>z</sup>) = e<sup>z</sup>

E, sendo assim, a probabilidade de ocorrência do evento se dá por: 
- p = e<sup>z</sup>/(1 + e<sup>z</sup>)
- p = 1 / (1 + e<sup>-z</sup>)      <sub># favor notar o expoente negativo</sub>
- p = 1 / (1 + e<sup>-($a$ + $b$<sub>1</sub>x<sub>1i</sub> + $b$<sub>2</sub>x<sub>2i</sub> + ... + $b$<sub>n</sub>x<sub>ni</sub>)</sup>)

No Python (ambos códigos resultam em respostas iguais):
```
modelo = smf.glm(formula='target ~ feature1 + feature2', dataset=df,
                 family=sm.families.Binomial()).fit()

modelo = sm.Logit.from_formula('target ~ feature1 + feature2', data=df).fit()  # mais simples, preferível
```
Da mesma forma que o modelo OLS, o modelo calculado com a expressão acima também possui o atributo `.summary()` para a consulta aos resultados em forma tabulada, onde constam as informações sobre os coeficientes, log-likelihood, estatísticas Z, etc.

Nota importante sobre a estatística Z citada acima: Z é a distribuição normal padrão, a estatística t de Student serve para os modelos gaussianos, para os demais modelos dentro do guarda-chuvas dos GLM utiliza-se a estatística Z de Wald.

### Verificações para o modelo
Após rodar o modelo temos os resultados com `modelo.summary()`, conforme citado acima.

#### Cálculo do $\chi$<sup>2</sup> ('chi-quadrado', para validade ou não do modelo)
Na modelagem através de `sm.logit.from_formula` o summary retorna também a informação LLR (log-likelihood ratio), equivalente ao **p-value** da estatística F no modelo OLS, contudo aqui é um $\chi$<sup>2</sup> por se tratar de target qualitativo.

O primeiro passo é rodar um modelo nulo (sem betas, apenas com o alfa). Depois verifica-se o incremento no loglike quando se altera do modelo nulo para o modelo funcional. Se o incremento do loglike for estatisticamente significante pelo menos um beta será diferente de zero.
```
modelo_nulo = sm.Logit.from_formula('target ~ 1', data=df).fit()
```

Os valores dos loglikes podem ser obtidos através do atributo `.llf` de cada modelo, por exemplo:
```
modelo_nulo.llf
```
  
O cálculo do $\chi$<sup>2</sup> é realizado conforme segue abaixo.
```
$\chi$<sup>2</sup> = -2 * (LL<sub>0</sub> - LL<sub>m</sub>)
```

onde LL<sub>0</sub> é o loglike do modelo nulo e LL<sub>m</sub> é o loglike do modelo estimado.

No Python:
```
chi2 = -2*(modelo_nulo.llf - modelo.llf)
pvalue_chi2 = stats.distributions.chi2.sf(chi2, modelo.df_model)
```
A análise é análoga à da estatística F dos modelos OLS:
- H<sub>0</sub>: se o p-value do $\chi$<sup>2</sup> > 0,05, então b<sub>1</sub> = b<sub>2</sub> = b<sub>3</sub> = ... = b<sub>n</sub> = 0. Deste modo, nenhum beta é estatisticamente significante e o modelo cai por terra, não podendo ser utilizada para fins preditivos;
- H<sub>1</sub>: se o p-value do $\chi$<sup>2</sup> $\le$ 0,05, então pelo menos um beta é diferente de zero e estatisticamente significante.

#### Comparação entre modelos
Os modelos de logística binária podem ser comparados através de três parâmetros: o pseudo R<sup>2</sup> (exibido no summary), o AIC (Akaike Info Criterion) e o BIC (Bayesian Info Criterion).
- Pseudo R<sup>2</sup>: quanto maior, melhor.
- AIC e BIC: quanto mais próximo de zero, melhor.

Os valores AIC e BIC são atributos dos modelos estabelecidos (`modelo.aic` e `modelo.bic`). Para cálculo manual:
```
aic = -2 * (modelo.llf) + 2 * (len(modelo.params) + 1)
bic = -2 * (modelo.llf) + (len(modelo.params) + 1)*np.log(len(modelo))
```

#### Predição
Pode ser 'aberta' a expressão do modelo e substituem-se os betas e os parâmetros, como pode ser utilizado o atribuito `predict` do modelo calculado:
```
modelo.predict(pd.DataFrame({target:value, 'feature1':value, 'feature2':value'}))
```

Para realizar a predição (ou, a probabilidade) de cada um dos eventos, pode-se registrar:
```
df['y_chapeu'] = modelo.predict()
```

#### Cutoff
Com a utilização de cutoffs (o nível é definido pelo programador, representando ponto de corte das probabilidades e estabelecido como float entre 0 e 1), temos que:
- Se `y_chapeu` $\ge$ cutoff: evento
- Se `y_chapeu` < cutoff: não-evento

#### Matriz de Confusão
Código copiado do prof. Fávero:
```
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    # Visualização dos principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores


matriz_confusao(observado=df[target],
                predicts=df['y_chapeu'], 
                cutoff=0.5)
```
Com o auxílio do pacote `scikit-learn`, a função acima já gera a matriz de confusão baseada no cutoff definido pelo operador. Reforçando: uma matriz de confusão qualquer é estabelecida sempre em função do cutoff definido pelo programador, em se alterando o nível do cutoff também alteram-se os valores distribuídos na matriz de confusão.

Na Matriz de Confusão, levando em consideração o cutoff específico que foi estabelecido, temos que:
> Sensitividade (ou *recall*), tida como taxa de acerto dos eventos = TP / (TP + FN)  # ou TP / Positives

> Especificidade, tida como taxa de acerto dos não-eventos = TN / (FP + TN)  # ou TN / Negatives

> Acurácia, tida como eficiência geral do modelo = (TP + TN) / total

> Precision = TP / (TP + FP)

> F1score, tida como média harmônica entre Recall e Precision = 2 * Recall * Precison / (Recall + Precision)

#### Curva ROC
Área abaixo da curva ou Receiver Operating Characteristic. É necessária a verificação da ROC para todo santo modelo de classificação que se programa.

O gráfico da curva ROC é estabelecido estipulando a Sensitividade (ou tpr - true positive rate) no eixo Y e (1 - Especificidade = fpr = FP / (FP + TN), onde 'fpr' é a false positive rate) no eixo X. 

A métrica de avaliação AUC é calculada como a área abaixo de uma curva ROC (característica de operação do receptor) e é uma representação escalar do desempenho esperado de um classificador. A AUC está sempre entre 0 e 1, com um número maior representando um classificador melhor.

A construção se dá mais ou menos assim:
```
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds =roc_curve(df[target], df['y_chapeu'])
roc_auc = auc(fpr, tpr)

# Cálculo do coeficiente de GINI, explicado na próxima seção
gini = (roc_auc - 0.5)/(0.5)

# Plot da curva
plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, marker='o', color='blue', markersize=10, linewidth=3)
plt.plot(fpr, fpr, color='gray', linestyle='dashed')
plt.title('Área abaixo da curva: %g' % round(roc_auc, 4) +
          ' | Coeficiente de GINI: %g' % round(gini, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()
```

#### Coeficiente de Gini
Acredito não ser o mesmo raciocínio do Gini da Economia, indicador da concentração (ou não) de qualquer coisa - normalmente tido como indicador de concentração de renda.

Este Gini calcula o valor da ROC acima da linha de 45° no gráfico (linha de 'chute' ou em inglês 'guess'), oscilando entre -1 e +1, com um score 0 resultando igual à linha guess. Para valores negativos de Gini pode-se descartar o modelo de imediato, uma vez que esse resultado é inferior à linha de chute.

#### Procedimento Stepwise
Pode (para não escrever 'deve') ser realizado, de modo a manter apenas as variáveis estatisticamente significantes no modelo final. Realizado conforme constante [acima](#procedimento-stempwise).

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
