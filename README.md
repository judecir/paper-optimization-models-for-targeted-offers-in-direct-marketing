# Seleção de ofertas em marketing direto

Implementação dos modelos M1, M2 e M3 de Talla Nobibon, Leus e Spieksma
(2011), com instância de validação, gerador inspirado na Secção 5.1 e execução
de experiências. O notebook [marketing_direto.ipynb](./marketing_direto.ipynb)
apresenta o exemplo e as chamadas principais; o código reutilizável está nos
módulos Python deste diretório.

## Requisitos e instalação

Python 3.11 ou superior. Instale os módulos e dependências de teste:

```powershell
python -m pip install -e ".[test,plots]"
```

HiGHS (`appsi_highs`) é o solver padrão e é instalado com `highspy`. Para Gurobi
ou CPLEX, instale/configure o solver e sua licença no ambiente e passe
`--solver gurobi` ou `--solver cplex`.
O extra `plots` instala Matplotlib para os gráficos comparativos do notebook.

## Testes e exemplo pequeno

```powershell
python -m pytest
```

O exemplo conhecido está em `validation_instance()` em [instance.py](./instance.py).
Os testes validam os oito conjuntos admissíveis de M3 e seus coeficientes,
confirmam o ótimo inteiro único 11 por enumeração de todas as 64 matrizes de
ofertas, resolvem M1/M2/M3, testam a ordem dos limites LP e verificam a
reprodutibilidade do gerador.

`models.py` expõe `build_model`, `solve_model`, `brute_force_optimum` e
`validate_solution`. `solve_model` aceita `formulation="M1"` ou `"M2"`,
`relax=True` para a relaxação LP e `solver="appsi_highs"`, `"gurobi"` ou
`"cplex"`. Toda solução inteira retornada é verificada independentemente das
restrições originais.

## Gerar instâncias

Gere por padrão os grupos S1–S3 (162 instâncias):

```powershell
python generator.py --output-dir instances --seed 1
```

Inclua M1, M2 e L (324 instâncias no total):

```powershell
python generator.py --output-dir instances --seed 1 --include-large
```

Cada combinação de tamanho, `n`, hurdle rate, variante de capacidade e semente
gera os dados comuns uma única vez; as três instâncias resultantes diferem
somente em `B_j`. O modo de orçamento `random` usa
`ceil(2 * (sum_i M_i / n) * C_j / m)` como limite superior, conforme a decisão
documentada no gerador, em vez da expressão aparentemente tipográfica do PDF.
Para evitar intervalos impossíveis de orçamento, o limite superior de MQC é
`floor(2 * sum_i M_i / n)` (em vez do teto da expressão): isso evita casos em
que `floor(O_j * C_j / m)` supera o orçamento máximo. Se essa faixa de MQC não
contiver nenhum inteiro, o gerador informa erro explicitamente.
Custos fixos são inteiros. Se o intervalo admissível de `f_j` ficar vazio ou
negativo, usa-se `max(0, ceil(limite inferior))` e emite-se um aviso.

## Resolver e interpretar os CSVs

Com as instâncias já geradas:

```powershell
python run_experiments.py --instances-dir instances --results results.csv --aggregate aggregate.csv
```

O limite de tempo padrão é 3600 segundos por resolução inteira ou LP. Opções:

```powershell
python run_experiments.py --instances-dir instances --solver gurobi --time-limit 600
python run_experiments.py --instances-dir instances --generate --include-large --seed 7
```

`results.csv` tem uma linha por instância e formulação M1/M2, com objetivo
solver utilizado, incumbent, melhor bound reportado pelo solver, gap MIP,
valor LP, gap LP em porcentagem, tempo, nós de branch-and-bound e estados.
O gap LP é
`(z_LP - z_IP) / z_IP * 100`; fica vazio quando não há solução ou quando o
objetivo inteiro é zero. `aggregate.csv` resume gaps LP/MIP médios, tempo médio
e quantidade de instâncias resolvidas até a otimalidade por grupo, `n`, modelo
e solver. O notebook também mostra como gerar, executar e comparar os 54 casos
S1 com HiGHS, Gurobi e CPLEX em arquivos CSV consolidados.
Durante essa execução, o notebook informa a conclusão de cada instância.
O limite é de 3600 segundos por resolução inteira ou LP; somente esse limite é
configurado explicitamente, e as demais opções ficam nos padrões dos solvers.
O campo `elapsed_seconds` registra o tempo da resolução inteira, não da
relaxação LP. Depois da execução, o notebook cria gráficos em inglês de tempo
geral e detalhados por `n` e formulação em `results/plots/`. As relaxações são
comparadas em tabelas por solver, tamanho e formulação, pois os gaps de uma
mesma formulação e instância são iguais entre solvers (salvo tolerâncias
numéricas). A tabela geral também apresenta tempos médios e medianos, vitórias
por grupo e razões de tempo pareadas.

M3 enumera explicitamente subconjuntos e é intencionalmente limitado a
`m <= 12`; use-o para instâncias pequenas como a validação, não para o plano
experimental de larga escala.

## Comparar as relaxações LP de M1, M2 e M3

O notebook inclui nove instâncias-base com `m = 4, 8, 12` clientes e mais 50
instâncias com `m = 4, ..., 12`, todas com até 12 clientes para permitir a
enumeração de colunas do M3. As instâncias adicionais variam `n`, hurdle rate,
capacidade e orçamento e usam seeds consecutivos a partir de 4000; candidatos
sem ótimo inteiro positivo são descartados, pois não permitem calcular gap
relativo. Para cada instância compara M1/M2/M3, verifica a concordância dos
ótimos inteiros e resolve as relaxações LP com HiGHS, Gurobi e CPLEX. As
instâncias ficam em `instances/M3_small/` e `instances/M3_small_extra/`; os
resultados detalhados e as tabelas agregadas ficam em
`results/M3_small_lp_results.csv`, `results/M3_small_lp_summary.csv`,
`results/M3_small_extra_results.csv` e
`results/M3_small_extra_summary.csv`. O gap é calculado em relação ao ótimo
inteiro comum; um valor menor indica uma relaxação mais forte. As tabelas
separam os solvers mesmo quando os gaps coincidem numericamente e incluem
tempo inteiro para as 50 instâncias adicionais.