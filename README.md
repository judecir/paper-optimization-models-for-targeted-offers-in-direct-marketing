# Seleção de ofertas em marketing direto

Implementação dos modelos M1, M2 e M3 de Talla Nobibon, Leus e Spieksma
(2011), com instância de validação, gerador inspirado na Secção 5.1 e execução
de experiências. O notebook [marketing_direto.ipynb](./marketing_direto.ipynb)
apresenta o exemplo e as chamadas principais; o código reutilizável está nos
módulos Python deste diretório.

## Requisitos e instalação

Python 3.11 ou superior. Instale os módulos e dependências de teste:

```powershell
python -m pip install -e ".[test]"
```

HiGHS (`appsi_highs`) é o solver padrão e é instalado com `highspy`. Para Gurobi
ou CPLEX, instale/configure o solver e sua licença no ambiente e passe
`--solver gurobi` ou `--solver cplex`.

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

M3 enumera explicitamente subconjuntos e é intencionalmente limitado a
`m <= 12`; use-o para instâncias pequenas como a validação, não para o plano
experimental de larga escala.