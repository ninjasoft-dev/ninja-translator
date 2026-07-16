# Benchmarks rápidos de tradução/refine (Ollama)

Siga sempre o padrão abaixo, separando resultados de tradução e de refine.

Os modelos dependem do hardware e do servidor disponíveis em cada ambiente. Informe-os por CLI ou mantenha a seleção somente no `config.yaml` local.

Os diretórios gerados (`benchmark/traducao`, `benchmark/refine`,
`benchmark/e2e`, `benchmark/literario`) e PDFs de teste locais são ignorados
pelo Git. Versione apenas este README e entradas textuais pequenas/sintéticas.

Tradução (inglês → PT-BR):
```bash
python -m tradutor.bench_llms \
  --input benchmark/teste_traducao_en.md \
  --max-chars 1500 \
  --out-dir benchmark/traducao \
  --use-glossary
```

Por padrão, `bench_llms` usa o pipeline real de tradução: preprocessamento,
chunking, retries/sanitização e parâmetros de `config.yaml` (`translate_num_ctx`,
`translate_num_predict`, `translate_chunk_chars`, etc.). Para comparar o texto
inteiro, use `--max-chars 0`. O modo antigo de prompt único está disponível com
`--single-prompt`, mas ele não representa o comportamento do tradutor completo.
Para modelos que funcionam bem no app do Ollama mas retornam vazio via
`/api/generate`, teste `--ollama-api-mode chat --ollama-think false`.
Com `--use-glossary`, o benchmark carrega `glossario/glossario_manual.json`
quando existir; caso contrário, usa `glossario/glossario_geral.json`. Cada
modelo também gera um `*_qa.json` com checagens de termos canônicos, inglês
residual, aliases ruins de nomes, marcadores internos e possíveis problemas de
gênero.

Refine (texto já em PT-BR):
```bash
python -m tradutor.bench_refine_llms \
  --input benchmark/teste_refine_pt.md \
  --max-chars 1500 \
  --out-dir benchmark/refine \
  --use-glossary
```

Sem `--models`, os scripts usam todos os modelos retornados por `ollama list`. Use `--models <m1> <m2>` para limitar a um subconjunto.

End-to-end (matriz tradutor → refinador):
```bash
python -m tradutor.bench_e2e_llms \
  --input benchmark/teste.pdf \
  --translate-models modelo-tradutor-a modelo-tradutor-b \
  --refine-models modelo-refinador-a modelo-refinador-b \
  --max-chars 2500 \
  --out-dir benchmark/e2e \
  --desquebrar-mode safe \
  --use-glossary
```

Sugestão prática: rode a matriz e2e primeiro em amostra (`--max-chars 2500` ou
5000) para encontrar boas combinações de tradutor/refinador. Depois rode o livro
completo (`--max-chars 0`) apenas nas melhores combinações. Para testar só pares
com o mesmo modelo nas duas etapas, use `--same-model-only`.

Estrutura sugerida de arquivos:
```
benchmark/
  README.md
  teste.pdf                # opcional, entrada em PDF
  teste.md                 # opcional, entrada simples
  teste_traducao_en.md     # entrada padrão para tradução
  teste_refine_pt.md       # entrada padrão para refine
  traducao/
    resumo_teste_traducao_en.md
    teste_traducao_en_<modelo>.md
    teste_traducao_en_<modelo>_qa.json
    ...
  refine/
    resumo_teste_refine_pt.md
    teste_refine_pt_<modelo>_refine.md
    teste_refine_pt_<modelo>_refine_qa.json
    ...
  e2e/
    resumo_e2e_teste.md
    teste_<tradutor>__<refinador>_pt.md
    teste_<tradutor>__<refinador>_pt_refinado.md
    teste_<tradutor>__<refinador>_qa.json
    ...
```
