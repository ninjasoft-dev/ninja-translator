# AGENTS.md — Contexto rápido para o Codex

## Objetivo do projeto (5–10 linhas)
- Pipeline completo para traduzir Light Novels de vários idiomas para **PT-BR** usando LLMs (Ollama, Gemini ou OpenAI).
- Fluxo cobre extração de PDF/MD, limpeza, “desquebrar” linhas, tradução em chunks com contexto deslizante, QA/repair seletivo, refine, revisão final determinística e geração de PDF.
- Configuração central em `config.yaml`, com overrides por flags de CLI.
- Saídas e auditoria gravadas em `saida/` (markdown final, métricas, manifests, relatórios de revisão e PDFs).
- Glossários manuais/dinâmicos são suportados e injetados por chunk.
- Projeto prioriza uso em Windows, mas funciona em qualquer ambiente Python 3.10+.
- Não commitamos dados reais (glossários, PDFs, chaves); use arquivos de exemplo.

## Mapa de pastas (o que é o quê)
- `tradutor/` — código principal do pipeline (CLI, tradução, repair, refine, PDF, utils).
- `data/` — PDFs de entrada (não versionar conteúdo real).
- `saida/` — saídas, caches e artefatos de debug (gerado em runtime).
- `glossario/` — glossários manuais (mantém só exemplos no Git).
- `benchmark/` — dados e scripts de benchmark.
- `tests/` — testes unitários/smoke.
- `config.yaml` — configuração local, criada pelo desenvolvedor e ignorada pelo Git.
- `config.example.yaml` — configuração portátil para copiar e adaptar.
- `desquebrar.py`, `tradutor.py`, `refinador.py` — wrappers legados para CLI.

## Principais entrypoints/CLIs (com exemplos)
- Interface gráfica (executa a CLI em subprocesso):
  \`\`\`bash
  python interface.py
  \`\`\`
- CLI principal (traduz/refina/pdf):
  ```bash
  python -m tradutor.main traduz --input "data/meu_livro.pdf"
  python -m tradutor.main traduz-md --input "saida/meu_texto_desquebrado.md"
  python -m tradutor.main refina --input "saida/meu_livro_pt.md"
  python -m tradutor.main pdf --input "saida/meu_livro_pt_refinado.md"
  ```
- Desquebrar direto (wrapper legado):
  ```bash
  python desquebrar.py --input "arquivo.md" --output "arquivo_desquebrado.md" --config config.yaml
  ```
- Wrappers legados: `python tradutor.py ...`, `python refinador.py ...` (chamam `tradutor.main`).

## Testes, lint, checks
- Testes (smoke/unit):
  ```bash
  pytest -q
  ```
- Lint e ordenação de imports: `ruff check .`
- Formatação: `ruff format --check .`
- Todos os hooks locais: `pre-commit run --all-files`

## Pipeline principal end-to-end (com flags úteis)
```bash
python -m tradutor.main traduz \
  --input "data/meu_livro.pdf" \
  --pdf-enabled \
  --translate-allow-adaptation \
  --request-timeout 180 \
  --num-predict 3072
```
- Flags comuns: `--skip-front-matter`, `--split-by-sections`, `--translation-repair/--no-translation-repair`, `--debug`, `--debug-chunks`, `--clear-cache {all,translate,repair,refine,desquebrar}`.
- Modelos/ctx/num_predict podem ser configurados no `config.yaml` ou sobrescritos por flags.

## Caches, progresso, estado (e como limpar)
- Caches por chunk (em `saida/` por padrão):
  - `saida/cache_traducao`, `saida/cache_repair`, `saida/cache_refine`, `saida/cache_desquebrar` (ver `tradutor/cache_utils.py`).
- Progress/state:
  - `*_progress.json` (tradução/refine), `state_refine.json` (refine).
  - Debug: `*_pt_chunks_debug.jsonl`, `*_chunks_debug.jsonl`.
  - Revisão final: `<slug>_source_sections.json`, `<slug>_pt_review_report.json`, `<slug>_pt_refinado_review_report.json`.
- Limpeza segura:
  - Preferível: `--clear-cache all` na CLI.
  - Manual: apagar `saida/cache_*` e `saida/*_progress.json` quando iniciar um run limpo.
- Para evitar “cross-run contamination”:
  - Use `--clear-cache` ao mudar modelos/prompts.
  - Garanta que `output_dir` e `data_dir` (em `config.yaml`) sejam específicos por projeto/volume.

## Convenções do projeto
- Estilo: não há guia formal; siga PEP 8 e o estilo existente (funções pequenas, logs com `logging`).
- Logs: use `logging.getLogger(__name__)` (padrão no código).
- Nomes de arquivo: saídas seguem `<slug>_pt.md`, `<slug>_pt_refinado.md`, `*_metrics.json` etc. (ver `tradutor/translate.py`, `tradutor/refine.py`).
- Padrão de commit/branch: **não encontrado** no repo (busquei em README/arquivos de config). Se definir, documente aqui.

## Regras de segurança
- **NÃO** inclua segredos (tokens, chaves, `.env`) em commits.
- Se encontrar `.env`, **não** copie valores; apenas documente as variáveis esperadas.
- Variáveis esperadas observadas:
  - `GEMINI_API_KEY` (quando backend Gemini é usado).
  - `OPENAI_API_KEY` (quando backend OpenAI é usado).
