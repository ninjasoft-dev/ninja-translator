# Outputs e Caches

## Arquivos principais (por etapa)
- Tradução (`tradutor/translate.py`):
  - `saida/<slug>_pt.md`
  - `<slug>_translate_report.json` (status + contagens)
  - `<slug>_translate_metrics.json` (por chunk, inclui `chunk_profile` e tamanho do contexto usado)
  - `<slug>_pt_progress.json` (resume)
  - `<slug>_source_sections.json` (metadados de estrutura usados na revisão final)
  - `<slug>_pt_review_report.json` (revisão final automática + QA)
- Repair seletivo (`tradutor/repair.py`, chamado pela tradução):
  - `<slug>_repair_report.json` (totais da etapa)
  - `<slug>_repair_metrics.json` (por chunk)
  - Com `--debug`: `debug_runs/<slug>/<run>/45_repair/repair_manifest.json`
  - Com `--debug`: `45_repair/debug_repair/chunkNNN_before_pt.txt` e `chunkNNN_after_pt.txt` para chunks reparados.
- Refine (`tradutor/refine.py`):
  - `saida/<slug>_pt_refinado.md`
  - `<slug>_refine_report.json`
  - `<slug>_refine_metrics.json`
  - `<slug>_pt_refinado_progress.json` (resume)
  - Opcional: `<slug>_pre_refine_cleanup.md` quando `cleanup_before_refine` aplica.
  - `<slug>_pt_refinado_review_report.json` (revisão final automática + QA)
- Desquebrar (`tradutor/desquebrar.py`):
  - `<slug>_desquebrar_metrics.json` (quando LLM é usado)
  - Arquivos `_raw_extracted.md`, `_preprocessed.md`, `_raw_desquebrado.md` se `--debug`.
- PDF:
  - `saida/pdf/<slug>_pt_refinado.pdf` (quando `--pdf-enabled` ou config).
- Tempos:
  - `saida/<slug>_timings.json` (sempre ao final de `traduz`/`traduz-md`, inclusive em falha após início do processamento)
  - Com `--debug`: `debug_runs/<slug>/<run>/99_reports/timings.json`
  - `stages.translate` inclui o repair seletivo; quando houver tempo de repair, ele aparece também em `nested_stages.translation_repair` como detalhe sem dupla contagem.
  - `stages.post_translate_review` registra a revisão determinística após a tradução; `stages.post_refine_normalize` inclui a revisão final após o refine.

## Caches (`tradutor/cache_utils.py`)
- `saida/cache_traducao`
- `saida/cache_repair`
- `saida/cache_refine`
- `saida/cache_desquebrar`

Use `--clear-cache {all,translate,repair,refine,desquebrar}` para limpar.

## Debug / estado
- Tradução: `*_pt_chunks_debug.jsonl` (se `--debug-chunks`), `debug_traducao/` para falhas.
- Debug completo: `saida/debug_runs/<slug>/<timestamp>/40_translate/translate_manifest.json` inclui metadados do glossário por chunk; `debug_traducao/chunkNNN_glossary.txt` guarda o bloco de glossário enviado ao prompt.
- Repair: `saida/debug_runs/<slug>/<timestamp>/45_repair/repair_manifest.json` inclui problemas detectados, tentativas, cache, suspeitas e se o chunk foi alterado.
- Refine: `*_pt_refinado_chunks_debug.jsonl` (se `--debug-chunks`), `debug_refine*` quando `--debug-refine`.
- Estados rápidos: `saida/state_traducao.json`, `saida/state_refine.json`.
