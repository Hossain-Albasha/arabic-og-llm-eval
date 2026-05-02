# Arabic / Oil & Gas LLM Evaluation Benchmark (`arog-eval`)

[![tests](https://github.com/Hossain-Albasha/arabic-og-llm-eval/actions/workflows/test.yml/badge.svg)](https://github.com/Hossain-Albasha/arabic-og-llm-eval/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Benchmark: CC BY 4.0](https://img.shields.io/badge/Benchmark-CC%20BY%204.0-lightgrey.svg)](./benchmark/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A small, opinionated evaluation benchmark and harness for testing how well large language models handle **bilingual Arabic / English Oil & Gas technical content**.

Most LLM evaluation suites are English-only, or treat Arabic as a generic translation pair. Industrial Arabic is different: terminology is technical, transliterations vary across operators, and code-switching with English is the norm. This benchmark tests for the failure modes that actually surface in production AR/EN industrial deployments.

> **Status:** v1 baseline. Code is functional, benchmark is hand-curated and small (50 questions). Designed to grow.

## Why this exists

If you fine-tune or deploy an LLM for any of:

- Arabic O&G technical Q&A
- Bilingual document retrieval over industrial corpora
- Code-switched chat assistants for Saudi / GCC enterprise users
- Translation of technical specifications, SOPs, or incident reports

...then standard benchmarks (MMLU, HellaSwag, TyDi-QA, even CIDAR) won't tell you whether the model actually works for your use case. They miss:

1. **Domain-specific terminology**: "valve" can be `صمام`, `محبس`, `بلف`, or anglicized `vahf` depending on operator. The model needs to handle all four.
2. **Code-switching**: real Saudi engineering text reads "افحص الـ wellhead pressure قبل الـ test." Pure-Arabic and pure-English benchmarks miss this.
3. **Diacritization sensitivity**: technical terms in formal documentation often carry diacritics that shift meaning. Models trained on undiacriticized web Arabic struggle.
4. **Hallucination on industrial proper nouns**: equipment tags (P-101, V-203A) and standards (API 5L, ISO 14224) need to round-trip exactly.

This benchmark probes those.

## What's inside

```
arabic-og-llm-eval/
├── arog_eval/              # The Python harness
│   ├── schemas.py          # Pydantic types for benchmark items + results
│   ├── runner.py           # Eval orchestration
│   ├── metrics.py          # Exact match, F1, semantic sim, LLM-judge, etc.
│   ├── adapters/           # Pluggable model adapters
│   ├── judges/             # LLM-judge implementations
│   ├── reports.py          # Markdown report generation
│   └── failure_modes.py    # Failure-mode taxonomy + classifiers
├── benchmark/v1/           # The benchmark itself (JSONL)
│   ├── terminology.jsonl   # 20 items: AR↔EN technical term recall
│   ├── safety.jsonl        # 10 items: incident reasoning, hazard ID
│   ├── process_qa.jsonl    # 10 items: drilling, refining, production
│   ├── code_switched.jsonl # 5 items: mixed AR/EN prompts
│   └── doc_grounding.jsonl # 5 items: answer from passage
├── scripts/
│   └── run_benchmark.py    # One-command entry point
└── results/                # Run outputs go here (gitignored)
```

## Quickstart

```bash
# Clone and install
git clone https://github.com/<you>/arabic-og-llm-eval.git
cd arabic-og-llm-eval
pip install -e .

# Set your API keys
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

# Run against a model
python scripts/run_benchmark.py --model claude-sonnet-4-5 --output results/sonnet-4-5.json

# Generate the report
python scripts/run_benchmark.py --report results/sonnet-4-5.json --out reports/sonnet-4-5.md
```

The runner streams progress, persists per-item results to disk as it goes (so a crashed run can resume), and produces a Markdown report with per-category scores plus failure-mode breakdown.

## Models supported out of the box

Pluggable adapter pattern. Bundled:

- **Anthropic**: Claude models via the official SDK
- **OpenAI**: GPT models via the official SDK
- **Ollama**: Local models served by Ollama (Qwen, Llama, Mistral, Jais, AceGPT, ...)
- **HuggingFace**: Direct `transformers` loading for self-hosted models

Adding a new provider is one file in `arog_eval/adapters/` implementing the `ModelAdapter` protocol.

## Metrics

Every item is scored on multiple axes. No single number tells the whole story.

| Metric | What it measures | Range |
|---|---|---|
| `exact_match` | Normalized string equality | 0 / 1 |
| `f1_token` | Token-overlap F1 against reference | 0 to 1 |
| `semantic_similarity` | Cosine similarity of multilingual e5-large embeddings | 0 to 1 |
| `llm_judge` | Claude rates response 0 to 10 against a rubric | 0 to 10 |
| `refusal_rate` | Heuristic for when model refused unnecessarily | 0 / 1 |
| `hallucination_rate` | When reference is "unknown" but model confidently answers | 0 / 1 |
| `cost_usd` | Computed from token counts × model rates | $ |
| `latency_ms` | End-to-end | ms |

Aggregate scores per category, plus an overall composite.

## Failure-mode taxonomy

Beyond pass/fail, every wrong answer is bucketed. From `arog_eval/failure_modes.py`:

- `WRONG_TRANSLITERATION`: picked the wrong common rendering of a term
- `WRONG_TERM`: picked a related but incorrect concept
- `OVER_FORMALITY`: used Modern Standard Arabic where dialect/technical was expected
- `UNDER_DIACRITIZATION`: dropped diacritics on a term where they disambiguate
- `CODE_SWITCH_FAILURE`: translated rather than preserving English technical terms
- `HALLUCINATED_PROPER_NOUN`: invented an equipment tag, standard, or vendor
- `REFUSED_VALID_REQUEST`: declined to answer a benign technical question
- `WRONG_LANGUAGE_OUTPUT`: answered in EN when AR was requested or vice versa

These are the patterns that show up over and over. Tagging them per-failure makes regressions visible across model versions.

## Roadmap

- [ ] v1.0: current scope, 50 hand-curated items, 4 adapters, 8 metrics
- [ ] v1.1: expand to 150 items, add diacritized variants
- [ ] v1.2: equipment-tag round-trip subtest (P&IDs)
- [ ] v1.3: RAG-grounded subtest with retrievable corpus
- [ ] v2.0: community submissions via PR template

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). PRs welcome for:

1. New benchmark items (use the JSONL schema; one item per line, with reference + metadata)
2. New model adapters (implement the protocol in `arog_eval/adapters/base.py`)
3. New metrics (add to `arog_eval/metrics.py` with tests)

Items added by others are credited in `benchmark/v1/CONTRIBUTORS.md`.

## License

- **Code**: [MIT](./LICENSE)
- **Benchmark items**: [CC-BY-4.0](./benchmark/LICENSE) (cite when using)

## Citation

If you use this benchmark in published work or in a model release, please cite:

```bibtex
@misc{albasha2026arogeval,
  author       = {Hossain Albasha},
  title        = {Arabic / Oil \& Gas LLM Evaluation Benchmark},
  year         = {2026},
  url          = {https://github.com/<you>/arabic-og-llm-eval},
  note         = {v1.0}
}
```

## Author

Built and maintained by [Hossain Albasha](https://hossainalbasha.com), bilingual AI / automation engineer working in Saudi industrial domains. Contact: `Albasha.Hossain@gmail.com`.

This benchmark grew out of work on [PetroLingua-AI](https://hossainalbasha.com/projects/petrolingua-ai), a bilingual O&G document intelligence platform. The failure modes I tagged manually while iterating on PetroLingua became the categories above.
