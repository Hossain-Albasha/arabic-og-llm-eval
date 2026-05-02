# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] (2026-05-03)

Initial release.

### Added

- Benchmark v1.0 with 50 hand-curated items across 5 categories:
  - terminology (20)
  - safety (10)
  - process_qa (10)
  - code_switched (5)
  - doc_grounding (5)
- Pluggable model adapter protocol with bundled implementations:
  - Anthropic Claude
  - OpenAI GPT
  - Ollama (local)
- 8 evaluation metrics:
  - exact_match (Arabic-aware normalized)
  - f1_token
  - semantic_similarity (multilingual e5-large)
  - llm_judge (Claude with strict rubric)
  - refusal detection
  - hallucination detection
  - cost (per-token pricing)
  - latency
- 8-category failure-mode taxonomy with heuristic classifiers:
  - WRONG_TRANSLITERATION
  - WRONG_TERM
  - OVER_FORMALITY
  - UNDER_DIACRITIZATION
  - CODE_SWITCH_FAILURE
  - HALLUCINATED_PROPER_NOUN
  - REFUSED_VALID_REQUEST
  - WRONG_LANGUAGE_OUTPUT
- Resume-friendly streaming runner that persists per-item results to JSONL
- Markdown report generator
- Test suite (47 tests, all passing)
- GitHub Actions CI workflow
- Documentation: methodology, failure-modes, adding-models

### Notes

- Benchmark items are CC-BY-4.0 licensed.
- Harness code is MIT licensed.
- Pricing tables in adapters reflect rates as of release date and will drift; PRs welcome.

[0.1.0]: https://github.com/Hossain-Albasha/arabic-og-llm-eval/releases/tag/v0.1.0
