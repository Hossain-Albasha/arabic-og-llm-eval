# Methodology

## Why this benchmark exists

Standard LLM benchmarks under-test the skills that matter for industrial bilingual deployments. MMLU and HellaSwag don't probe technical Arabic. TyDi-QA tests Arabic comprehension but on Wikipedia-style topics. CIDAR is human-reviewed Arabic instruction-following but again domain-general. None of them tell you whether a model can answer "what's the recommended action when an ESP shows high vibration alarm?" or correctly distinguish *صمام* (valve) from *صنبور* (faucet) in the context of well control.

This benchmark is opinionated. It targets the specific failure modes that surfaced repeatedly while building production AR/EN AI systems for Oil & Gas. It is not exhaustive. It is meant to be cited *alongside* general benchmarks, not in place of them.

## What the benchmark covers

Five categories, each probing a different competence:

- **Terminology**: Can the model recall the standard or alternative renderings of technical terms across languages?
- **Safety**: Can it reason about hazard analysis, permit-to-work systems, well control, and incident response?
- **Process Q&A**: Does it know how the underlying physics and chemistry of upstream and downstream operations actually work?
- **Code-switched**: Can it handle the way Saudi/Gulf engineers actually communicate, mixing Arabic and English mid-sentence?
- **Document grounding**: Can it answer questions strictly from a provided passage, including correctly saying "unknown" when the passage doesn't contain the answer?

## Item authoring guidelines

Items are written by domain practitioners, not crowd workers. Each item is reviewed for:

1. **Technical correctness.** Reference answers must be technically defensible by O&G operations professionals.
2. **Linguistic appropriateness.** Arabic items use the dominant industrial register, not classical or colloquial Arabic.
3. **Multiple valid answers.** Where a term has multiple common renderings (e.g., صمام / محبس / بلف for valve), all are listed in `alternative_references`.
4. **No PII or proprietary data.** Items reference public standards (API, ISO, ASME, BSEE) and publicly published incident analyses, not specific operator procedures.

## Scoring philosophy

A response is scored on multiple axes because no single number captures whether the model is fit for purpose:

- **Exact match** is the strictest test, applied after Arabic-aware normalization (alef variants, diacritics, teh marbuta, Arabic-Indic digits).
- **F1 token** captures partial credit on longer answers.
- **Semantic similarity** (when enabled) uses multilingual e5-large embeddings to catch semantically equivalent phrasings.
- **LLM-judge** uses Claude as an external evaluator with a strict rubric that downweights fluency in favor of factual accuracy and terminology precision.
- **Refusal rate** flags excessive caution.
- **Hallucination rate** flags fake standards, equipment tags, or vendor names that don't appear in the item.

A model is *passing* a category if mean F1 > 0.6 *and* judge mean > 6.0 *and* refusal rate < 5%. These thresholds are deliberately strict; calibrate as the benchmark matures.

## Failure-mode classification

Beyond pass/fail, every wrong answer is bucketed into one or more failure modes. This is the most useful output for tracking model behavior across versions: a model can have the same overall F1 score in two different runs while having very different failure-mode distributions, and the failure-mode distribution is what tells you whether the model is regressing on terminology while improving on safety, or vice versa.

The classifiers are heuristic. They're not perfect. They're tuned to over-flag rather than under-flag, on the principle that human review of false-positive flags is cheaper than missed regressions.

## Running the benchmark responsibly

A few things to keep in mind:

- **API costs are real.** A full v1.0 run (50 items × ~500 tokens average) is roughly 25,000 tokens. Cost ranges from <$0.05 (local Ollama) to ~$1.00 (Opus). With LLM-judge enabled, double those numbers.
- **Don't fit to the benchmark.** If you train against benchmark items, you'll get good scores and a model that doesn't generalize. The benchmark is in the open exactly so you can't game it without overfitting.
- **Cite when used.** The benchmark items are CC-BY-4.0. Attribution is required.
- **Contribute back.** If you find a missing failure mode, an under-represented terminology cluster, or an item with a contestable reference answer, open a PR.

## Versioning

The benchmark is versioned. A v1.x release means the items themselves are stable. v2.0 will introduce new items and possibly new categories; results across major versions are not directly comparable.

When you publish results, always cite the version: "EM 0.78 on arog-eval v1.0" is interpretable; "EM 0.78 on arog-eval" is not.
