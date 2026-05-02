# Failure-mode taxonomy

A categorization of *why* models get arog-eval items wrong. The harness applies these heuristically per item; the labels are useful for tracking model behavior across versions even when overall F1 stays roughly constant.

## The taxonomy

### `WRONG_TRANSLITERATION`

The model picked a different but related Arabic rendering of a technical term than the reference. For example, the reference is `صمام` (valve) and the model returned `محبس`. Both are valid common renderings; the model isn't *wrong* in a vacuum, but it failed to match the reference. Tracking this is useful when you want to verify the model knows the operator's preferred terminology.

**Detector:** matches against a curated set of transliteration groups in `arog_eval/failure_modes.py`. If both response and reference normalize to different members of a known group, the flag fires.

### `WRONG_TERM`

The model picked a related concept that's still incorrect. Example: question asks for "valve" (`صمام`), model returns "tap" (`صنبور`). Detector relies on F1 dropping below a threshold while semantic similarity remains moderate.

### `OVER_FORMALITY`

The model gave a verbose answer when a short one was expected. For terminology items where the reference is one or two words, a multi-paragraph answer is a regression even if the right term is buried in it. Real users want the term, not a tutorial.

**Detector:** for terminology category, response length > 4× reference length plus 5 tokens.

### `UNDER_DIACRITIZATION`

The reference preserves diacritics (tashkeel) because they disambiguate the term, but the model dropped them. Example: reference is `حُرقة` (burning sensation), response is `حركة` (movement). The skeleton is the same; only diacritics distinguish them.

**Detector:** Arabic-text reference contains diacritics, response doesn't.

### `CODE_SWITCH_FAILURE`

For mixed AR/EN prompts, the model translated the English technical terms into Arabic instead of preserving them. Real Saudi and Gulf engineering text reads "افحص الـ wellhead pressure". Translating to "افحص ضغط رأس البئر" misses the register the user actually writes in.

**Detector:** code-switched category items where reference contains English tokens but response does not.

### `HALLUCINATED_PROPER_NOUN`

The response cites a standard, equipment tag, or vendor name that doesn't appear in the item. Common patterns: "API 520" when no API standard is referenced; "ASME 9999" (a fake number); equipment tags like "P-101" invented from thin air.

**Detector:** regex for typical industrial standard patterns; flags any matches in the response that aren't in the prompt or reference.

### `REFUSED_VALID_REQUEST`

The model refused to answer a benign technical question. Examples: declining to explain how a blowout preventer works because it sounds dangerous; refusing to translate a technical term because it requires "expertise the model doesn't have."

**Detector:** keyword matching on common refusal markers in English and Arabic.

### `WRONG_LANGUAGE_OUTPUT`

The prompt requested Arabic, the response is in English (or vice versa). For mixed-language prompts, this is not flagged.

**Detector:** Arabic character ratio in response vs. expected language of item.

## How to interpret the failure-mode distribution

A run's overall accuracy can be the same across two model versions while the failure modes shift. That shift is the real signal:

- **More `WRONG_TRANSLITERATION`**: the model is learning new terminology but not the operator's preferred form. May indicate a need for fine-tuning on the operator's glossary.
- **More `OVER_FORMALITY`**: the model is hedging more, possibly due to RLHF tuning toward verbose helpfulness. Penalize in the system prompt.
- **More `REFUSED_VALID_REQUEST`**: the model has been trained to be more cautious. Test whether this is industry-wide or specific to a model version.
- **More `HALLUCINATED_PROPER_NOUN`**: the model is fabricating citations. Critical for any deployment that surfaces standards or equipment tags.
- **More `UNDER_DIACRITIZATION`**: the model is treating diacritics as noise. Bad for deployments where the source documents are diacritized (formal/safety/legal text).

## Adding new failure modes

Open a PR. Each new mode needs:

1. An entry in the `FailureMode` enum in `schemas.py`
2. A detector function in `failure_modes.py`
3. Inclusion in `classify_failure_modes`
4. Tests in `tests/test_failure_modes.py`
5. A short description in this file

Detectors should over-flag rather than under-flag. False positives are easier to filter than false negatives.
