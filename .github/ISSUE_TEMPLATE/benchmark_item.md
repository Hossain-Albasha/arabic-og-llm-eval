---
name: New benchmark item
about: Propose adding a new item to the benchmark
title: "[item] "
labels: benchmark
---

**Category**
- [ ] terminology
- [ ] safety
- [ ] process_qa
- [ ] code_switched
- [ ] doc_grounding

**Item (JSONL format)**

```json
{
  "id": "<category-prefix>-NNN",
  "category": "...",
  "language": "ar" | "en" | "mixed",
  "prompt": "...",
  "reference": "...",
  "alternative_references": [],
  "metadata": {
    "domain": "...",
    "difficulty": "easy|medium|hard",
    "tags": []
  }
}
```

**Why this item is valuable**
Briefly: what failure mode does it probe? Why isn't it covered by existing items?

**Sources for reference answer**
Public standard, textbook, or published incident report. No proprietary or operator-internal sources.

**Confirmation**
- [ ] I have authority to release this item under CC-BY-4.0
- [ ] No PII or operator-proprietary information
- [ ] Reference answer is technically correct in industrial-Arabic register
