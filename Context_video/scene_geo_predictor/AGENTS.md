# Project maintenance rules

- After every code bug fix in this project, append a dated entry to `BUGFIX_CHANGELOG.md`. Do not replace earlier entries.
- Each entry must identify the symptom, confirmed cause, changed files/functions, validation commands and outcomes, versioned artifacts, limitations, and whether the change is enabled by default or remains experimental.
- Use the actual execution date. Historical entries must be explicitly labelled retrospective; never invent a validation result or equate a completed run with a correctness pass.
- Preserve existing user changes and old experiment artifacts. Do not silently replace visual/physics models or relax validation thresholds. Ask the user about material uncertainty before dependent execution.
