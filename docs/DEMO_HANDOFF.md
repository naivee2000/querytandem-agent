# QueryTandem Agent — website demo handoff

**Start here on the computer that has the website project.** This package supplies
the product brief, verified examples, measured claims, and implementation checks.
It does not supply a finished interface. Discuss placement and styling with
Shaoyang; the initial preference is somewhere in the website's **About Me** area.

## 1. The intended demo

Build a small, interactive **recorded-example explorer** for portfolio visitors.
Use the website's existing framework, components, typography, and colors. Do not
introduce a second framework or a separate service for this feature.

Visitors should be able to:

1. Choose one of three saved questions.
2. Read the original question and its supplied evidence/hint.
3. Compare the baseline and compact configurations' **final pipeline SQL**.
4. Inspect their result previews, the reference result, and correctness labels.
5. Compare local-generator tokens and whole-pipeline tokens as separate metrics.
6. Expand a schema summary and follow links to the project and experiment reports.

Keep **“Recorded experiment examples”** visible. Explain that predictions came
from saved runs; answer previews were rechecked locally using those saved queries.
These are deliberately selected illustrations, not a representative sample or
a live chat. Display benchmark-wide metrics separately from the selected example.

Layout, the exact About Me placement, and whether to use a card, modal, inline
section, or dedicated detail view should be discussed on the website computer.
This handoff intentionally does not choose those design details.

## 2. What the project does and what Shaoyang contributed

QueryTandem extends [HKUSTDial/DeepEye-SQL](https://github.com/HKUSTDial/DeepEye-SQL),
an existing text-to-SQL pipeline. DeepEye supplies schema linking, multiple SQL
generators, repair, and final selection. QueryTandem adapts one example-based
generator to use a local Qwen3-4B model and a shorter prompt.

The contribution includes:

- Local MLX quantized LoRA training on a Mac, using 500 BIRD training examples;
  checkpoint 500 and a zero-example prompt were selected on validation data.
- Optional generator model, prompt, and example-count configuration, preserving
  the surrounding DeepEye pipeline and `<result>SQL</result>` format.
- A four-way evaluation with shared preprocessing/candidates, read-only SQL
  scoring, uncertainty estimates, and usage/budget accounting.
- Follow-up routing and selector experiments with explicit rejection criteria.
  Neither follow-up qualified to replace the existing behavior.

Only the local generator runs the small fine-tuned model. Other model-driven
stages used Together. **Do not describe the entire agent as offline or locally
trained, or imply that every final SQL query was written by the local model.**

Suggested short project description:

> I extended DeepEye-SQL with a locally fine-tuned 4B model and compact prompts,
> then measured accuracy and token use across the full pipeline. These recorded
> examples show both successful answers and limitations of the approach.

## 3. Verified claims and their boundaries

Use `demo-data.json` as the interface's single data source. These headline numbers
come from the **original 200-question BIRD dev benchmark**, not the three examples:

| Configuration | Correct | Accuracy | Recorded pipeline tokens |
| --- | ---: | ---: | ---: |
| Original model, 7 prompt examples | 134/200 | 67.0% | 11,436,511 |
| Original model, 0 prompt examples | 135/200 | 67.5% | 10,908,668 |
| Fine-tuned model, 7 prompt examples | 137/200 | 68.5% | 11,507,773 |
| Fine-tuned model, 0 prompt examples | 140/200 | 70.0% | 11,221,739 |

Relative to the original seven-example baseline, the selected configuration used
**69.5% fewer local-generator tokens** and **1.9% fewer recorded pipeline tokens**.
The observed accuracy difference was **+3.0 percentage points**, with a paired
95% database-bootstrap interval of approximately **−0.5 to +7.1 points**.
The interval includes zero; do not claim a proven general accuracy improvement
or a BIRD leaderboard score. These results apply to this configuration and sample.

At the same zero-example prompt, fine-tuning got five more answers right but used
2.9% more pipeline tokens than the original model. Prompt-only used the fewest
tokens. Do not present fine-tuning as uniformly better on every measure.

The selector follow-up is a **separate 100-question validation result**: the
trained selector and existing selector both scored 77/100. It was not adopted,
and its proposed fresh 200-question benchmark was not run. Keep these counts
separate from the original completed benchmark.

Sources: [benchmark JSON](../experiments/compact_icl/results/compact-icl.json),
[selector report](../experiments/compact_icl/results/selector-validation.md),
and [project README](../README.md).

## 4. Data bundle and implementation contract

Download or copy [demo/demo-data.json](demo/demo-data.json) into the website's
normal local data/asset directory. Pin the copy to the same GitHub commit as this
handoff. Do not fetch a moving `main` branch every time a visitor opens the page.
The website needs this JSON, not the Python project or its local workspace.

| JSON field | Use |
| --- | --- |
| `schema_version`, `mode` | Expect version `1` and `recorded_examples` |
| `variant_labels` | Names for `baseline` and `tuned-compact` |
| `benchmark` | Full 200-question counts, tokens, savings fractions, and accuracy interval |
| `selector_followup` | Separate validation outcome and adoption decision |
| `examples[]` | The three selectable recorded cases, in their intended order |
| `examples[].question`, `evidence` | Original question and supplied hint, preserving source wording |
| `examples[].available_schema` | Inference-linked table/column names and types, without sample values |
| `examples[].reference_result` | Result of the reference/gold SQL, for comparison |
| `examples[].variants[variant].final_pipeline_sql` | Final SQL selected by the complete pipeline |
| `examples[].variants[variant].result` | Execution status, column labels, bounded row preview, and complete row counts |
| `examples[].variants[variant].correct` | Verified comparison of complete result sets; never recompute from the preview |
| `examples[].variants[variant].tokens.local_generator` | Saved local ICL input/output/total tokens for this question |
| `examples[].variants[variant].tokens.pipeline` | Saved full-pipeline input/output/total tokens for this question |
| `source`, `metric_notes`, `attribution` | Provenance, measurement definitions, attribution, and data-license notice |

Each token object has `prompt_tokens`, `completion_tokens`, and `total_tokens`.
Show totals initially; input/output breakdowns may be expandable. A missing/null
measurement means **unavailable**, not zero. Fractions such as `0.0187795` display
as percentages after multiplying by 100; accuracy deltas are percentage points.

A result has `status`, `columns`, `rows_preview`, `row_count`,
`distinct_row_count`, `preview_limit`, `truncated`, and `preview_order`.
Render up to ten rows. When truncated, say **“Showing 10 of N rows.”** Preview
rows use a stable display sort; their order does not imply SQL ordering.
`empty` means a successful query with zero rows. `sql_error`, `timeout`,
`row_limit`, and `missing_sql` indicate unavailable results, not empty answers.
Render SQL and all dataset content as plain text, never raw HTML.

Correctness follows this experiment's scorer: full unordered row sets are
compared with gold results, ignoring row order, duplicates, and column aliases.
The supplied label already accounts for that. Round display numbers only;
do not use rounded values or matching previews to decide correctness.

Aggregate usage was reconciled with request ledgers, including retries. The
per-example fields are saved stage/snapshot measurements; they are not a complete
allocation of all retry spending and must not be summed to reconstruct the
headline benchmark. No per-example latency or API-price claim is supplied.

### The three examples

| ID | Outcome | What it illustrates |
| --- | --- | --- |
| `bird-dev-358` | Both correct | Both return `black`; local tokens fall from 1,696 to 358, while pipeline tokens rise from 56,271 to 57,223 |
| `bird-dev-507` | Baseline correct; compact incorrect | An extra join changes the percentage's denominator: approximately 0.8264% versus 1.0920% |
| `bird-dev-341` | Both incorrect | Both return 72 rows; the reference has 67, even though preview rows may look identical |

All three are from the existing `card_games` dev database. They were selected
to illustrate behavior, including failures, rather than to maximize a displayed
success rate. Do not calculate a new “demo accuracy” from these three cases.

## 5. Website behavior and checks

- Discuss placement with Shaoyang first; reuse existing website conventions.
- Make question selection and comparison work with keyboard and touch input.
  Stack comparison panels on narrow screens and contain wide SQL/tables locally.
- Include visible correctness text; do not communicate success/failure by color
  alone. Make schema details and SQL copy controls accessible if included.
- Display selected-example token values separately from benchmark-wide savings.
  Preserve the pipeline-token increase in example 358.
- Show saved results immediately when switching cases. Do not simulate a live
  model with typing, artificial waiting, a “generating” state, or invented traces.
- A conceptual pipeline diagram is acceptable if labeled as an explanation.
  This bundle does not contain a complete recorded intermediate-stage trace.
- Test all three cases, the long SQL/column label, the ten-row truncated preview,
  mobile overflow, keyboard access, source links, and the replay label.
- Confirm the page works without AI API requests, credentials, databases, model
  weights, or a running Mac. Follow the website project's usual preview workflow.

## 6. Attribution and data handling

Keep a short visible credit near the demo or its details:

> QueryTandem Agent, extending HKUSTDial/DeepEye-SQL. Recorded examples from
> BIRD; example data adapted under CC BY-SA 4.0.

Link DeepEye to [upstream](https://github.com/HKUSTDial/DeepEye-SQL), BIRD to
[its project page](https://bird-bench.github.io/), and the data license to
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). BIRD identifies
that data license on its project page. Preserve the attribution/license/change
notice when copying or adapting the example bundle. The example-data notice
is distinct from this repository's MIT code license and the Qwen model's
Apache 2.0 license.

The package contains no API keys, database files, or local filesystem paths.
Source artifact paths inside the JSON are relative provenance identifiers for
ignored files on the experiment Mac; they are not URLs or website dependencies.
The `source.reference_commit` points to the published experiment reference,
not the later commit that adds this handoff. Use the handoff's own commit URL
to pin the website's copy of the data.

## 7. Verification on this Mac and on another computer

The export helper uses only the Python standard library. To reexport or recheck
against the original local snapshots and SQLite database on the experiment Mac:

```sh
python3.12 experiments/compact_icl/export_demo.py
python3.12 experiments/compact_icl/export_demo.py --check
```

To validate the portable bundle without datasets, API access, or dependencies:

```sh
python3.12 experiments/compact_icl/export_demo.py --verify
```

The website does not need to run these commands. It can import the JSON directly.
The exporter verifies reference and predicted queries over complete results
before truncating the displayed rows, using read-only connections, a 30-second
timeout, and a 100,000-row execution limit. It records source-file SHA-256 hashes
and fails if the three predetermined correctness categories change.

## 8. Message to send with the pinned link

Copy the message below and replace the link line with the commit-pinned GitHub
link supplied with this package. Open the website repository on the other
computer, or tell it which repository/folder to use.

> I want to add a small QueryTandem Agent demo to my existing website,
> potentially in the About Me section. Read this handoff and its linked
> `demo-data.json`: **[paste the commit-pinned handoff link here]**.
>
> Please inspect the website project and discuss placement and visual design
> with me, then implement the agreed demo in its existing framework and style.
> Visitors should explore the three recorded questions, compare final SQL,
> answers, correctness, and token usage, and understand the project contribution.
>
> Use the bundled data and metric definitions exactly. Clearly label recorded
> examples, preserve attribution, and include the failures. Do not call an AI
> service, execute SQL, download models/databases, invent metrics, or retrain
> anything. Use the JSON from the same pinned commit as the handoff.
>
> Show me a working preview and verify desktop, mobile, and keyboard interactions.
