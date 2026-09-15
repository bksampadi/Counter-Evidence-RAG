# Counter-Evidence Loss in Retrieval-Augmented Generation

What happens when evidence capable of correcting an answer exists, but does not reach the generator?

<p align="center">
  <img src="figures/main_result.svg" alt="Main result" width="820">
</p>

Across six matched claims, correctness fell from **6/6** when counter-evidence was exposed to **0/6** when the same evidence remained available but was withheld from the generator-visible context. Removing the counter-evidence from the corpus also produced **0/6** correctness.

Context-anchored evaluation did not distinguish the failure reliably. Mean RAGAS Faithfulness remained high across all three conditions (**0.917 / 1.000 / 0.917**), and a reference-free sufficiency judge classified **18/18 contexts as sufficient**.

The same **6/6 → 0/6 → 0/6** correctness pattern was reproduced with **Qwen3.6-27B**, which was not used to construct or preflight the benchmark.

**Technical report:** [`paper/technical_report.md`](paper/technical_report.md)

## Reproducing

All frozen stimuli and outputs are in `benchmarks/` and `benchmark_results/`. Scripts are run from the repository root with `GROQ_API_KEY` set.

```bash
pip install -r requirements.txt
python scripts/Exp1/prepare_formal_corpora.py
```

| Script | Needs SignalRank-RAG |
| --- | :---: |
| `scripts/Exp1/prepare_formal_corpora.py` | No |
| `scripts/replication/replicate_e2_independent_generator.py` | No |
| `scripts/validation/run_counter_evidence_restoration_trace.py` | No |
| `scripts/validation/run_counter_evidence_baseline_trace.py` | No |
| `scripts/Exp1/run_formal_exp1.py` | Yes |
| `scripts/Exp2/run_formal_exp2.py` | Yes |
| `scripts/validation/validate_external_metrics.py` | Yes |

The SignalRank-dependent scripts were run against [SignalRank-RAG](https://github.com/bksampadi/SignalRank-RAG) 0.2.0 at commit `bc6dc9d`. Hosted Groq models may change over time, so reruns are not guaranteed to reproduce outputs exactly.

## License

Code in `scripts/` is released under the [MIT License](LICENSE). The benchmark, results, figures and paper are released under [CC BY 4.0](LICENSE-DATA).

This is a proof-of-concept study using six synthetic claims. The NOT EXPOSED condition is an explicit exposure intervention; the study isolates a failure mechanism rather than estimating its prevalence in deployed RAG systems.
