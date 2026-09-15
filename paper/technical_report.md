# Counter-Evidence Loss in Retrieval-Augmented Generation

**Bharath Sampadi**  
Independent Researcher

## Abstract

Retrieval-augmented generation (RAG) systems are often evaluated against the evidence visible to the generator. This creates a blind spot when evidence capable of changing an answer exists but does not reach that context.

We tested this with six synthetic claims under three matched evidence conditions. When counter-evidence was exposed, all six claims were answered correctly. When the same counter-evidence remained available but was withheld from the generator-visible context, correctness fell to **0/6**. Removing it from the corpus also produced **0/6** correctness.

Context-anchored evaluation did not distinguish these conditions reliably. Mean RAGAS Faithfulness remained high across EXPOSED, NOT EXPOSED, and ABSENT contexts (**0.917, 1.000, and 0.917**, respectively), despite correctness of **1.00, 0.00, and 0.00**. A reference-free sufficiency judge classified **18/18 contexts as sufficient**, including all contexts associated with incorrect answers.

The **6/6 → 0/6 → 0/6** correctness pattern was independently reproduced with Qwen3.6-27B, a model not used to construct or preflight the benchmark. These results show that local support within retrieved context does not establish that the evidence set itself was adequate.

## 1. Background

RAG conditions generation on retrieved evidence rather than parametric memory alone [1]. Evaluation therefore often asks whether retrieved material is relevant and whether the answer is faithful to that material [2].

That framing has an observational boundary: an evaluator cannot assess evidence it never receives.

Related work approaches different parts of this problem. Glockner et al. showed that missing counter-evidence creates unrealistic assumptions in automated fact-checking [3]. Joren et al. introduced *sufficient context* to distinguish answerable from insufficient retrieved contexts [4]. CUE-R uses evidence interventions such as removal and replacement to measure the operational utility of retrieved evidence [5].

This study isolates a narrower RAG failure: **answer-changing counter-evidence remains available to the system but is not exposed to the generator**. The central question is whether a system and its context-anchored evaluators can distinguish that state from one in which the counter-evidence never existed.

## 2. Methods

### 2.1 Benchmark

The frozen benchmark contains six retained synthetic binary claims: three with predefined TRUE labels and three with predefined FALSE labels. Synthetic entities were used to reduce contamination from real-world or parametric knowledge.

Each claim contains misleading support, answer-changing critical evidence, four neutral passages, and matched replacement material. The frozen benchmark version is `exp1-formal-candidate-v1`.

The public artifact preserves the six retained claims but does not preserve a formal count of drafted or discarded candidate claims; no attrition count is therefore reported.

### 2.2 Preflight and evidence conditions

Before the primary generation experiment, GPT-OSS-20B was used in two anti-leak preflights. Both passed **36/36** checks. The strict preflight additionally verified that file and role labels were not visible to the model.

The preflight was designed to verify that the frozen evidence combinations behaved as intended before formal generation. Consequently, the primary-generator correctness contrast should be interpreted partly as confirmation that the intervention worked as designed rather than as independent evidence of model-general behavior.

Each retained claim was then evaluated under three conditions:

| Condition | Counter-evidence in corpus | Visible to generator |
| --- | :---: | :---: |
| **EXPOSED** | Yes | Yes |
| **NOT EXPOSED** | Yes | No |
| **ABSENT** | No | No |

This produced 18 primary claim-condition instances.

In EXPOSED, the critical passage remained in the generator-visible context. In NOT EXPOSED, it remained indexed and naturally ranked within the top five, but was deliberately removed before the five-document generator context was assembled. In ABSENT, the critical passage was replaced in the corpus by matched replacement material.

The intervention therefore separates **evidence exposure** from **evidence availability**. It is an explicit causal intervention, not a naturally occurring retrieval failure.

### 2.3 Retrieval

Formal retrieval used the SignalRank-RAG retrieval stack:

- `sentence-transformers/all-mpnet-base-v2` embeddings,
- hybrid BM25 + dense retrieval,
- reciprocal-rank fusion with `rrf_k = 60`,
- candidate multiplier `4`,
- FlashRank reranking with `ms-marco-MiniLM-L-12-v2`,
- generator-visible context depth `k = 5`.

For all six claims in the available corpus, the target critical passage ranked **#2** before the NOT EXPOSED intervention. All 18 manipulation checks passed.

### 2.4 Primary generation

Primary answers were generated with:

- **OpenAI GPT-OSS-20B**
- served through **Groq**
- temperature `0`
- no fallback model.

The model was instructed to judge each claim using only the supplied evidence and return a binary TRUE/FALSE verdict.

Correctness was exact agreement with the predefined truth label in the frozen benchmark.

### 2.5 External evaluation

The 18 frozen primary outputs were evaluated with **GPT-OSS-120B** through Groq.

RAGAS Faithfulness measured whether the generated answer was supported by the generator-visible context.

A separate reference-free sufficiency prompt showed the evaluator only the question and generator-visible context. It did not receive the reference answer and returned either SUFFICIENT or INSUFFICIENT.

The formal generator emitted only a binary verdict. For RAGAS evaluation, that verdict was placed in a deterministic natural-language wrapper (`For the factual question '{query}', the answer is {verdict}.`). RAGAS uses the evaluator LLM to decompose this response into statements and then judges each statement against the context. Both steps are LLM calls and were not deterministic across runs at temperature 0: the same response could be decomposed into a different number of statements,[^decomp] and a wrapper-derived statement such as "The answer to the question is true." was sometimes judged unsupported because the context itself contains no question. A single such rejection lowers one case to 0.5 and shifts a six-case condition mean by 0.083. Differences of that size are therefore treated as evaluator variability rather than substantive condition effects.

### 2.6 Independent generator replication

The frozen primary contexts were independently evaluated with:

- **Qwen3.6-27B**
- served through **Groq**
- temperature `0`
- reasoning effort disabled.

This model was not used for stimulus selection or benchmark preflight. The Qwen result is therefore the stronger model-independent check that the exposure manipulation generalizes beyond the model used during benchmark construction.

### 2.7 Restoration analysis

A matched restoration analysis tested whether the faithfulness evaluator could respond once omitted counter-evidence became visible.

First, a statement-level restoration trace evaluated the six frozen incorrect NOT EXPOSED answers against the matched EXPOSED contexts containing the restored critical evidence. The generated answers were held fixed.

A second paired trace reran the corresponding hidden-context baselines using the same RAGAS 0.4.3 evaluator configuration and prompt fingerprints as the restoration trace. This produced a directly matched hidden-versus-restored comparison.

## 3. Results

### 3.1 Counter-evidence exposure determined correctness

| Condition | GPT-OSS-20B | Qwen3.6-27B |
| --- | ---: | ---: |
| **EXPOSED** | **6/6** | **6/6** |
| **NOT EXPOSED** | **0/6** | **0/6** |
| **ABSENT** | **0/6** | **0/6** |

For the six matched EXPOSED→NOT EXPOSED pairs, all six changed from correct to incorrect. An exact two-sided McNemar test gives **p = 0.03125**. Given the purpose-built six-item benchmark, this statistic is descriptive of the paired result and should not be interpreted as evidence about population prevalence.

The primary-generator result confirms the intended manipulation. More importantly, the same pattern was reproduced by Qwen3.6-27B, which was not involved in benchmark selection or preflight.

### 3.2 Faithfulness remained high regardless of correctness

External evaluation produced:

| Condition | Accuracy | Mean RAGAS Faithfulness |
| --- | ---: | ---: |
| **EXPOSED** | **1.00** | **0.917** |
| **NOT EXPOSED** | **0.00** | **1.000** |
| **ABSENT** | **0.00** | **0.917** |

The robust observation is not the small difference between 0.917 and 1.000. Rather, faithfulness remained **at or above 0.917 in all three conditions**, including both conditions in which every generated answer was incorrect.

A matched rerun of the NOT EXPOSED contexts produced a mean faithfulness of 0.917 rather than 1.000, demonstrating run-to-run variability in the evaluator even at temperature 0. In that rerun, the single sub-1.0 case (`false_02`, 0.5) had its content statement judged supported; only the wrapper-derived statement "The answer to the question is true." was judged unsupported. We therefore do not interpret the 0.083 difference between EXPOSED and NOT EXPOSED as a meaningful increase.

### 3.3 The sufficiency judge classified every context as sufficient

The reference-free sufficiency judge classified:

**18/18 contexts as SUFFICIENT.**

This included all 12 NOT EXPOSED and ABSENT contexts associated with incorrect answers.

The censored contexts contained enough coherent evidence to support a definite answer, but not enough evidence to establish whether the visible evidence set itself was adequate.

### 3.4 Restoring counter-evidence made the hidden conflict observable

In the matched six-case trace, mean faithfulness decreased from **0.917 to 0.000** after omitted counter-evidence was restored to the evaluator-visible context.

Faithfulness decreased in **all six paired cases**.

At the statement level, all **12/12** statements evaluated against the restored contexts were judged unsupported. For **10/12**, the evaluator's stated reason cited the restored corrective evidence (for example, an audited value superseding the earlier figure). The remaining **2/12** were wrapper-derived statements (`false_02`, `false_03`) rejected because the context contains no question, not because of the restored evidence. Every case nevertheless had its content statement rejected on the basis of the restored evidence.

The matched trace is a separate evaluator execution from the external-metric run, and its 0.917 hidden-context baseline reflects evaluator run-to-run variability. The result is therefore interpreted only as a **within-trace paired change**.

Because the generated answers were held fixed, the restoration result shows that the evaluator can react strongly to counter-evidence once that evidence enters its observational boundary.

## 4. Discussion

The central finding is not simply that removing corrective evidence can change an answer. The benchmark was intentionally constructed so that the support and critical passages pull the answer in opposite directions, and the primary exposure intervention was manual.

The more important systems-level observation is that **NOT EXPOSED and ABSENT become operationally indistinguishable from the perspective of the generator and context-only evaluators**. In one case the corrective evidence exists in the indexed corpus but is lost before generation; in the other it does not exist in the corpus at all. Yet both produce the same 0/6 correctness outcome, uniformly positive sufficiency judgments, and high local faithfulness.

This suggests a boundary on context-only evaluation. A RAG system can answer consistently from the evidence it sees without revealing whether relevant counter-evidence was lost upstream.

A useful evaluation stack therefore needs to distinguish:

1. **Local faithfulness** — is the answer supported by the visible context?
2. **Evidence adequacy** — did retrieval and context construction expose the evidence needed to challenge or qualify that answer?

The first does not guarantee the second.

## 5. Limitations

This is a proof-of-concept study with **six synthetic claims**.

The benchmark was preflighted using the same GPT-OSS-20B model used for primary generation. The primary 6/6 → 0/6 contrast therefore partly verifies a purpose-built manipulation. The independent Qwen3.6-27B replication reduces, but does not eliminate, concerns about benchmark construction.

All six target critical passages ranked #2 before intervention, and the NOT EXPOSED condition manually removed them before context assembly. The study therefore does not show that this failure occurred naturally under the tested retrieval stack. Future work should test naturally occurring exposure loss through rank cutoffs, token budgets, query rewriting, or larger heterogeneous corpora.

The critical passages use deliberately clear corrective evidence. More subtle contradictions that do not explicitly announce correction or override would provide a harder test.

The external evaluation uses one evaluator model and one sufficiency prompt. RAGAS Faithfulness also showed run-to-run variability on identical inputs. Larger benchmarks, natural corpora, multiple retrieval stacks, and multiple evaluators are needed before making broader claims.

The public repository preserves the frozen benchmark, outputs, and experiment scripts, but the retrieval runners depend on the separate SignalRank-RAG codebase and are not a standalone software package.

## Appendix A. Prompt templates

### Primary generator

System prompt:

```text
You are a claim-verification system.

Judge the claim using ONLY the supplied evidence.
Do not use outside knowledge.

Return TRUE if the supplied evidence supports the claim.
Return FALSE if the supplied evidence contradicts the claim.

You must return one binary verdict.
```

User prompt template:

```text
Claim:
{query}

Evidence:
{five ranked evidence passages}

Determine whether the claim is TRUE or FALSE based only on the supplied evidence.
```

### Reference-free sufficiency judge

System prompt:

```text
You are a blind context-sufficiency judge.

You receive only a question and the context supplied to a RAG generator.
You do NOT know the reference answer and must not use outside knowledge.

Return exactly ONE word:
SUFFICIENT
or
INSUFFICIENT

Use SUFFICIENT if the supplied context appears to contain enough direct
evidence to give a definite answer to the question from the context alone.

Use INSUFFICIENT if the context does not appear to contain enough evidence
to give a definite answer.

Important:
- Judge apparent answerability from the supplied context only.
- Do not ask whether hidden evidence might exist elsewhere.
- Do not compare against a reference answer.
```

[^decomp]: For example, the frozen `false_03` response was decomposed into one statement when evaluated against the hidden context in the paired trace, but into two statements when evaluated against the restored context in the same trace and in the restoration trace.

## References

1. Lewis, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** NeurIPS, 2020.
2. Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. **RAGAs: Automated Evaluation of Retrieval Augmented Generation.** EACL System Demonstrations, 2024.
3. Glockner, M., Hou, Y., & Gurevych, I. **Missing Counter-Evidence Renders NLP Fact-Checking Unrealistic for Misinformation.** EMNLP, 2022.
4. Joren, H. et al. **Sufficient Context: A New Lens on Retrieval Augmented Generation Systems.** ICLR, 2025.
5. Jain, S. & Vedam, V. N. **CUE-R: Beyond the Final Answer in Retrieval-Augmented Generation.** arXiv:2604.05467, 2026.
