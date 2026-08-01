# Drishtikon-40B — Plan for an India-First 40B Foundation Model Report

## Context

This is an ML-systems-design submission (course assignment). The deliverable is a single-page, netlify-style interactive report ("solution") arguing a fully-specified plan for pretraining + post-training + aligning a 40B-parameter model that (a) reaches Gemma-4 parity, (b) excels at coding and agentic tool-use, (c) is genuinely fluent in India's 22 scheduled languages, and (d) defaults to an Indian frame of reference. The user explicitly asked to be evaluated on *depth of thought*, penalized for *length*, and — critically — to be **creative/novel enough that ideas could seed real conference papers**, not just replay the standard playbook.

The user supplied 4 existing submissions as inspiration and required that anything reused from them be explicitly attributed in the report. **This phase is plan-only**: no report/UI files get built yet; this document locks every number and every novel idea so the build phase is just execution.

### What I did to prepare
- Read all 4 reference submissions in full (fetched live).
- Web-searched to ground the plan in **real, current (2026) facts** rather than guesses — this is itself a differentiator, since all 4 references invented plausible-sounding numbers for a hypothetical "Gemma-4."

### Key real facts gathered (to cite in the report)
- **Gemma 4 is real**, released April 2026 (12B shipped June 2026), sizes E2B/E4B/12B/26B-MoE(3.8B active)/31B-dense, Apache-2.0, 128K–256K context, trained through Jan 2025. Real 31B-dense scores: **MMLU-Pro 85.2%, GPQA 84.3%, LiveCodeBench 80.0%, AIME 2026 89.2%**. → use these as the literal parity bar, not invented targets. [Gemma 4 overview](https://www.labellerr.com/blog/gemma-4-open-weight-ai-model-overview/), [benchmarks](https://gemma4-ai.com/blog/gemma4-benchmark)
- **AI4Bharat Sangraha**: real 251B verified/OCR/ASR tokens across 22 languages — the honest ceiling on "clean, authentic" Indic supply today. [HF dataset](https://huggingface.co/datasets/ai4bharat/sangraha)
- **MILU** (NAACL 2025, 11 Indic languages, 8 domains/42 subjects): GPT-4o tops out at **72%** average, weakest in Arts/Humanities and Law/Governance — the exact "India-context knowledge" gap to target. [arXiv:2411.02538](https://arxiv.org/abs/2411.02538)
- **Sarvam-1**: 68,096-vocab tokenizer, fertility **1.4–2.1** across supported languages, translated-MMLU 44.44 — the "small, cheap, Indic-first" baseline to beat. [Sarvam-1 blog](https://www.sarvam.ai/blogs/sarvam-1)
- **BrahmicTokenizer-131K** (arXiv, May 2026): 131,072-vocab drop-in retrofit of `o200k_base`; matches English fertility (1.235 vs 1.232); **26.7% fewer tokens** than Tekken/Sarvam-m on Indic text; per-language gains from **15.79% (Tamil, hardest case) to 76.79%/4.31× (Odia, easiest case)**; drops 9 non-Brahmic scripts (so Urdu/Kashmiri/Sindhi/Santali are explicitly out of scope for it). [arXiv:2605.29379](https://arxiv.org/pdf/2605.29379)
- **IndicSuperTokenizer** (OpenReview, Oct 2025): two-stage curriculum — whitespace-BPE then unconstrained "superword" multi-word merges — beats LLaMA-4 fertility by 39.5%, Sutra by 18%, 44% inference-throughput gain. [OpenReview](https://openreview.net/forum?id=CSrGFB070m)
- No existing agentic benchmark targets India's Digital Public Infrastructure (UPI/DigiLocker/ONDC/CoWIN/e-Shram) — confirmed genuine gap via search.

### What the 4 references already cover (attribute as "common/expected baseline," not claim as novel)
All four independently converge on: fertility-tiered tokenizers (200–256K vocab), an "always-on floor" for scarce-language sampling, OCR/ASR pipelines for Indic data manufacturing, per-language (never averaged) eval reporting, DPDP-Act-compliant PII scrubbing, India-Stack APIs (UPI/Aadhaar/ONDC) as code-training domain data, a constitutional/pluralism alignment layer, and a bespoke "IndiaBench"-style custom eval. These are legitimate best practices — the report will use them as the *baseline* layer, credited generically to the 4 references, and spend its real word-budget on 5 genuinely new contributions below.

---

## Locked Decisions (the numbers)

**Model name:** *Drishtikon-40B* (दृष्टिकोण, "point of view/perspective") — ties directly to the "India-first worldview" thesis and to the novel eval metric below. 40B dense (not MoE) — kept dense for auditability/on-prem sovereign deployment, clean comparability to Gemma-4-31B-dense.

**Scale:** 14T pretraining tokens (~350 tokens/param) — anchored to the real Gemma-3-generation precedent (~14T tokens for 27B) rather than an arbitrary pick, since a 40B model carrying a much larger multilingual+code+agentic surface than Gemma's mix needs at least that ratio. Post-training: 25B SFT tokens, 10B DPO/preference-pair tokens, 20B RL/agentic rollout tokens (incl. DPI-Gym rollouts).

### Pretraining data mix (14T tokens)
| Category | Share | Why |
|---|---|---|
| English (edu-filtered) | 30% | Reasoning/knowledge backbone for Gemma-4 parity |
| Code (repo-level + India-Stack SDKs) | 20% | Coding/agentic parity target (LiveCodeBench ≥80) |
| Indic — 22 languages (Sangraha-anchored + Oral-Commons, see below) | 20% | Pushed above all 4 references' 16–18% — backed by real 251B verified floor + new oral-audio channel |
| Math & science (NCERT-anchored CoT + olympiad + arXiv) | 8% | AIME/JEE parity |
| India-context civic/legal ("Panchayat-to-Parliament" corpus — gram panchayat minutes → municipal → state assembly → Parliament records, all in-language) | 7% | Closes MILU's real Law/Governance/Arts gap; deeper federal-tier granularity than any reference's "law + gazettes" bucket |
| Parallel/cross-lingual (Samanantar, BPCC, FLORES + Rupantar synthetic pairs, see below) | 5% | Alignment across scripts/languages |
| Agentic/tool-use trajectories (DPI-Gym + general tool corpora) | 5% | Agentic parity, India-grounded |
| World multilingual (non-Indic, ~140 langs, small share) | 3% | Global competence, not the focus |
| Long-context/books/structured docs | 2% | Long-context capability |

### Cleaning pipeline (6 stages — kept short; novel filters flagged)
1. Extraction + script-invariant language ID (robust to code-mixing)
2. **Transliteration-canonicalized dedup** *(novel, see below)* — catches cross-script duplicates hash/MinHash dedup misses
3. Per-language quality filtering + Unicode NFC normalization
4. PII removal (DPDP Act 2023: Aadhaar/PAN/UPI patterns + Indic NER)
5. Safety + **Nirapeksh consistency check** *(novel, see below)* on identity-bearing sentences
6. Benchmark decontamination (12-gram/fuzzy overlap strip vs every eval below) + memorization audit gate before release

### Evaluation framework (real benchmarks as the floor, novel ones for the hard-to-measure objectives)
| Objective | Benchmark | Target |
|---|---|---|
| Gemma-4 parity | MMLU-Pro / GPQA / AIME 2026 | ≥85.2 / ≥84.3 / ≥89.2 (real Gemma-4-31B scores) |
| Coding | LiveCodeBench, SWE-bench Verified | ≥80.0 / competitive-verified-resolve |
| Agentic | BFCL v4, τ-bench, **DPI-Gym** *(novel)* | ≥ open-model SOTA; DPI-Gym is the new bar |
| Indic knowledge | MILU, IndicGenBench, FLORES (never averaged) | ≥75% MILU avg (beats GPT-4o's real 72%), explicit Arts/Law sub-score target |
| Code-mixed coding help | **CodeMix-Bench** *(novel, brief)* | native-rater ≥4.3/5 |
| India-first worldview | **Drishtikon-Bench** *(novel, see below)* | displacement toward Indian frame with a factuality floor, not just accuracy |
| Tokenizer fairness | Fertility vs. BrahmicTokenizer-131K/Sarvam-1 (held out) | see fertility table below |
| Safety/fairness | disaggregated caste/religion/region/gender probes | no single axis <90% of best axis |

### Fertility targets → tokenizer vocab size
Total vocab: **~180K** (justified against real prior art, not a round guess like the 4 references' 200K/256K):
- Latin/English/European + code: 72K (preserve `o200k_base`-class code compression, per BrahmicTokenizer's finding that this is achievable without loss)
- Brahmic core (9 Unicode blocks, LP-allocated as in BrahmicTokenizer-131K, scaled up): 70K
- **Non-Brahmic Indian scripts (Perso-Arabic Urdu/Kashmiri/Sindhi, Meitei Mayek, Ol Chiki)**: 14K — *the gap BrahmicTokenizer-131K explicitly dropped*; restoring it is required for a genuinely 22-language model
- Superword/multi-word merge reserve (IndicSuperTokenizer-style stage-2): 10K
- Math/LaTeX/atomic-digits/structured/special+tool-call tokens: 6K
- Matryoshka elastic reserve (nested sub-vocab for edge/low-RAM deployment): 8K

| Tier | Languages | Fertility target | Grounding |
|---|---|---|---|
| English/code | — | ≤1.25 / within 2% of o200k_base | measured 1.235 real baseline |
| Indo-Aryan core | Hindi, Marathi, Gujarati | 1.5–1.7 | vs Sarvam-1's 1.4–2.1 |
| Eastern Nagari-derived | Bengali, Assamese, **Odia** | 1.6–1.8 (Odia →1.5) | BrahmicTokenizer's Odia result was its *easiest* win (76.79%/4.31×) |
| Dravidian (agglutinative) | Tamil, Telugu, Kannada, Malayalam | 1.85–2.1 (down from typical 2.3–2.6) | BrahmicTokenizer's Tamil was its *hardest* case (only 15.79%) — exactly what Sandhi-BPE below targets |
| Gurmukhi | Punjabi | 1.7–1.9 | — |
| Perso-Arabic | Urdu, Kashmiri, Sindhi | 1.9–2.2 | restored script budget (see above) |
| Tier-3 / low-resource | Santali, Manipuri, Bodo, Maithili, Konkani, Dogri, Nepali, Sanskrit | ≤2.8, zero-UNK via byte-fallback | Oral-Commons-fed |
| Math | — | 1.0 tok/atomic digit, digits never merged | — |
| Agentic schema | tool-call JSON keys/function names | ~1.0 (protected atomic vocab) | — |

**Language tiers for training-time attention/investment** (not just tokenizer): Tier-0 (Hindi, Bengali, Tamil, Telugu, Marathi — bridges Indo-Aryan/Dravidian, largest combined L1+L2 population) get full parity-level SFT/RLHF investment; Tier-1 (Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu, Assamese) get full pretraining + lighter native-rater alignment; Tier-2 (remaining scheduled languages incl. Sanskrit) rely on the always-on floor + Oral-Commons + byte-fallback safety net.

---

## AWS Cost Estimate (grounded in real July-2026 pricing)

**Compute (pretraining):** FLOPs ≈ 6·N·D = 6 × 40B × 14T = **3.36 × 10²⁴ FLOPs**. At 40% MFU on H100 SXM (989 TFLOPS bf16 peak → 3.96 × 10¹⁴ effective FLOPs/s/GPU) on a 4,096-GPU cluster (512 × `p5.48xlarge`): **≈24 days**, **≈2.36M GPU-hours**.
- On-demand (`p5.48xlarge` ≈ $6.88/GPU-hr post the 2025 AWS P5 price cut): **≈$16.2M**
- EC2 Capacity Blocks, reserved (confirmed July-2026 rate, $5.191/accelerator-hr): **≈$12.2M**
- +15–20% for EFA networking, FSx for Lustre, checkpoint/S3 storage, egress → **pretraining subtotal ≈$14.5M (reserved) to $19M (on-demand)**

**Post-training + RL/agentic (incl. DPI-Gym rollouts):** SFT/DPO FLOPs are <1% of pretraining; the real cost driver is RL rollout/inference volume (multiple sampled trajectories per prompt + reward/verifier calls) plus DPI-Gym's CPU-bound mock-API fleet (cheap, non-GPU). Budget **≈$1.5–2.5M**.

**Data pipeline (OCR/ASR/dedup/quality classifiers/embedding dedup) + storage:** OCR of scanned print archives and ASR over Oral-Commons audio are the dominant cost here (smaller GPU instances, not H100s), plus S3 storage across raw→cleaned passes. Budget **≈$3–5M**.

**Eval/monitoring across training checkpoints:** **≈$0.3–0.5M**.

**Contingency (job restarts/failures at this scale — standard for multi-week, thousand-GPU runs):** +10–15%.

### Headline: **≈$18M–$26M total AWS bill** for the full pretrain → post-train → eval program (pretraining GPU time is ~70–75% of it).

**Assumptions stated explicitly (so the number is auditable, not a black box):** 40% MFU, 4,096 H100s, 14T tokens, blended Capacity-Block + on-demand pricing. Total GPU-hours (and therefore cost) is roughly invariant to cluster size — 2,048 GPUs for ~48 days costs about the same as 4,096 for ~24 days, with a possible small MFU edge from simpler interconnect topology; the larger cluster just buys calendar time. **Procurement note:** self-service EC2 Capacity Blocks cap at 512 GPUs per block, so reaching 4,096 GPUs means stacking 8 blocks or moving to an AWS enterprise custom-cluster agreement — worth flagging as a real logistics step, not just a pricing line. **Cost lever to evaluate (not the default plan):** AWS Trainium2 (Trn2) UltraServers are marketed at meaningfully better price-performance than H100 for training; worth a small-scale pilot, but H100/CUDA has far more proven large-scale multilingual-LLM training recipes, so it stays the primary path unless Trn2 pilots de-risk it.

---

## Data Licensing & IP Risk (hard requirement: model must be freely usable by others, professionally)

**Real, current fact that none of the 4 reference submissions engaged with:** India's Copyright Act, 1957 has **no explicit text-and-data-mining exception** (unlike EU DSM Art. 4, Japan Art. 30-4, Singapore's TDM exception). This is *not* a theoretical gap — it is being litigated right now: **ANI Media v. OpenAI**, Delhi High Court (filed Nov 2024, 32 hearings through March 2026, order reserved by Justice Amit Bansal). The court has framed the exact questions that matter for this project: does storing copyrighted text to train a model infringe; does generation infringe; does Section 52 "fair dealing" cover AI training at all. Two court-appointed amici curiae gave **opposing opinions**. Treat this as unresolved, not settled in AI builders' favor.

**Sourcing policy, tiered by legal certainty:**
1. **Safe tier (prioritize):** explicitly permissive-licensed data — **Sangraha is confirmed CC-BY-4.0** (commercial use OK with attribution); Indian court judgments (copyright-free under Section 52(1)(q) — the one unambiguous public-domain-equivalent government-text category); CC0/public-domain/openly-licensed corpora.
2. **Permission-required tier:** other government works (NCERT textbooks, gazettes, ministry reports) — under Indian law the **government retains copyright** on these (Section 2(k)), unlike the US public-domain treatment of federal works. Do not assume "government = free." Requires an actual data-sharing MoU with the issuing body, not just scraping the public site.
3. **Contested tier (cap exposure, don't build the model's foundation on it):** general web crawl / news / blogs — exactly what ANI v. OpenAI is about. Keep this share modest, and architect for exit: every training shard is tagged in a **provenance ledger** (source, license, collection date, content hash) so a specific publisher's content can be surgically excised and the model continued-pretrained without it if the ruling requires — cheaper and faster than discovering this risk after the fact and needing a full retrain.
4. **Code:** permissive-SPDX only (MIT/Apache-2.0/BSD), excluding GPL/AGPL — standard BigCode/StarCoder precedent, avoids ambiguous copyleft obligations flowing into model outputs.
5. **Parallel corpora (Samanantar/BPCC etc.):** audit per-release license individually before ingestion — AI4Bharat's various releases are not uniformly licensed; default posture is include only if CC-BY/CC0/ODC-By-equivalent, exclude anything tagged research-only/non-commercial.

**Model weight license: Apache 2.0**, not a custom "responsible-use" license. This is the actual point that satisfies "used by others professionally" — Apache-2.0 is OSI-approved with no field-of-use restriction and no monthly-active-user threshold (unlike Meta's Llama Community License) and no acceptable-use-policy-as-license-condition friction (unlike Google's Gemma Terms). Pair it with a separate, non-binding Responsible Use Guide for safety norms, kept out of the license itself so it creates no legal friction for commercial adopters.

---

## The 5 novel, paper-seedable contributions (the actual differentiators — report should spend most of its words here)

1. **DPI-Gym** — a simulated-environment suite (à la WebArena/OSWorld), built around India's actual Digital Public Infrastructure, that is simultaneously an SFT/RL trajectory generator and a held-out benchmark:
   - **Environments:** UPI (payment initiation/collect-request/dispute flows), DigiLocker (fetch/issue/e-sign documents), ONDC (buyer-app + seller-app sides — **ONDC's protocol specs are genuinely open, confirmed on GitHub at `ONDC-Official/ONDC-Protocol-Specs`**, so a simulator built on the public spec is low legal risk), CoWIN (legacy-API deprecation handling), e-Shram (worker registration/benefit lookup), Aadhaar-eKYC (OTP/consent flows), FASTag, and the RBI Account Aggregator/DEPA framework (consent-based financial data sharing — rich ground for agentic financial tasks).
   - **Realism knobs that make this harder than WebArena:** injected rate-limiting, OTP timeouts, payment idempotency requirements, vernacular user utterances that must map to correct API parameters, and simulated 2G/low-bandwidth conditions forcing retry/checkpoint/USSD-fallback behavior — this models real Indian connectivity constraints, not a clean-network toy.
   - **Reward:** RLVR-style, execution-verified against the simulator's resulting ledger/DB state (SWE-bench-style gold-trajectory checking), not "looks plausible."
   - **Safety/licensing:** simulated sandboxes only, never live production APIs or real user data — zero PII exposure risk.
   - **Paper framing:** first benchmark grounding agentic LLM evaluation in a nation-scale digital-public-infrastructure stack; the methodology (simulate a country's DPI stack, inject its real operating constraints) directly generalizes to Singapore's SingPass, Brazil's Pix, Estonia's X-Road — a strong "broader impact/generalizability" argument for a paper, not a one-off India toy benchmark.

2. **Sandhi-BPE** — morphology-seeded superword tokenizer for the Dravidian languages, where fertility gains have historically been hardest (BrahmicTokenizer-131K's own reported numbers show Tamil as its weakest case, only 15.79% improvement vs Odia's 76.79%). **It is one single shared tokenizer/vocabulary — not a different tokenizer per language.** Concretely:
   - **One merge table, one vocab file** (the full ~180K budget: Latin/code 72K + Brahmic-core 70K + non-Brahmic scripts 14K + superword/math/reserve), applied identically to every language and to code — exactly like a normal multilingual BPE (Llama's, `o200k_base`'s). This is required, not optional: Hinglish/code-mixed sentences interleave scripts *within one sentence*, so separate per-language tokenizers would break on the most common real-world Indian text.
   - **Devanagari is fully covered** — it sits inside the Brahmic-core 70K budget alongside the other 8 Brahmic Unicode blocks, inheriting BrahmicTokenizer-131K's LP-allocation approach (Devanagari was one of its explicit target blocks). Hindi, Marathi, Nepali, Sanskrit, Konkani, and optionally Bodo/Maithili/Dogri all use this same shared Devanagari allocation.
   - **What actually differs per script/family is only the *training-time merge-selection procedure*, not the runtime artifact:** (a) per-script pretokenization boundary rules (matra/vowel-sign placement, RTL joining for Perso-Arabic) — same idea IndicSuperTokenizer already uses; (b) for the 4 Dravidian languages specifically (Tamil, Telugu, Kannada, Malayalam — agglutinative, heavy noun-case/verb-agreement stacking), an existing Dravidian morphological analyzer supplies gold morpheme boundaries that *bias* which candidate merges get selected during vocab construction, instead of pure pair-frequency — so the vocab learns reusable morpheme-respecting sub-words instead of memorizing whole compositional word forms; (c) an IndicSuperTokenizer-style "superword" stage specifically for common Hinglish/Tanglish code-switch boundary patterns.
   - **Fallback/risk mitigation:** the morph-analyzer bias is a *soft prior* gated by analyzer confidence, not a hard constraint — where confidence is low it silently falls back to plain frequency-based BPE, so worst case reproduces BrahmicTokenizer/IndicSuperTokenizer behavior with zero regression, and the analyzer dependency never blocks training.
   - At inference/deployment time: exactly one tokenizer object, applied uniformly, no per-language branching, no runtime detection logic required.
3. **Rupantar Chains** (रूपांतर, "transformation/adaptation") — synthetic data generation that goes beyond every reference's plain back-translation-with-COMET/chrF-gating. Takes an English fact/reasoning chain → translates → **a culturally-grounded rewriter re-expresses it in Indian idiom/units/analogy (transcreation, not translation)** → NLI-verified for faithfulness back to the source. Produces data that's natively Indian in *framing*, not just in *language* — directly targets "India-first" where mere language coverage doesn't.
4. **Drishtikon-Bench** — the model's namesake contribution. Instead of accuracy-style worldview probes (what all 4 references do), measures a **"perspective displacement score"**: KL-divergence between the model's answer distribution on ambiguous/ethnocentric-prone questions (discovery attribution, historical framing, "typical" X) versus a Western-default baseline model, gated by a factuality floor so displacement toward an Indian frame never overshoots into inaccuracy or majoritarianism. Turns "India-first" from a vibe into a number.
5. **Samvidhan-CAI** (संविधान, "constitution") — a Constitutional-AI-style critique/revision RLAIF loop where critique prompts are drawn from *specific, named Articles* of the Indian Constitution (14, 15, 19, 25, 29, 30 — equality, expression, religious freedom, minority/cultural rights) and Directive Principles, each with legal-linguist-validated native-language templates. More rigorous and reproducible than every reference's generic "constitutional framing" gesture.

**Tier-2 (name-drop only, ~1 line each in the report, not fully elaborated — keeps length down):** Nirapeksh Loss (निरपेक्ष, counterfactual caste/religion/region name-swap consistency regularizer applied at training time, not just eval-time probing); Bandit-DoReMi (multi-armed-bandit online data reweighting with hard floors, extending DoReMi for extreme long-tail languages); Oral-Commons (ASHA/community-radio-partnered active audio elicitation + ASR for PVTG/tribal languages with near-zero digital text, feeding continual pretraining rather than one-time scraping); Matryoshka vocab nesting (deployment-elastic sub-vocabularies for low-RAM/feature-phone inference); CodeMix-Bench (Hinglish/Tanglish code-mixed programming-help benchmark); transliteration-canonicalized dedup (already listed in cleaning pipeline).

---

## Report / UI build plan (next phase — not executed yet)

- **Format:** single-page scrolling HTML/CSS/JS site, visually original (do not copy the 4 references' layouts) — sticky in-page nav across: Hero/Thesis → Data Mix → Cleaning → Tokenizer & Fertility → **Cost (AWS)** → **Licensing/IP Risk** → Evaluation → **5 Novel Contributions (visually distinguished, e.g. a badge/accent treatment; DPI-Gym and Sandhi-BPE get the most space)** → Language Tiers → Attribution/Inspiration footer → (optional) Risks & open questions.
- **Word budget:** target ≤ ~3,000–3,500 words total content (nudged up slightly from the original ≤2,500–3,000 to fit the new Cost and Licensing sections without starving the novel-contributions section) — explicit discipline against the stated length penalty. Tables/charts carry numeric density so prose can stay tight.
- **Visualizations** (invoke the `dataviz` skill at build time, not now): data-mix donut/stacked-bar, fertility-by-tier bar/heatmap, vocab budget stacked bar. Keep to 2–3 charts max, not one per section.
- **Attribution section (explicit, per user's requirement):** a visible "Inspiration & Attribution" block listing the 4 source URLs and naming exactly which baseline ideas (fertility-tiered tokenizer, always-on floor, OCR/ASR manufacturing, non-averaged per-language eval, DPDP PII scrubbing, India-Stack code data, constitutional-alignment gesture, custom IndiaBench-style eval) are common/inspired-by vs. which 5 (+ tier-2) are this report's own novel contributions.
- **Tone:** written as a memo from "a team of top-tier data scientists / infra engineers / ML researchers / Indic-language experts / a business executive" per the user's framing — confident, decisive, numbers-first, not hedgy.
- **Hosting:** per explicit instruction, build the artifact/file only — do not deploy to Netlify or any external host this round.

### Verification (once built)
- Every number in the tables above must appear consistently in the final prose (no drift between plan and report).
- Word count check against the ≤3,500 budget.
- Confirm attribution footer explicitly names all 4 source URLs and correctly separates "baseline/common" vs. "novel" ideas.
- Spot-check that all cited real facts (Gemma-4 scores, Sangraha 251B/CC-BY-4.0, MILU 72%, Sarvam-1 68,096/1.4–2.1, BrahmicTokenizer-131K numbers, IndicSuperTokenizer numbers, AWS P5/Capacity-Block July-2026 pricing, ANI v. OpenAI case status) are reproduced accurately from the sources gathered in this session.
- Cost section's arithmetic (FLOPs → GPU-hours → $) must be re-derivable from the stated assumptions (40% MFU, 4,096×H100, 14T tokens) — no unexplained numbers.
- Licensing section must state the Apache-2.0 weight-license decision explicitly and must not claim any government work other than court judgments as "public domain."
