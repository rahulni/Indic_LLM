# Drishtikon-40B (दृष्टिकोण) — India-First 40B Model Design Memo

Report: [`drishtikon-40b-report.html`](./drishtikon-40b-report.html) — open directly in a browser.
Planning log / backing research: [`plan-notes.md`](./plan-notes.md).

## Inspiration & attribution

Four prior submissions were used as inspiration, per the assignment's instructions:

- → [gaurav-eagv5-s3.netlify.app](https://gaurav-eagv5-s3.netlify.app/) ("Akshara-40B")
- → [illustrious-licorice-50be3e.netlify.app](https://illustrious-licorice-50be3e.netlify.app/)
- → [yasir-s3.netlify.app](https://yasir-s3.netlify.app/) ("The 40B India-First Blueprint")
- → [candid-lollipop-8a0195.netlify.app](https://candid-lollipop-8a0195.netlify.app/) ("Bharat-40B")

All four independently converge on a shared baseline playbook. Every specific idea reused from them is itemized below, rather than gestured at generically.

| Idea / practice | Source(s) | What we did with it |
|---|---|---|
| Fertility-tiered tokenizer targets | all four | kept the format; re-derived the numbers from real 2026 papers |
| 200–256K vocab as the default sizing instinct | Akshara, Blueprint, Bharat-40B | same instinct, landed on ~180K via BrahmicTokenizer-131K's real result |
| "Always-on floor" bypassing quality filters for scarce languages | Akshara (named exactly) | reused directly, §8 Tier-2 in the report |
| OCR + ASR as scarce-Indic-data manufacturing | Akshara, illustrious-licorice, Bharat-40B | reused as baseline; **Oral-Commons** (active ASHA/radio elicitation) is our extension |
| Per-language, never-averaged eval reporting | Akshara, illustrious-licorice, Bharat-40B | reused directly |
| DPDP Act PII patterns (Aadhaar/PAN/UPI) | illustrious-licorice, Blueprint, Bharat-40B | reused near-verbatim, cleaning stage 4 |
| India-Stack APIs as code-training data | illustrious-licorice, Bharat-40B | reused as baseline; **DPI-Gym** builds a full simulated benchmark on top |
| Constitutional-AI-style pluralism/secularism alignment | illustrious-licorice, Blueprint, Bharat-40B | reused the concept; **Samvidhan-CAI** grounds it in named Constitution Articles |
| Custom India-specific worldview benchmark | all four ("IndiaBench" in Bharat-40B) | reused the concept; **Drishtikon-Bench** makes it a quantitative displacement score |
| Samanantar / BPCC / FLORES parallel corpora | illustrious-licorice, Bharat-40B | reused directly |
| AI4Bharat Sangraha as anchor Indic corpus (incl. real 251B figure) | Bharat-40B | reused the same real dataset/number |
| Execution-verified (RLVR-style) coding/agentic rewards | Akshara, Bharat-40B | reused as baseline; DPI-Gym extends it to a new environment class |
| 12-gram overlap for decontamination | Blueprint, Bharat-40B | reused the identical technique/threshold |
| MinHash/LSH deduplication | illustrious-licorice, Blueprint, Bharat-40B | reused as baseline; **transliteration-canonicalized dedup** is our extension |
| BFCL / τ-bench agentic eval | illustrious-licorice, Blueprint, Bharat-40B | reused directly; DPI-Gym added on top |
| MILU / IndicGenBench / FLORES eval suite | Akshara, illustrious-licorice, Bharat-40B | reused directly as our eval floor |
| NCERT / JEE / NEET / UPSC as India-context sources | illustrious-licorice, Blueprint, Bharat-40B | reused as baseline; "Panchayat→Parliament" federal-tier corpus extends it |
| Byte-fallback / zero-UNK tokenizer guarantee | Bharat-40B | reused directly for Tier-3 languages |
| Atomic/protected digit tokenization for math | Blueprint, Bharat-40B | reused directly |

**Not in any of the four references, to our knowledge:** DPI-Gym, Sandhi-BPE, Rupantar Chains, Drishtikon-Bench, Samvidhan-CAI, Nirapeksh Loss, Bandit-DoReMi, Oral-Commons, Matryoshka vocab nesting, CodeMix-Bench, transliteration-canonicalized dedup, the Uday Schedule, canary reserved tokens, Mool Data, Nyaya-CoT, Bahumat-DPO, Gopaniya-RL, Gram Adapter Network, and Nagrik Grievance Loop — plus the real-dollar AWS cost estimate, the ANI v. OpenAI / Indian-copyright-TDM-gap analysis, the Apache 2.0 licensing recommendation, and restoring the non-Brahmic script budget the tokenizer papers below dropped.

## Real facts cited throughout

Gemma 4 technical overviews and benchmark trackers (labellerr.com, gemma4-ai.com); AI4Bharat's Sangraha dataset card (Hugging Face); MILU (arXiv:2411.02538, NAACL 2025); Sarvam-1 (sarvam.ai); BrahmicTokenizer-131K (arXiv:2605.29379, May 2026); IndicSuperTokenizer (OpenReview, Oct 2025); ONDC's public protocol specs (GitHub, ONDC-Official); AWS P5/Capacity Block pricing pages (July 2026); and reporting on *ANI Media v. OpenAI* (Delhi High Court, 2024–2026).
