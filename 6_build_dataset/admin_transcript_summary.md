# Admin-Only Summary

Source transcript: [ERA V5 Session - 2026_08_01 06_42 IST - Transcript.txt](ERA%20V5%20Session%20-%202026_08_01%2006_42%20IST%20-%20Transcript.txt)

## 1. Why this session matters
- This session is one of the most important because it connects model architecture with the real driver of quality: data.
- The Admin says that even if the transformer or attention block changes, the final outcome still depends on the data recipe and curriculum.

## 2. Data recipe and curriculum
- The model should not be trained on random data.
- The Admin stresses a staged curriculum:
  - first basic language grounding
  - then harder domains such as code, math, science, and agentic tasks
  - language preservation for Indic languages and other important domains
- This is similar to a student learning step by step instead of jumping into advanced material too early.

## 3. Capability buckets and proxy selection
- The Admin explains that the training data must be selected to improve specific capabilities.
- A "golden proxy" or reference set is used to find which samples actually help the model improve.
- The goal is not to train on everything, but to train on the examples that move the model in the right direction.

## 4. Training loop signals
- The model learns from next-token prediction.
- Important signals:
  - loss
  - perplexity
  - attention behavior
  - position information
  - mixture tags

## 5. Loss and perplexity
- Loss measures how wrong the model is.
- Perplexity measures how surprised the model is.
- A model may sound fluent but still be wrong.
- So the real test is not just good wording, but accurate next-token prediction.

## 6. Sequence length and sample structure
- A sequence is the number of tokens in one sample.
- Sequence length matters because:
  - short context may miss dependencies
  - long context increases memory cost
- Training examples must be structured properly so the model sees useful context.

## 7. Batch, microbatch, and global batch
- Microbatch: a small batch on one GPU
- Global batch: the total batch across many GPUs
- The Admin explains that large training runs need parallelism across GPUs.

## 8. Gradient accumulation
- Due to memory limits, the model cannot always do a full update on every microbatch.
- So gradients may be accumulated over several steps and then applied together.
- This helps reach a useful effective batch size.

## 9. Checkpointing and replayability
- Training is expensive and fragile.
- The Admin stresses the need for:
  - model weights
  - optimizer state
  - scheduler state
  - RNG state
  - data loader state
  - ledger offset

These are needed so the training run can recover safely after a crash or restart.

## 10. EOS, BOS, and loss mask
- EOS = end of sequence / sentence
- BOS = beginning of sequence / sentence
- These markers help the model understand where one context ends and another begins.

Loss mask:
- In supervised fine-tuning, the loss is usually applied only to the answer portion, not the question.
- This prevents the model from learning the wrong target.

## 11. From document to training sequence
A raw document becomes a training sequence through:
1. provenance tracking
2. quality metadata
3. tokenization
4. span creation
5. packing into fixed-length windows
6. grouping into microbatches and global batches
7. optimizer update

## 12. Padding and packing policies
The Admin spends a lot of time on fixed-shape training.

Main choices:
- pad tokens at the end
- concatenate and cut
- greedy packing
- best-fit packing
- structure-preserving packing for special data types

Important point:
- Padding wastes compute.
- A good packing strategy reduces wasted GPU work and preserves meaningful boundaries.

## 13. Why EOS is critical
- EOS tells the model that context has switched.
- Without that signal, the model may incorrectly carry over reasoning or language from the previous document.
- That boundary is essential for stable training, especially in mixed or multi-domain data.

## 14. Final concept
The Admin's core teaching is:

- architecture alone is not enough
- the data pipeline is the real training engine
- every part of training must be controlled, measured, and replayable

## One-line takeaway
The Admin is explaining that training a strong LLM is not only about models and math, but about designing a disciplined, curriculum-driven, checkpointable, and context-aware data pipeline.
