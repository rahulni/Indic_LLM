# ERA V5 — Session 6: Building the Training Dataset & the Data Ledger

**Source:** [ERA V5 Session - 2026_08_01 06_42 IST - Transcript.txt](ERA%20V5%20Session%20-%202026_08_01%2006_42%20IST%20-%20Transcript.txt) · Duration 02:40:02 · Speaker focus: **The Admin (Rohan)**

---

## How to read this document

- Sections follow the **exact order the Admin taught them**, with transcript timestamps.
- Everything in normal text is **what the Admin actually said**, cleaned up from speech-to-text.
- Blocks marked **`▸ Added context`** are outside knowledge added so you never have to look elsewhere. They are clearly separated from the lecture.
- Blocks marked **`⚠ Transcript note`** flag places where the auto-transcript garbled a number, and give the arithmetic that must have been meant.
- Student questions are folded into the section they belong to, plus a full Q&A index at the end.

---

## 0. The one-paragraph version

Session 6 converts the *human* dataset plan from Session 5 (capability buckets, curriculum stages, annealing, Opus selection, benchmark targets) into an **executable system**. That system is not a "data loader." It is a **data ledger** — a double-entry accounting book that records both *what was sent into the model* and *what came back out of the model* (per-token loss, perplexity, gradient norms, Opus decisions). The purpose is to make a 45–60 day, multi-million-rupee training run **controlled, inspectable, and replayable**. The assignment is to build that ledger end-to-end.

---

## 1. Why this session is "the diamond on top of the gold" (00:00 – 00:05)

The Admin opens by saying this is his best contribution to the course so far, and that it is personal — it comes from what he *wished* he had done during the previous ERA V4 training run and didn't.

His core argument:

> From Session 10 onwards we talk about transformer architecture. But those things are **deterministic**. If you swap one attention block for another algorithm, you already know what you'll get — a hit on memory, a change in perplexity, a change in loss. Everything still ties back down to **the data**.

Three things follow from this:

1. **Architecture choices are predictable; data choices are not.** Nobody publishes their data recipe, so this is where the real, unshared knowledge lives. "People talk about it in hush-hush kind of things."
2. **Naive framing is wrong.** Training is not "collect all data → put in folder → train." A model is fed *by an algorithm* that selects a slice of the dataset (the curriculum), hands it to the model, the model learns, then the **next shard** comes in.
3. **Scale changes everything.** "If you're going to train a model for an hour, whatever we're discussing today is of no use." At one hour you don't need any of this machinery. At 45–60 days you absolutely do.

**The motivating pain:** imagine you are on day 50 of a 60-day run and you need to know *what you trained on during day 40*. You open the folder and find 30 GB of files. You have no idea. That is the failure this whole session exists to prevent.

> **▸ Added context — why "deterministic" is the right word for architecture.**
> Architecture research at this point is largely a set of known trade-off curves: attention variant → memory/throughput/quality; width vs. depth → known scaling behaviour. Data composition has no equivalent published curve, because the labs that know it treat it as their primary competitive moat. This is why the Admin frames data as the frontier.

---

## 2. What Sessions 1–5 handed forward (00:02 – 00:07)

Session 5 ended with a **training data recipe**. Session 6 must make it executable. The carried-forward pieces:

| From Session 5 | What it means | Session 6 must turn it into |
|---|---|---|
| **Capability buckets** | The model must be *capable* in named areas: science, LaTeX, Python (specifically in code), agentic traces | Lane weights in a machine-readable mixture schedule |
| **Protected flows** | Some data types are always present and never dropped — above all **Indic** | A `protected` flag that survives Opus rejection |
| **Curriculum stages** | Teach in order. Don't teach LaTeX before English. Don't send agentic traces while the model is still learning functions | An ordered stage registry with token budgets |
| **Annealing reserves** | Never cut hard between difficulty buckets — overlap them | Warm-up bands + reserved annealing data per stage |
| **Opus selection** | A model-in-the-loop filter that decides which samples are worth training on | An accept / reject / defer decision log |
| **Benchmark ↔ mixture targets** | Each capability has a benchmark; each benchmark needs specific datasets | Capability lane tags on every shard |

### The Indic / NRI analogy (protected flows)

> It's very similar to how NRIs outside India behave at home — at least they talk in the native language, so the kid learns, and they need to do it **regularly**. It's not that you train for the first three years, then ten years with no native language, and suddenly in the 12th year you start again — it will feel absurd. The reason you don't speak your third or fourth language is that you don't practice it regularly.

The point is **not** that the model *can't* learn Indic if you feed it late. The point is that learning it "in a very rudimentary way" is unacceptable. So a portion of Indic must be present **throughout** training, not batched into one phase.

### Annealing (the gradual handover)

> Let's say normal English is Stage 1 and PhD-level English is Stage 2. We'll not say "today your English ends and PhD-level English starts." A bit of PhD-level English should start **very early on**, and a bit of normal English should **continue** as PhD-level English starts.

So stage transitions are ramps, not switches. This is why every stage in the schedule needs a **warm-up band**.

---

## 3. Opus selection, explained end-to-end (00:04 – 00:07)

This is the most novel idea carried in from Session 5, and it gets a full re-explanation. The procedure:

**Step 1 — Build the golden proxy.**
Take a small, very high-quality dataset representing exactly how you want the model to behave. Run it through the current model. **Do not train on it.** Observe which weights are behaving badly — i.e. which weights have not yet learned the feature that data exercises. Store this as a **map** of "these weights need updating."

**Step 2 — Probe the candidate pool.**
Take ~1,000 candidate training samples. Each sample is a full training sequence (4K / 8K / 16K / 32K tokens). But **only send the first ~512 tokens** of each. That is enough for this test and keeps the probe cheap.

**Step 3 — Compare.**
For each candidate, look at which weights *it* would move. Compare against the Opus map.

- Candidate moves the weights the map says need updating → **keep it**.
- Candidate doesn't → **throw it away**.

**Step 4 — Train.**
Say 200 of the 1,000 survive. Train on those 200 at the **full** sequence length (e.g. 32,000 tokens), not 512.

**Why the ratio matters:** the probe costs money too. You run the golden proxy once, then probe 1,000 samples at 512 tokens each, then do the real training. The probe cost must stay small relative to training cost, or the speed-up disappears. "That balancing has to be done."

### The Admin's framing of what this really is

> It's a sort of biasing towards exams — like how Indian education is. Your parents bias you towards JEE or NEET. We are introducing our Indian parenting culture into training the model. Does it work? I don't know. But it definitely works for NRIs — when people go out of India. We're hoping our model also goes out of India. It'll get US citizenship and life is sorted.

### The rejections are as valuable as the selections

> Which samples did we throw away? What was in those? **Should we have sent those samples for training earlier on**, because now Opus is saying "I don't need it"? It's like — right now we started learning A for apple, B for ball. At *this* stage, maybe that discussion should have happened earlier.

This is the seed of the whole ledger idea: a rejection is a *timestamped statement about the model's current state*, and it is only observable during training.

---

## 4. Signals the training loop must expose (00:07 – 00:15)

For any of this to be recordable, the training run has to hand back specific things. The Admin lists them:

1. **Token window** — how big each sample is, and how many are sent together (this is where GPU parallelism enters).
2. **Loss on this batch.**
3. **Attention mask / attention map** — what the model looked at while predicting each token.
4. **Position information** — sometimes position matters, sometimes it doesn't.
5. **Mixture tags** — this batch had *this* mixture.
6. **Packed microbatch structure** — global batch vs. per-GPU batch; correctness must hold across many workers (GPUs). "At the same time we don't work on one GPU — 8, 16, 24, 32, depending on our training strategy."
7. **Training restarts** — because you will stop, and you will crash.

### The attention-map analogy

> The first time I introduced "Opus" to you, the word *Opus* was getting trained in your head. Every time we said Opus, you said "okay, I need to link this word to all of the concepts we're discussing." That's the attention map. We need access to that also.

### The position analogy

> "The umbrella" — *the* is always followed by some noun or name or thing. So we need to know the position where exactly that word came in.

### Why we might *stop* a training run

The Admin lists the real reasons:
- Your GPU rental ends / someone shuts things down at 5 pm.
- Your model crashes.
- Your loss goes to a value you're not happy about.
- **Your loss drops far too much, far too fast.**

That last one gets a whole story:

> If you start training a model and within the first 100–200 samples you see the model beating OpenAI — **we should restart**. That is really, really bad. It means something has gone into the training set and it's cheating. The NEET paper is leaked and it has all the answers.

**The school-hack story (benchmark contamination, illustrated).** At a friend's school, all the maths marks for a third-class section changed on the website. They traced the IP to a house. The father had three sons — a 21-year-old computer science student ("no idea"), a 15-year-old cricketer ("no idea"), and the third-class student ("I don't know hacking"). They opened the third one's computer: ChatGPT and Claude installed, and he'd asked them to change the scores. He changed marks for the *whole class*, and gave himself 95–96, not 100.

> Unfortunately, models do not have that kind of consciousness — "let me not give myself 100%, let me keep it at 95." If it did that, we'd be tricked. As of now we can handle it.

> **▸ Added context — this is "benchmark contamination" / "eval leakage."**
> The technical name for what the story illustrates: if evaluation data leaks into pretraining data, benchmark scores become meaningless because the model has memorised the answers rather than learned the capability. It is a documented, widespread problem across public LLM benchmarks. This is exactly why the session later mandates an **evaluation firewall** and a per-shard `eval_overlap` status field — a suspiciously fast loss drop is the *symptom*; the manifest fields are the *prevention*.

---

## 5. The checkpoint disaster stories (00:12 – 00:14)

Two failure modes the Admin has personally hit:

**Failure 1 — you edited the checkpoint code.** You decided you also want the loss written into the checkpoint filename, and you made a mistake in that one Python line. Your model now trains for 1,000 steps, reaches step 1,000, tries to save, and crashes.

**Failure 2 — out of disk.** "And this happened, by the way." Everything is working, the model is training, and then checkpointing fails because storage isn't there. Your 200 GB model wants to write itself to disk and there's no space, because you forgot to delete the old checkpoints. Model crashes.

The saving grace in both cases: **because a prior checkpoint was saved, you can go back to a prior state.** Which is also why checkpoints go to **cloud storage**, not just local disk — "we don't want to save on local, machines will keep on changing."

---

## 6. Requirements pulled forward from the transformer foundations (00:13 – 00:20)

The Admin walks through what the architecture side demands from the data side.

### 6.1 EOS and BOS

- **EOS** = end of statement / end of sequence.
- **BOS** = beginning of statement / sequence (sometimes called SOS).

> It's like *lights, camera, action* — that is a token.

Why they matter: you may be training at 32,000 or 128,000 token context length, and **no single book or paragraph runs that long**. So you have to mix and merge multiple documents into one training sequence. Without a boundary marker:

> Let's say we're talking Mahendra Singh Dhoni and then suddenly we're talking about nuclear weapons and suddenly microbiology. The model will get confused. So we send the Dhoni discussion, then add the EOS token — okay, the context is changing.

The human analogy: one phone call is customer support, the next is a bank loan, the next is your kid's principal calling about an exam. **You do the same thing — but you know the context is switching.** EOS is how the model knows.

### 6.2 Position policy

> When we send tokens to the model for training, we need to tell it: this is token number one, token number two, token number three. Right now when I'm talking to you I don't have to say "one, two" — because you and I live in time. Everything comes to you in time order.

So position information is required *initially*. He then notes a live research finding they used in V4:

> A few people came in — we implemented it last V4 training — that we can actually **get rid of it**. Position embeddings can be removed after a while in the model.

He also flags this later: "you're going to remove the RoPE, your position embedding" is one of the structural changes you may make mid-run, which is another reason you must be able to stop and resume exactly.

> **▸ Added context — what "removing position embeddings" refers to.**
> RoPE (Rotary Position Embedding) is the standard positional scheme in modern decoder-only LLMs — it rotates query/key vectors by an angle proportional to position, so attention scores depend on *relative* distance. There is a line of research ("NoPE" / no positional encoding) showing that causal decoder-only transformers can encode position implicitly through the causal mask alone, so explicit positional encoding can sometimes be reduced or removed. The Admin is referring to this class of result. Treat it as an experiment they ran, not settled doctrine.

### 6.3 Next-token loss

> The capital of India is ___ — and here we're expecting Delhi. If instead it says something else, the loss is going to be big.

Mechanically: the model predicts a token, the training loop compares the predicted token against the actual next token, and **the bigger the gap, the bigger the loss.** That's it. Full mathematical treatment is in §16.

### 6.4 Perplexity, and the "talks for 15 minutes and says nothing" test

> Loss tells us how bad it is. The **exponential of loss** tells us how *surprised* the model is. That term is called perplexity.

Why a high-perplexity word is gold:

> If you get a much higher perplexity for a particular word, we know the model actually does **not** know about this. And that tells us whether the model is **faking it** or not.

> Have you met those people who can speak for an hour without saying literally anything? They keep talking, and you ask yourself — the person has been speaking for 15 minutes, what has he said? And you realise: nothing.

The concrete version:

> You ask "what is the speed of light?" and it says *"light is an electromagnetic wave that travels through space, it is very fast, blah blah"* for 15 minutes. It has not said 2.9 × 10⁸.

The killer property — **this can only be measured *during* training:**

> After training, of course we can get perplexity, because we can run it on data where we know the ground truth. But **while training**, if we record it, then: at the initial stage the model had very high perplexity on these words; as we train, the same words have lower perplexity; as we train further, lower and lower. That is gold — because **nobody in the world releases perplexity data**, and it is the actual measure of how intelligent a model is.

> If someone is so fluent in quantum mechanics that he's using all the right terms, he must have got some knowledge to be able to predict the right tokens.

### 6.5 The tokenizer/dataset chicken-and-egg

> The tokenizer has to be frozen — but it is going to be **built from the dataset**. And when we're building the dataset we need to make sure some of the tokenizer tags are in the dataset also.

The resolution he gives is an explicit ordering:

1. Build a *bit* of tokenizer (enough for the special tags).
2. Build the dataset.
3. **Then finish and freeze the tokenizer.**

### 6.6 Loss masking (introduced)

> For SFT — supervised fine-tuning — we ask the model to write a poem on George Bush. We're **not** going to penalise the model on the question. The loss is only calculated on the **answer**.

> Imagine a NEET or JEE paper where the first line says "write your own question and write your own answer." You can't punish the student for that. Actually — that's a good way of figuring out who got the leaked paper.

### 6.7 Cleaning and deduplication

> What is the point of going through the same example your teacher taught, then you saw it with your friend, then it was also taught in tuition, then again during revision?

Deduplication is what makes the dataset "really tight" — every step the model should be learning something new.

---

## 7. The full list of what the loader must record (00:20 – 00:23)

The Admin renames the component here:

> Right now our data loader — I'm going to change the word from *loader* to something else.

That something else is the **ledger**. What it must collect:

- Next-token loss map
- **Tokenizer hashes** — so I can compare what I just trained on with a shard I'll train on in future
- **Source provenance** — I know where this dataset came from
- **Cleaning manifest** — yes, this shard was actually cleaned
- **Evaluation firewall** (see below)
- **Mixture proof / mixture flows** — how the mixture happened, and it changes over time
- **Opus logs** — what was rejected, what was selected, and whether the selection made sense
- **Annealing reserves** — is this part being annealed, are we in the annealing stage

### The evaluation firewall — defence on both sides

> We're going to write the training code in such a way that it will **ask** whether this data is for training or evaluation. We will make sure we never send the evaluation or test dataset to train on. But **we will also** make sure that while training, the code asks whether this is training or evaluation data — and **rejects** test and eval.

> So we're going to do it from both sides — because who knows, maybe a mistake in copying may still happen.

### Why the mixture must be recorded continuously

> There are two buckets and we change them as we train. It's not that we're 30 / 100 and that's it. The 40 is going to slowly become 20. A new topic comes in. The 30 slowly increases — maybe it was coding — and the other 30 might slowly reduce. **At every step you have a different ratio.**

> So we need to know what the ratio was when I was training at a particular point. It's like GDP — you don't store "1950 GDP, 2026 GDP." **You store it every single quarter.**

### An Opus log that reveals a broken proxy

> Okay, Opus did select this — but my benchmark still says I'm bad at coding. So there's a mismatch. Maybe Opus is selecting coding, but that coding happens to be **CUDA / compiler code**, while the code in my dataset is more **Python**. So we have selected the wrong proxy in Opus.

That diagnosis is impossible without the log.

---

## 8. Why `torch.utils.data.DataLoader` is not enough (00:23 – 00:25)

> The data loader is a program freely provided by Python/PyTorch. It's very simple. The simplest data loader does this: I'm going to give you a batch, I'm going to give you the next batch. And there's a way of making sure it gives the same batch using a Python **seed**.

> But what people don't know is: when you use a Python seed — supposed to give you a fixed sequence of random numbers — that is true only for **Python working on that machine on that day**, as long as the kernel was alive. Turn the kernel off, turn the machine off and on again, you'll get a different value. There's no way you're asking 10 random functions to give you the same value. Think about how many ways it can break.

> **⚠ Transcript note / ▸ Added context — what is precisely true here.**
> `random.seed(1667)` in pure Python *is* reproducible across restarts and machines; that specific claim is too strong. But the Admin's **conclusion is correct**, for stronger reasons. End-to-end reproducibility of a distributed training run breaks because of:
> - **Multi-process dataloader workers** — with N worker processes, the *interleaving* of which worker returns which batch first is not guaranteed, so ordering can differ even with a fixed seed.
> - **GPU non-determinism** — floating-point atomics and non-deterministic reduction orders in CUDA kernels mean the same inputs can give bitwise-different outputs; cuDNN may also autotune to a different algorithm on a different run.
> - **Different world size** — resuming on 16 GPUs instead of 8 changes the sharding and therefore the sample order.
> - **Library / driver version drift** — a different CUDA or PyTorch build changes kernel selection.
>
> This is the technically sound justification for the ledger: **don't re-derive the order, record it and replay it.** That is exactly the answer the Admin gives Vardhan at 02:25 (see §19).

---

## 9. Core vocabulary (00:25 – 00:27)

The Admin stops to define terms: *"otherwise we're talking Farsi and you're going to say 'okay it was a good session, I have not understood anything.'"*

| Term | Definition as given |
|---|---|
| **Token** | Can be a full word, but usually a sub-word. Whether a word survives as one token depends on **how many times it was repeated in the dataset**. Frequent words become single tokens. |
| **Sequence (sequence length)** | How long one training sample is, in tokens. |
| **Sample** | One training example handed to the model. In pretraining, a sample is usually a **fixed-length** token sequence. |
| **Batch size** | How many samples are sent at once. |
| **Microbatch** | The small batch processed on **one** GPU before gradients are accumulated. |
| **Global batch** | `num_GPUs × microbatch × grad_accum_steps` — the samples covered by one optimizer update. |
| **Training step** | **One optimizer update.** Not one forward pass. |
| **Shard** | An immutable, pre-tokenized chunk of the dataset — enough for ~1,000 training steps. |
| **Checkpoint** | Many training steps' worth of state, saved to disk/cloud. |

### Why long-sequence training needs long-sequence data

> If we want a model that is really good at long sequences, we have to train the model **on** long sequences. The reason is we need to see that **attention survives** — we need to make sure perplexity doesn't start increasing.

> It's like you're very sleepy and someone is telling you something — that will be magical if you catch it. Or you're so tired, you think it's the weekend, you're about to sleep, and someone wakes you: "tomorrow you have to wake up at 5, it's a flight." We want our model to be **super tight always**.

### Batch size = GPU RAM

> If I'm training on one sample, I need a copy of the model once. Two samples — two copies. 100 samples — 100 copies. What do I mean by a copy? **That is the amount of RAM I need to process it.** One model cannot look at two samples simultaneously. Sequentially, of course. But simultaneously, for 100 samples, I need that much GPU RAM.

### The Nvidia aside

> That is why Nvidia is killing it. Nvidia makes GPUs — everybody does. The GPU innovation is sort of okay, it's not magically changing anything. **Nvidia's wealth is built on memory** — the RAM that comes from Hynix. Nvidia is a RAM company, basically. The valuation really depends on how much RAM it can sell on top of the GPU.

> **▸ Added context.** The technical substance behind this: modern LLM training is overwhelmingly **memory-bandwidth-bound**, not FLOP-bound. What differentiates a datacentre GPU is its High Bandwidth Memory (HBM) capacity and bandwidth. HBM is manufactured by SK Hynix, Samsung and Micron — not by Nvidia — and HBM supply has been the binding constraint on datacentre GPU output. So "how much RAM can it sell on top of the GPU" is a fair description of the economics.

### SFT samples are structured differently

A pretraining sample is just tokens. An SFT / agentic sample contains:

- **Prompt tokens** — sent to the model, **not trained on**
- **Response tokens** — trained on, this is where loss comes from
- **Tool observations** — sent to the model, **not trained on**
- **Masked segments** — never sent
- **Labels** — may not be sent

---

## 10. The batch-size arithmetic, worked (00:27 – 00:35)

### The target

> In general, people want to train on **1 million tokens** before one back-prop. The general advice from big labs is **1 to 4 million** — they can go up to 4. **1 million is the sweet spot.** If you hit 0.5 million tokens per back-propagation, that's a good place to be, but 1 million is what they're expecting.

### Configuration A — derive the batch from the target

```
target tokens/step = 1,000,000
sequence length    = 4,096

samples needed  = 1,000,000 / 4,096 = 244
GPUs per cluster = 8  (one standard cluster)
per-GPU samples  = 244 / 8 = 30.5  →  must be a whole number  →  30
```

> We can't do 30.5. What is 30.5? You're sending something in some short sequence — that can't happen.

```
actual tokens/step = 8 × 30 × 4,096 = 983,040  ≈ 0.98 M   ✔ close to 1 M
```

### Configuration B — push it to a power of two

```
8 GPUs × 32 samples × 4,096 = 1,048,576 ≈ 1.05 M   ✔
```
Here **32 is the microbatch** — how many samples each GPU receives.

> Can we pack in more tokens on our GPU? That's where some of the algorithms we'll discuss in future come in. One way of pushing more together is just to increase this — and now you have a batch of a billion tokens. **Bad idea. Really bad idea.** It's like explaining the whole of quantum physics in one session and then asking "explain what it is."

### Configuration C — when the GPU can't hold it (the gradient accumulation case)

Real situation with a big model:

```
8 GPUs × 8 microbatch × 4,096 seq × 1 grad_accum
  = global batch of 64 samples
  = 64 × 4,096 = 262,144 tokens/step  ≈ 0.26 M   ✘ far short of 1 M
```

Fix — **gradient accumulation of 4**:

```
262,144 × 4 = 1,048,576 ≈ 1.05 M   ✔
```

> We're going to train for step one, two, three, four. We accumulate all the loss, accumulate all the gradients, and then **back-propagate once**. So we fake it.

> It's like: are you going to give a test after every single page of the book, or are you going to let at least the chapter end?

> **⚠ Transcript note.** At 00:35 the transcript records "that means right now I'm sending 32,000 token" for the 8×8×4096 case. The arithmetic is 64 × 4,096 = **262,144**. The speech-to-text garbled the figure; the ×4 accumulation to reach ~1.05 M is consistent and correct.

### The gradient-accumulation ego confession

> I **hate** gradient accumulation because it feels like I'm poor. It's an ego issue for me. If you ask me to use gradient accumulation I'm going to be like… it's like going to a small village where my mother is telling her mother "beta IIT clear" and she's saying it — that's how gradient accumulation feels.

But he immediately concedes it's a good strategy, and notes the alternative:

> The other way of hitting zero gradient accumulation is by **increasing the sequence length** — which means we need some optimization where somehow we can send more tokens to the same model. That is difficult. So we need to figure out a way. This microbatch is going to be the main thing, and that is where we're going to talk about something called **reversibility**.

> **▸ Added context — the memory levers behind "reversibility."** The general family of techniques for fitting more tokens per GPU: **activation checkpointing** (discard intermediate activations in the forward pass and recompute them during the backward pass — trades compute for memory), **reversible layers** (architect the block so activations can be *reconstructed* exactly from the outputs rather than stored at all), **mixed precision** (bf16/fp8 activations), and **memory-efficient attention** (FlashAttention-style, which avoids materialising the full N×N attention matrix). The Admin defers the detail to a later session.

### The exponential memory wall

> The amount of RAM you need for a 4K token sequence versus a 16K token sequence — there's an **exponential change** in the RAM required, because of how attention behaves.

He demonstrates with a 120-billion-parameter model where only a single 32,000-token sequence fits.

> **▸ Added context — precisely why.** Standard self-attention computes an N×N score matrix for sequence length N, so its activation memory grows **quadratically** in sequence length, while the parameter memory stays fixed. Going 4K → 16K is 4× the length but ~16× the attention activation footprint. (Memory-efficient attention implementations reduce this to linear in N by tiling and never materialising the full matrix — but the quadratic *compute* remains.) The Admin's "exponential" is loose speech for this super-linear blow-up.

---

## 11. The actual machine: AWS p4de.24xlarge (00:29 – 00:31)

The Admin pulls up the instance spec live.

| Spec | Value | What it's for |
|---|---|---|
| **vCPUs** | 96 | Reading/decoding data. "CPUs do sequential; GPUs do parallel really well. Those 96 are going to be used by us to **read the data** — at least 90 threads loading data at the same time, so my reading speed increases and I can feed the GPU faster." |
| **System RAM** | ~1 TB | "Looks a lot, but you're going to see that for some things this is not going to be enough. We may need a bigger machine." |
| **GPUs** | 8 × A100 **80 GB** | |
| **Total GPU RAM** | **640 GB** | |
| **Cost** | ~**₹2,700 / hour** (Admin's figure) | "This is the bare minimum we need. We can't even train on lower than this." |

### Why 1 TB of system RAM isn't obviously enough

> Our model is also going to be divided into eight parts and living on eight GPUs separately. So when you do a checkpoint, you have to **load parts of the model from different GPUs, pack it, and then send it**. All of that is sequential. So your CPU is required at times — and at that time **training has to stop**.

He confirms this later with real numbers (§20): reassembling the V4 checkpoint needed around a terabyte of CPU RAM.

---

## 12. Shards and how big they are (00:31 – 00:32, revisited 01:25)

A shard is the unit you download and hold locally so training doesn't stall waiting on the network.

```
global batch     = 8 GPUs × 32 microbatch = 256 samples
tokens per step  = 256 × 4,096            = 1,048,576
steps per shard  = 1,000
tokens per shard = 1,000 × 1,048,576      ≈ 1.05 billion tokens
```

> A shard is going to contain at least 1,000 steps — or 400 steps — of GPU training. That means each shard is around **a billion tokens**. That's the kind of number we're going to come across.

Properties of a shard:

- **Immutable.** "A shard is an immutable training object." Once written, never modified. Same code on a different day must produce a byte-identical shard.
- **Pre-tokenized.** "Tokenization has happened before the training begins. Tokenization is expensive and it must be frozen."
- **Stored as a binary token array**, compressed. "If you send the raw tokens from AWS to GPU, all the money is going to be spent on data transfer. We don't want that."
- **Overlappable with download.** "The model can learn for 1,000 steps before the local data is exhausted, and **in parallel we download the next one**."
- **May be a tar** if individual samples are large — download the tar, decompress locally.
- **Carries structure** for SFT and agentic training, not just flat tokens.

> **⚠ Transcript note.** At 01:25 the Admin counts digits aloud and says "around 10 billion token shard." The arithmetic he states (256 × 4,096 × 1,000) gives ≈ **1.05 billion**, matching what he said earlier at 00:32. Take ~1 B as the working figure.

---

## 13. When do you checkpoint? (00:36 – 00:40)

The Admin turns this into a classroom exercise. Wrong answers first:

| Student answer | Admin's response |
|---|---|
| "Whenever the loss is reduced" (Chandrahaas) | We may have 100 million back-propagation steps. |
| "After one back propagation" (Dattatreya) | Same — far too many. |
| "After one epoch" (Rahul) | **"There is no epoch concept in LLMs."** |
| "At every 25% loss" (Satyanarayana) | "We don't know what the final loss is going to be." |

### "There is no epoch"

> My one epoch is 10 trillion tokens. Two epochs is me training again on this 10 trillion. **Nobody does that if they have data.**

> **▸ Added context.** In classical supervised learning you loop over the dataset many times (epochs). Frontier LLM pretraining is effectively **single-epoch**: the corpus is so large relative to compute that you see most data roughly once. Repeating data is done sparingly and deliberately (up-sampling high-quality subsets), and heavy repetition is known to degrade returns. This is why "which epoch am I in" is a meaningless recovery coordinate, and why the Admin needs a **ledger offset** instead.

### The right answer: money

> Guess what is the one thing I keep answering when you guys ask me — **money**. We're going to see that we have spent $100, let me save it. We've spent $100, let me save it.

> Ideally it's going to be an hour — because we'll not add the value of the GPU consumed otherwise. Every hour you save it, or every 100 steps — but **it will be linked to the amount of money you have spent**, and you need to be conscious about that.

> Do I want to lose the last $50, or do I want to lose the last $1,000? Everything else you calculate is going to be backed by this number.

**The rule:** checkpoint interval is a **risk-in-rupees** decision. Pick the amount of GPU spend you are willing to throw away on a crash, and checkpoint at that interval.

### What is actually inside a checkpoint

Not just weights. All six of these:

| Component | Why it's needed |
|---|---|
| **Model weights** | Obvious. |
| **Optimizer state** | "Each weight's *change behaviour* — because when we restart we need the **momentum** also. That's what Adam, AdaDelta, SGD are." |
| **Scheduler state** | "How the learning rate was changing. If we just stop, we'd have to recalculate, and we might be wrong." |
| **RNG state** | "Where available. I need to explain what RNG is to you." (deferred to a later session) |
| **Data loader state** | "When I restart I do not want to train on data I've already trained on. So I need to ask the data loader — give me the data from step 1,000 onwards." |
| **Ledger offset** | "Is there any delta compared to where I'm training? If I'm going back, I need to go back in my ledger and start there." |

> **▸ Added context — why optimizer state dominates checkpoint size.** With Adam/AdamW you store, per parameter: the parameter itself, the first-moment estimate (m), and the second-moment estimate (v) — plus often an fp32 master copy of the weights when training in mixed precision. That's why the Admin's V4 checkpoint (§20) was ~256–298 GB for a model whose raw bf16 weights would be far smaller. **You cannot skip saving optimizer state** — resuming without momentum causes a visible loss spike, because Adam's adaptive step sizes have to be re-estimated from scratch.

### Ledger, defined (answering the student "M")

> Data loader loads the data, gives it to the GPU. **We are designing a data ledger.** It's a very similar concept, but the ledger does not only store the incoming detail — **it is also going to store the outgoing detail.** It will do everything the data loader is supposed to do, but we're also going to take data **back from the model** and save it there.

> Let's say we're saving the whole entry and we are at step 2,566, and then you ask for a checkpoint at 2,000 — in the ledger we can peacefully go back and continue from there.

**This bidirectionality is the single defining idea of the session.** A data loader is one-way. A ledger is double-entry.

---

## 14. Document → training sequence (00:44 – 00:50)

The transformation pipeline, in order:

```
Document
  → tokenize into token IDs
  → split into spans        (if longer than sequence length)
  → pack spans into fixed-length sequences
  → group sequences into microbatches   (one microbatch → one GPU)
  → aggregate microbatches into a global batch  (one saved unit on disk)
  → optimizer step updates the model
```

> Why spans? Because let's say a 4K sequence — I'm going to create my examples of 4,000 tokens. If the document was longer, we divide it into spans. If it's shorter, we're going to do something [packing/padding].

> As I shared last time, we're going to save 100 or 1,000 samples **of the same length** together, and then ship it — because that's easier for a model to go through.

### What a "clean document" must carry before it can become a training sequence

- **Provenance** — where exactly did it come from, what is the source, who downloaded it, **which group among you produced that shard**
- **Quality metadata** — do we have an idea how good this data is
- Then, added through the pipeline: token IDs → token spans → packed sequence → microbatch → global batch → optimizer step

### Meaningful boundaries that must be preserved

| Field | Purpose |
|---|---|
| **EOS token** | Marks context switch (see §15) |
| **Document ID** | Identifies the source. All of it is stored → and *"if a lot of data is stored and is reproducible and we can read it, it becomes a ledger."* |
| **Loss mask** | Determines which tokens contribute loss |
| **Attention mask** | "Where exactly I looked at while predicting that particular token" |
| **Position IDs** | "1, 2, 3, 4 — where exactly the training path was" |

### The dream dataset (why attention + gradient records matter)

> Imagine someone could give this dataset. I can't tell you how amazing it would be — that if you train a model at a mid-stage and you train on *this* sentence, the attention is going to be very high on *this* word, and this is the amount of learning you'll have. But for another example the **gradients jump even higher** — so that's a much better example. On another example the gradients are so high that **the model's behaviour will suddenly change.**

> So we can be very, very cautious about what to train on.

### Full loss recording

> The loss mask determines the amount of loss we had for that sample. We have the **full sample loss**. We also have **each token's loss**. Both are going to be saved.

---

## 15. EOS in depth (00:47 – 00:50, 01:00 – 01:10)

### Where EOS goes

> **End of document, not end of sentence.** — (answering Abhishek)
> **It's end of the *context*, to be very clear.**

> After a paragraph you already have a newline character — that thing you can't see, but if you enable it in some text editors you see those arrows. They're separate. **We add EOS after a document ends.** Be very clear on this.

> **We are not lucky enough that we'll always find exactly 4,096 tokens per document. Not possible.**

### The chatbot argument for EOS

> We have to tell the model that the turn has ended. Say we're talking to a chatbot. "Hi." The model responds "hello" and **stops**. We're expecting to talk. So it needs to stop.

> You say "hi" to a random stranger — he's going to look at you and this is going to continue. We don't want that. We want the stranger to understand: I said hi, gave you the end-of-statement, now **you** say something.

### The contamination argument for EOS

> Without an explicit boundary, the model will learn unrelated topics. Suddenly you're talking about physics and it learns it's okay to talk about maths — which is okay. But it will start talking about **religion** — that's not okay. It will start talking about, for example, **suicide** — we don't want that. Suddenly it starts talking in a different language: you're talking in English and it starts talking in Malayalam.

> **EOS immediately resets.** It tells the model that the behaviour has changed — the new text you're going to predict has a very different behaviour.

> It's like your school: 8:30 a.m. English class, and after that your Hindi class. **EOS is your small break.**

### Q — Mukund: "Won't the model learn boundaries anyway from loss?"

> **Mukund:** You mentioned EOS helps the model differentiate two documents. But won't that also be learned during training itself? If it starts using an unrelated document to predict another, the training loss will be higher, right?
>
> **Admin:** Exactly. Hence it will not do it. Back-propagation will immediately tell it "you're using EOS in the wrong way."
>
> **Mukund:** So does EOS matter then?
>
> **Admin:** EOS exists **to tell the model that these two are different contexts**. If the model starts using the older context for the new prediction, back-propagation will fix it. [i.e. EOS is the *signal*; back-prop is the *mechanism that teaches the model to obey the signal*. Without the signal there is nothing to key off.]

### Q — Swati: "How does the model *know* the context ended at EOS?"

> Imagine token 1, token 2, token 3, then EOS, then the context switch. The model saw this, and when predicting it used attention from *this* earlier token. **Back-propagation comes in and says: dude, wrong. This was not a token to be predicted. I need to unlearn this.** So the model will learn automatically that whenever EOS comes in, all the attention going back to prior tokens is not required and should not be looked at. Back-propagation will teach it.

### Q — Dattatreya: "Is it back-prop or is it how attention is implemented?"

> **Admin:** Back-propagation is required for attention to work. Everything that is learned in a model is learned through back-propagation.
>
> **Dattatreya:** But attention is just matrix multiplications. If you don't consider anything before EOS, that automatically means what you learn won't consider the previous document.
>
> **Admin:** **You are confusing the *memory* of the model with the *context* of the discussion.** Do you know how much money [X] made? That is not today's context — you did not learn it today. From prior memory the model will be able to predict. It is **told not to look at the context** of what has happened. That is where EOS comes in.

> **▸ Added context — reconciling both views.** Both mechanisms exist and are used in practice:
> - **Learned boundary (what the Admin describes):** EOS is an ordinary token; back-prop teaches the model that attending across it is unhelpful. Cross-document attention is *possible* but *discouraged by the loss*.
> - **Enforced boundary (what Dattatreya was reaching for):** you can additionally build a **block-diagonal / document-causal attention mask** so tokens physically cannot attend across a document boundary within a packed sequence. This is sometimes called intra-document masking or "document masking." It makes the boundary hard rather than learned.
> The Admin's design uses the learned-boundary approach and records the attention mask; the enforced version is a stricter option available when packing unrelated documents.

### Q — Gaurav: "Why both EOS and BOS? Can't we use one?"

> EOS and BOS are required because that's how people train on different strategies. There are companies who train on EOS **and** BOS; there are companies that use **only** EOS. It literally depends on the strategy — do you want to use EOS or BOS?
>
> **Gaurav:** But we'll not use both together, right?
>
> **Admin:** Both will not be used. **Earlier people were doing it, but now they know it's a waste of space. So only EOS is used.** That's why the whole document session talks about EOS only.

### Q — Sachin: "Samples come from multiple documents — how does the model know?"

> **Admin:** Why don't you give me a dataset where you can find 4,096 tokens continuously? I'll be super happy to receive it. **This whole page I'm presenting today will not have 4,000 usable tokens.** Point one. Point two: we have bigger sequences also. Point three: if a document is long we're going to cut it, and after token 4,095 we add EOS and it becomes part of something else.

### Q — Harini: "If a document is chopped across two sequences, do we still add EOS?"

> We have EOS and BOS, but **the documents are lost in the sea of tokens** — and as I said, concat-and-chop is [the crude option]. The best way for us is **best-fit packing**. We will try to figure out a mechanism where **I do not have to chop documents.**

### Q — Harini: "Can we compress the tokens?"

> If your tokenizer is not good and we have **13 tokens per word for Telugu**, then the document has to be chopped — a 1,000-word Telugu document becomes a 13,000-token document. **That is another reason why the tokenizer is important.** If my fertility was only 1.3, then I have 1,300 tokens — so I can actually add three more documents.

> **▸ Added context — "fertility."** Fertility = average number of tokens the tokenizer emits per word. Fertility 1.0 means one token per word (ideal); fertility 13 means the tokenizer is falling back to near-byte-level fragments because it never learned that language's subwords. High fertility is a **triple** penalty: (1) fewer real words fit in the context window, (2) you pay 13× the compute for the same semantic content, (3) inference costs 13× more per word for that language's users. This is the direct technical reason the Admin later states the Indic goal as **"fertility of 2 for all the languages"** (§22).

---

## 16. Loss masking, resolved (00:50 – 00:55)

**Definition:** the loss mask says, per token, *do we compute loss here — yes or no?*

> Write a letter to my manager saying I'm not coming to office tomorrow. Now there's a letter. **The question is not part of the loss.** When I'm asking the question, loss is not calculated on the question — it's only calculated on the answer. That is a loss mask.

Critically:

> It's **still part of the training sequence** — it just doesn't produce loss. The model still reads it. How do we train the model? Token by token. It has to predict [the answer]. You can't do that for a question, so we don't do it for the question.

### Q — Chandrahaas: "Do you mask the tokens with the highest next-token loss?"

No. The Admin's answer redirects entirely: masking is **structural** (prompt vs. answer), not **value-based**. You never mask tokens because their loss was high — high loss is the *signal you want to keep*, not something to suppress.

### Q — Suresh: "Why does only the agentic batch have a loss mask policy?"

> Where else should we add it? How do I add it to a general text example — what do I not train on? **Loss masks come in from the SFT stage. This is not there in pre-training.** We discussed this last session.

| Stage | Loss mask? | What's masked |
|---|---|---|
| **Pre-training** (general web, code, Indic) | No | Everything is trained on |
| **SFT** | Yes | Prompt/question masked out |
| **Agentic** | Yes | Prompt **and tool observations** masked out; only the model's own generated actions/responses carry loss |

> **▸ Added context — why tool observations must be masked in agentic data.** In an agentic trace the sequence alternates: model action → environment/tool output → model action → … The tool output is *not something the model produced*, and it is often non-deterministic (an API response, a file listing, a stack trace). Training the model to *predict* tool outputs teaches it to hallucinate environment responses instead of calling the tool. So tool observations are **context** (attended to) but **not targets** (no loss).

### Q — Mukund: "Is EOS part of the loss mask policy?"

The Admin's answer is fragmented in the transcript, but he says EOS **has to be at the end of the sentence** and that its treatment differs because "the number of times it appears is very different." Practically: EOS is a token the model **must learn to predict** — see §18, where he explicitly tracks **perplexity at EOS** as a health check ("if the EOS comes in and the model has to predict EOS, it's going to say 'I'm not done'"). So EOS is *not* masked out in pretraining; you want loss on it.

---

## 17. Padding — the three kinds and what each wastes (00:55 – 01:10)

### Why fixed shapes are mandatory

> Training systems prefer fixed shapes. Natural text does **not** arrive in fixed shape — it's 100 tokens, or 500, or 1,000, or 10,000.

> How many samples do we have in one global batch? **256.** And all 256 must be **the same length**. It's a requirement of how the GPU works. And even if it were **not** a GPU requirement — you can't send 1,000 tokens to one and 8,000 to another, because the one that receives 1,000 will spend its time **waiting** for the next 7,000.

> There was an ad from Visa: money can't buy love, but for everything else there's Mastercard. **The answer to every single question is money.**

The illustrated version with 4 GPUs:

> It is not possible for us to load a model into four GPUs and say one model runs for *these* number of tokens, another for *these*, another for *these*. Not possible. **GPU is a parallel machine. It needs to compute everything for the same number of iterations.** So unfortunately we have to fill with a **pad token** — things that have no meaning. That is a wastage of space.

### The three padding strategies

| Strategy | How it works | Cost |
|---|---|---|
| **Right-side padding** | Real tokens first, pads appended at the end. "Generally people add on the right — it's easier to do." | Model learns to predict pad after pad. Loss collapses to ~0 for those positions. |
| **Left-side padding** | Pads first, then real tokens. | **"Better than right-side, at least — because it has to learn something at the end.** You will see slightly higher loss, but for 4,000 tokens it's really happy." |
| **Batch-level (dynamic) padding** | Sort by length, then set *this batch's* sequence length to the **longest example in the batch** instead of the model's full context length. | Best of the three. "Instead of adding more padding here, my batch sequence length for this step is 3,096, and I only add padding to this one." |
| **Fixed context padding** | Pad every sample to the model's full context length (e.g. 128K). | Worst — maximum waste. |

Batch-level padding, in the Admin's own worked example:

```
model sequence length = 4,096
8 examples in this batch; longest is 3,069 tokens
→ set this batch's length to 3,069, not 4,096
→ saves ~1,000 wasted positions on every one of the 8 examples
```

> This is better, because otherwise I have to add something like 1,000 more pad tokens **to all the examples**. That is a wastage of computation. Nothing is achieved there.

### Why padding is genuinely harmful, not just wasteful

> The model still has to predict empty, empty — and that's not a good thing, because **the model will learn that behaviour**. The loss is very low. If I have to predict 4,000 pad tokens, can you imagine the amount of loss reduction? **I've seen that.** If you ask the model to predict just empty 4,000 times, the loss is going to be zero, because it will do it effortlessly. It knows that after five empties, it's just empty.

**Consequence:** a padded run's loss curve is a *lie*. Your average loss looks great and the model has learned nothing. This is a direct argument for the ledger's "useful loss-bearing tokens" metric (§21).

### Q — Udit: "You said all GPUs must run the same length, but batch-level padding cuts to 3,069?"

> **Admin:** I didn't change what I said. I said **all GPUs must run for exactly the same number of sequences.** [All 8 GPUs use 3,069 for *this* step — they are still identical to each other. What varies is step-to-step, not GPU-to-GPU.]
>
> **Udit:** So it takes the highest across all of them, cuts there, and pads the rest?
>
> **Admin:** **There is no "it." You will need to decide to do it. It needs to be part of the data loader.**

### Q — Lakshmanarao: "Does sequence length matter at inference? I trained at 4,096 but serve at 2,048 and it works."

> **Admin:** I'll give you an analogy. You have a car that can go up to 100 km/h. You're driving at 60. So you're asking "does the sequence length matter?" **You're already under-utilising it.** The question you *should* have asked is: I've trained at 4K, can I now send 800K? **Less is always okay. The problem is more.**
>
> Have you tested your car at 4,000 km/h? Can it fly? Throw it from space — will it work? We don't know. **We have to train it to figure out whether it works at 4,000 km/h.**

> **▸ Added context.** This is correct and worth stating precisely: a decoder-only transformer can process *any* sequence up to its trained length without issue — shorter is always safe. Going **beyond** the trained length is where behaviour degrades, because the positional encoding is being asked to extrapolate to distances it never saw. Extending context after the fact is a deliberate, separate training phase ("long-context extension"), not something you get for free — which is exactly why the Admin insists (§18) on **reserving long documents for a later long-context stage** instead of chopping them at 4K.

### Q — Avnish: "Which padding logic is best / typically used?" & Q — Vardhan: "Is that dynamic batching?"

Both deferred: *"Wait for five minutes"* — because the answer is **packing**, not padding, which is the next topic.

---

## 18. Can we cut documents? (01:03 – 01:05)

Two questions the Admin poses and answers:

### "Can we cut a document in the middle of a line?"

> **For plain next-token pre-training — yes, fine.** We have a long document, we can cut in the middle. A Shakespeare novel is huge; the whole Vedanta book, the Bhagavad Gita, Mahabharata, Ramayana — these are long poems. **You can cut them in the middle. Not a problem while pre-training.**

> **However — a careless cut can damage.** You can't cut in the middle of **code**. You have a 16,000-token code file — don't cut it in between, you're losing it.

**And here is the important consequence:**

> That means you **cannot** train on a big code file at the beginning of training when your sequence length is only 4,096. **You need to preserve that shard for the future**, because in future you might train on a longer sequence.

> Moment you find datasets with documents longer than 4K, 16K, 32K — **you're lucky**, because now you can actually use it for that stage of training.

**So long documents are an asset to be reserved, not a problem to be truncated.**

### "Can we fill the remaining window with a different topic?"

> **At the pre-training stage — perfectly yes**, because the EOS token tells the model the context is switching. The attention logic will see the EOS and say "okay, I'm not going to focus on anything from before. I can focus on the next thing."

> **But for SFT, agentic data, reasoning traces — it's complicated. We may not like to do it.** Suddenly you have sunscreen and after that you have an agent trace. It needs flexibility in the SFT stage where we are not cramming different things together, because the logic there is **linked to retention** — we want the model in the agentic trace to continue to work on a long sequence.

> Those sequences are going to be **millions** of tokens. Here [pre-training] we're talking billions and trillions. So for agentic traces we won't face this problem, because we reduce the batch size — we train on **one long sequence** rather than many long sequences at once.

---

## 19. Packing policies (01:10 – 01:20)

> Packing is the process of filling fixed-length sequences with **useful** tokens.

### The five policies

**1. Pad-only**
> I just keep adding pad, pad, pad, pad. At college level that's okay. **At our level it's not**, because every run costs money. It preserves structure and you reach 4,096 — but you wasted a lot of computation, the model learned to predict pad-pad, and your loss will be very low.

> What if you were paid for a job where you just need to be on your chair for 8 hours? You'd sit on the chair for 8 hours and get the money, and you'd be one of the best employees. But what if we actually asked you to **do** something and validated you on that? **Padding is literally being on the chair for 8 hours.**

**2. Concatenate-and-chop**
> I take one document, take another document. A 3,000-word document plus another 3,000-word document, but my sequence is 4,096. So — 3,000 tokens more, and drop [the rest]. That is one strategy.

*Cost: you destroy the tail of every second document.*

**3. Greedy packing**
> Place each example in the first available sequence. Whatever comes first, we keep adding it, and we try to collect as many as possible.

> Some holes remain because **document order is preserved**. We had 45 documents; in that sequence we take document 1, put document 2, put document 3, until I've filled the full 4,096.

> **I'm not interested in packing efficiency — I'm interested in speed of making this dataset. You have to do this for 10 trillion tokens.** So greedy is the first thing that will occur to you.

**4. Best-fit packing**
> Sort the bucket by length and put things in such a way that we have the **tightest geometry** where documents are not [chopped]. Find those sequences that will not force me into bigger chops or a large number of chops. I would like to pack in as much as possible **without actually getting to a chop**.

> The best fit looks at the **whole dataset** and comes back with one of the best packings possible. It requires a bit of processing on the dataset — at least some statistics — to make sure we can pack efficiently. **And that is the best thing we actually are looking for.**

**5. Structure-preserving packing**
> That's something we're going to add to **SFT, where tool use and agentic data** live. Unreal[ated] examples are not going to be linked into each other.

> **Q — Manjith:** When do we use structure-preserving over best-fit?
> **Admin:** For SFT / agentic traces. Let's say I'm working with Claude Code and this is how one task is done — say we're making two different websites. When we train on the agentic part, we **do not do difficult stuff. We only make sure that everything is preserved properly.**

**6. Long-context packing** *(handled separately)*
> Long-context batches are expensive. When training for long context, **you cannot have a 4K sequence entering a 16K long-context training.** So you have to take those documents out.

### Greedy vs. best-fit — the Admin's analogy

> **Greedy packing is the packing you do five minutes before you have to leave for the airport.**
> **Best-fit packing is when you are changing countries** — you're going to think about what all should I add. You spend a lot of time thinking "is this the right way or not" to get the best thing into that 30 kg of allowance.

### The live widget comparison

He runs a demo with **context length 128K, 16 documents**:

| Policy | Result |
|---|---|
| **Pad each document** | Enormous visible waste. "All of you agree this is not useful, right?" |
| **Concat + chop** | Reduced to ~5 sequences |
| **Greedy** | Fewer still — "whatever comes first we keep adding" |
| **Best-fit** | **Least number of sequences. This is what we want.** |
| **Structure-preserving** | More sequences, but boundaries intact — required for agentic |

And the honest limitation:

> If your context length is 128K and your agent traces are only 32K, **you have to live with it.** We have to pad it. Fortunately the batch size is going to be small, so we won't waste a lot. So there we go back to that approach: the maximum for us is 59 — so I'll not do 128, **I'll stop at 59.** So there we need some sort of dynamic [batching].

### Q — Dattatreya: "What scale of data are we talking about? 10 million?"

> **Admin: 10 trillion tokens.**
> **Dattatreya:** Then it becomes a lot of compute. I'm assuming this is some sort of knapsack solution?
> **Admin:** *(assents)*

> **▸ Added context — yes, it is bin-packing.** Best-fit packing is exactly the **bin-packing problem**: items (documents) of varying size, bins (sequences) of fixed capacity, minimise the number of bins. Bin-packing is NP-hard, so nobody solves it optimally at 10 T tokens. The standard practical algorithm is **First-Fit-Decreasing (FFD)** or **Best-Fit-Decreasing (BFD)** — sort documents by decreasing length, then place each into the fullest bin that still has room. FFD is provably within ~11/9 of optimal, and runs in O(n log n) — which is why "sort by length, then place" is precisely the recipe the Admin describes. The Admin's own framing of the trade-off is the right one: **greedy buys throughput, best-fit buys utilisation**, and at 10 T tokens the cost of the sort is real.

### Q — Umesh: "Since each document is self-contained, is document order unimportant? Can a sequence contain any docs?"

> **Admin: I don't know. Answer is — you also don't know.** 90% of the time what you said is true. But the moment your **curriculum** shuffles, then you say "I want my coding shards separate, I want my Indic shards separate." **So the answer to your question goes back to Session 5 — what have you decided in Session 5?**

### Q — Manjith: span ID / trace ID from distributed tracing

Manjith proposes borrowing distributed-tracing concepts (a trace ID for the whole transaction, span IDs for each request within it) to implement structure-preserving packing.

> **Admin: That is a best-fit.**
> **Manjith:** But best-fit only has trace ID, not span ID.
> **Admin: What is a span ID?** … **How do I think of what you're saying in terms of documents?** This is a Wikipedia page on India. What is the span ID and the other ID for this?
> **Manjith:** …probably it's an out-of-context question.

*The Admin's point: the analogy breaks because a plain document has no internal request/response structure to hang spans off. It would only apply to agentic traces — which is precisely the case that already has its own policy.*

---

## 20. The shard manifest (01:25 – 01:30)

Every shard ships with a manifest. This is the "identity card" of a shard.

| Manifest field | What it records |
|---|---|
| **Shard ID** | e.g. "this is shard number 47" |
| **Source dataset** | Where is the information in this shard coming from |
| **Document IDs** | Which documents this shard was assembled from |
| **Content hash** | *"A signature of the whole [shard] — for me to figure out have I trained on this shard or not."* |
| **Token count** | How many tokens in this shard |
| **Language & script** | Is it mostly English, Python, or some script |
| **Capability lane** | From the Session 5 benchmark targets — e.g. "this is coding" |
| **License** | Is this something we can train on |
| **Provenance tier** | The link to that license / chain of custody |
| **Cleaning pipeline** | **What code was used to clean this, and the hash of that code file** — "so I can match that code also" |
| **Dedup status** | Yes — which batch/group of students deduplicated this |
| **Contamination status** | Yes, we have checked for contamination |
| **Eval / test overlap status** | It is not an eval shard and does not overlap with test |
| **PII removal status** | Personal information removed |
| **Parent / ancestor shard** | Is this shard part of a bigger shard, or independent |
| **Parent manifest / registry entry** | Which registry & **which curriculum stage** it belongs to |

### The minimum bar

> **We are not going to train on a shard that does not have** — as a minimum — the hash, the cleaning hash, dedup, eval [status], and PII [status].

### The PII example

> We would not like [a situation where] this shard has the address of [a person], and [the model] knows that [person A] and [person B] are friends and they went to a particular school. **Personal information needs to be removed.**

### Q — Nikhil: "Will we use the manifest to group batches with similar context?"

> **Admin:** What will happen for curriculum stage — you're going to say Stage B. So when we say **parent manifest**, that's a Stage B. **This shard belongs to Stage B.** But to *what*… which curriculum, which stage does it belong to? **We're going to save that part also.**

The Admin's own aside on this section:

> Any question? Right — **boring things don't have questions, and these are the things that matter.**

---

## 21. Compiling the mixture into a schedule (01:29 – 01:35)

> Session 5 gave us the mixture and the curriculum **in human terms.** Session 6 must convert that into something we can follow. **The schedule must know which stage the run is currently in.**

> This is not related to the dataset any more — but it's still related to the data-loader stage, because **this is the registry being built. This is that accounting book.**

> You cannot look at a shard as a standalone thing. We're looking at the whole training dataset as one. There's a **full bank** that we're building — and a bank has money, where the money is going, who's given a loan. All of that is required to be saved. **So we're talking about dataset and timeline together now.**

### What each stage record must contain

- Which stage the run is currently in
- **How many tokens** belong to this stage
- **Which capability lanes are activated right now** — "Are we focusing on coding right now, or Sanskrit, or general knowledge — because our GK score is not going up?"
- **What share each lane must receive** — coding bucket, language bucket, GK bucket, web bucket
- **What is protected right now** — are we protecting Indic, are we protecting agentic traces
- **Is there any protected item from Opus** (i.e. immune to Opus rejection)
- **What data is reserved for annealing** in this stage
- **How transitions are warmed up** — how fast or slow is the annealing

### The GK example (why this matters operationally)

> Your GK score is not going up — that's the **MMLU-Pro** benchmark. So you'll think: my GK is not improving, should I have more dataset for that? **Because today we don't know** whether, after 100 billion tokens of training, my general knowledge will improve or not.

> And without general knowledge, everything else is useless. A person can be a really good Python programmer, but ask him to write a simple function to check whether an elephant description is correct, and he'll fail — **because common sense says an elephant should have four legs.** If the programmer didn't know the common-sense thing, it can't check.

> **▸ Added context — MMLU-Pro.** MMLU-Pro is a harder, cleaned-up successor to MMLU (Massive Multitask Language Understanding). It is a multiple-choice academic-knowledge-and-reasoning benchmark spanning many subject areas, with more answer options and a stronger reasoning emphasis than the original, making it much less saturated for strong models. It is the standard stand-in for "general knowledge + reasoning breadth."

### A worked stage plan

> In the Balanced stage we will decide that, let's say, we have a **240 billion token budget**, [and] the warm-up band — how much is the warm-up between each of them.

> This information is required to figure out that at **batch number 4,698, *this* is the shard that is going to be trained on.**

Stages named across the session: **Balanced → Code-heavy → Indic → [later stages] → Annealing**, each with its own budget, lane weights, protected flows and warm-up band.

### The Opus rejection rate is a budget multiplier

> **Opus rejection rate is going to change the actual data we need to train on.** Let's say we're going to collect 10 trillion tokens. What is the Opus rejection rate we're going to go with? If the rejection rate is 50%, then our actual training is going to be 50% [i.e. 5 T].
>
> …That takes us to 40 trillion. That means — why don't we select even better data? We reject more from Opus. **These are the numbers we need to discuss. We've not decided yet what the Opus rejection rate is as of now for us.**

**The principle, stated cleanly:**

```
raw tokens you must COLLECT  =  tokens you want to TRAIN on  ÷  Opus acceptance rate
```

At 25% acceptance, a 10 T training budget requires collecting **40 T raw tokens**. Tighten the filter to get better data, and your collection requirement multiplies.

> But it is not meeting our shortfall. So we need to make sure we cover all the stages. **If you say you want 38 billion tokens and you don't have the dataset, it will fail from that point on.** This is common sense, but we still need to talk about it at 120-billion scale.

### And things break

> "I'm not doing good in GK, so let me add more GK data" — and those files become **unavailable**, because someone deleted them, or they're corrupted. **So we have to calculate again. We have to process again.**

---

## 22. The Data Ledger — full field list (01:35 – 01:36)

Every row the ledger writes as data flows *in*:

| Field | Meaning |
|---|---|
| **Run ID** | Which training run |
| **Batch ID** | Which global batch |
| **Global step** | Which optimizer step |
| **Checkpoint** | Which checkpoint this falls under |
| **Rank** | Which GPU/worker |
| **Microbatch** | Which microbatch on that rank |
| **Batch sample** | Which sample within the microbatch |
| **Shard ID** | Which shard it came from |
| **Tokens** | The tokens themselves |
| **Loss mask** | Which tokens carried loss |
| **Attention policy** | What masking scheme was applied |
| **Position policy** | What positional scheme was applied |
| **Mixture** | The lane ratios at this moment |
| **Curriculum** | Which stage |
| **Tokenizer** | Which tokenizer (hash) |
| **Data loader** | Loader state |
| **Opus decision lane** | Accepted / rejected / deferred / overridden |

> And all is required **just to debug how good we are doing.**

---

## 23. The Learning Ledger — what comes *back* from the model (02:00 – 02:10)

This is the half nobody builds, and the half the Admin most regrets not building in V4.

### Loss and perplexity, precisely

> **Perplexity is literally the reverse of loss.** If the loss is −log p of the next token, then perplexity is the exponential [of the loss].

```
loss(token)       = −ln P(correct token)
perplexity(token) = e^loss
```

### Where the initial loss comes from

The Admin derives this live with the coin analogy:

> I have a coin. I'll flip it. What accuracy can we have? **50%** — because it's either head or tail.
> So when my tokenizer is 30,000, what is the probability of any token when the model has not learned?
> **Sachin:** log 30,000.
> **Admin: One in 30,000. That's the loss you're going to see by default.**

The exact arithmetic, and the base correction he makes on screen:

```
log10(30,000)     = 4.477   ← WRONG BASE. "not that log — that log is with base 10"
ln(30,000)        = 10.31   ← correct; matches BERT-style pretraining curves starting ~10.3
ln(131,072)       = 11.78   ← our V4/V5 tokenizer (2^17 = 131,072)
```

> **If our tokenizer is 131,072 as we had for V4, that is the kind of loss we're going to see. Initially, when we start training, loss will start at 11.78.** This value will keep on reducing.

> **▸ Added context — why it's exactly ln(V).** At initialisation the model has no information, so it assigns uniform probability 1/V to every token in a vocabulary of size V. Cross-entropy loss is −ln(1/V) = **ln(V)**. So the initial loss is a pure function of vocabulary size — nothing to do with your data or architecture. Equivalently, initial **perplexity = V** = 131,072, meaning the model is "as confused as if choosing uniformly among 131,072 options." This is the single most useful sanity check in all of LLM training: **if your step-0 loss is not ≈ ln(vocab_size), your loss function, your masking, or your label alignment is broken.** The Admin's whole point in showing the BERT curve was that published papers rarely state both the loss curve *and* the vocab size, so this check is hard to do from the literature.

### Loss targets, as stated

> That number is very close to the end of model training. **We want to be this time ending at 1.8.** … If you can hit ~1.6, you're talking ChatGPT level. If you hit ~1.5x, we're talking Fable and that level. **That's how fast things have changed. 1.2 already means we can now enter at least a 2024 model stage.**

> **⚠ Transcript note.** The transcript renders these decimals as bare digits ("hit 6", "6.54"), dropping the leading "1.". The surrounding numbers (11.78 start, 2.3 mid-training average, 1.8 target, 1.2 cutoff) make the intended scale unambiguous. Treat the exact second decimals as approximate; the **ordering and the thresholds are the point.**

Reference conversion so you can reason in either unit:

| Loss | Perplexity (e^loss) | Interpretation |
|---|---|---|
| 11.78 | 131,072 | Step 0 — uniform guessing over the whole vocab |
| 3.4 | ~30 | Healthy mid-training shard |
| 2.3 | ~10 | Admin's example "current average at mid-stage" |
| 1.8 | ~6.0 | **V5 target at end of training** |
| ~1.6 | ~5.0 | "ChatGPT level" |
| **1.2** | **~3.3** | **Cutoff — do not train on this shard at this stage** |
| 0.3 | ~1.35 | **Alarm — this is boilerplate or contamination** |

### The core diagnostic the Admin builds

> We are in the mid-stage. That means we have seen something like **200 billion tokens.** My average loss right now happens to be **2.3**. We send a shard. We look at each token's loss as well. But on average, on the shard, **my loss is 1.2.**
>
> **What does it mean, guys?**
>
> *(students: "we already have similar context" / "already seen this" / "it's pretty confident")*
>
> **Already seen the concept before. It's not useful.**

> **If a shard is already at 1.2 and I'm in the middle of training and my average loss is 2.3, then I have just wasted my computation.**

**The rule:**

```
shard_avg_loss  ≥ 2.0   → fine, train on it
shard_avg_loss  ~ 3.4   → healthy, good learning signal
shard_avg_loss  ≤ 1.2   → CUTOFF. Model already knows this. Skip, or move to an earlier stage.
shard_avg_loss  ~ 0.3   → BROKEN. Something is very wrong. Investigate.
```

### The 0.3 case — the GNU licence example

> **That means something is really wrong. Maybe it's all garbage.** Do you know how easy it is to predict this? Have you seen this somewhere?
> *(Umesh: "Code commits")*
> Very easy to predict. **If you send code where a lot of it was the GNU licence — if you don't remove the GNU licence, the model will become very fluent at the GNU licence.** I was in sixth class when I learned this and I still remember it, and that's the best Hindi I can speak. The model will learn the whole GNU. It sees the C++ code where GNU was there and just blurts it out.

**Diagnostic value: suspiciously LOW perplexity is as informative as high perplexity.** It means repeated boilerplate, duplicated data, or leaked eval content.

### Why token-level, not just shard-level (the averaging trap)

The Admin works a concrete example on screen:

```
shard perplexity by region:  4.6 , 1.2 , 1.2 , ...
shard average               ≈ 2.0     ← looks fine!
```

> That is still **misleading** — that's why I need **token level**, so I can look at the **boundary of the document**. Did the document have a higher perplexity? **That's the actual objective.**

And why he wants inside-token detail specifically:

> Inside tokens will allow me to see **whether this was packed greedily** or, let's say, smartly.

**So the packing policy is auditable from the perplexity trace itself.**

### The reshuffle this enables

> So next time when I'm thinking of training, I'll try to do it in such a way that the **high-perplexity ones come in [at the right stage]**. Let's say this is two stages, A and B. So this is wrong here — this can be moved somewhere here. This is already at 2, so that may not be required to train on.

Q — Soma states it back and the Admin confirms:

> **Soma:** By knowing the perplexity and the phase in which it was scored, next time when you use the same shard you're not going to use it where it has already been learned — instead you'll put it in other phases.
> **Admin: Exactly.**

### Q — Sachin: "Do you annotate shards for early/mid/annealing phase?"

> **Admin:** We are going to annotate that this is for the beginning, end, middle and so on. **Unfortunately, [only] after the perplexity data will we know whether we were correct or not — and that is why I hate myself for not doing it in V4.**
>
> **Sachin:** But it's a chicken-and-egg problem. If you give a shard in the early phase the perplexity will be high; the same shard in a later phase will have low perplexity. So you're better using it in the early phases.
>
> **Admin: So it is chicken-and-egg for the first time. But after [that], it's not — we know whether it was a chicken or an egg.**

That is the entire justification for building the ledger even though *this* batch can't benefit from it.

### Q — Umesh: "How do we use perplexity to filter data, if you only get it after processing?"

> **That is why I said I'm so happy for the V6 batch. And I feel myself very stupid that it was not in my head when V4 was being trained — because it's literally maybe 10 lines of code and I could have done it. My mind was occupied in so many different places, I just couldn't remember to do it. That data would have been super important. I can't tell you how bad I feel.**

### Q — Avnish: "Are we starting from the V4 model as base?" / "Isn't perplexity model-specific?"

> **Admin: Not at all. That model has a different DNA.**
>
> On model-specificity: **all models are 90% the same.** The 10% structural difference is the difference between being an Olympian, or a PhD student, or a 12th-class student. But **all models have extremely similar behaviour.** Mistakes are made at different levels, and that's why the behaviours change.

*i.e. perplexity measured on model A is a usable — not perfect — signal for scheduling model B's curriculum.*

### The full learning-ledger field list

Per token:
- Token ID
- **Decoded text** ("if possible")
- Position in the packed sequence
- Document ID / shard
- Token loss → token perplexity
- **Perplexity at EOS specifically** — *"Someone is asking how do I know EOS is working or not. So we need to see the value at EOS. Suddenly if EOS comes in and the model has to predict EOS, it's going to say 'I'm not done.'"*
- Special-token perplexities: BOS, EOS, observation markers

Per batch / step:
- **Gradient norm** at that time
- **Gradient alignment** (if computed)
- **Opus score**
- **If I send the same batch again, what was my loss** — replay comparison
- **Model stage** — early / mid / annealing
- **Tokens consumed when this data was seen**
- **Useful / harmful classification**

> Not only what we are **sending** — we are also storing things we **get back** from the model. **It's a two-layer learning ledger.**

And it applies to eval too:

> The same thing needs to be done for the shards used for **test and evaluation**, because the same ledger is required when we're testing and evaluating.

### The annealing note

> When we are in the annealing stage, there are some things that are calculated for the loss and some that are not. **Same things that are hard [now] — as we proceed in the training, you're going to see they become easier and easier. So we need to know what the difficult tokens are.**

---

## 24. The Opus trail (01:45 – 01:55)

> Opus will reject dataset [samples]. We're sending 1,000 samples; our ratio is, let's say, 25%. So 250 [kept], 750 thrown away. **The rejections are very valuable.**

### The four questions the Opus trail must answer

1. **What is being thrown away?** The selector considered its value to be low.
2. **What did the protected flows rescue?** *"We always had Indic. We never took Indic to be part of the Opus selection, and hence we know it was [otherwise] thrown away."*
3. **What was the model already comfortable with?** The loss was very low, so we threw the data away.
4. **What may deserve review after later phases?** — the deferral pile.

### The gradient-distribution diagnostic (the sharpest idea in this section)

Track, for the samples Opus *selects*, the distribution of their gradient magnitudes over time:

```
Early — 1,000 candidates, 250 selected:
    gradient scores range 0.60 … 0.99,  average ≈ 0.9

After ~100 more steps — 1,000 new candidates, 250 selected (same 25% ratio):
    average ≈ 0.1,  with only a couple of samples at 0.5–0.6
```

> **How do you compare this? Suddenly this data is revealing a lot more to you. It's telling you the quality is bad.** Opus is selecting these 25% because **there's nothing else. It's just selecting the bad-quality data** — the old data is so bad, it *has* to select something.
>
> **That tells you the shard quality itself is very bad.**

**Key insight:** because the acceptance *ratio* is fixed at 25%, the selector will always return 250 samples — it cannot tell you the pool is exhausted. **Only the absolute gradient scores can.** A collapsing average with a fixed acceptance rate = your candidate pool is spent.

### The Opus decision log

Every decision gets a reason code:

| Decision | Reason |
|---|---|
| Selected | Above proxy threshold |
| Rejected | **Stage mismatch** |
| Rejected | **Below proxy threshold** |
| Kept anyway | **Protected-flow override** — "I'm anyway going to send it even if Opus rejects it" |
| Deferred | Review after later phases |

Plus, on every record: **shard ID, compatibility, train flag, curriculum stage, model age (early / mid / annealing).**

### Rejections are stage-dependent

> A shard that is 75% rejected by Opus in the annealing stage, [versus] in the SFT stage, is going to be different. **The same 25% selected might be very different when the model is being trained initially versus in the late stage.** So we need to score that, and that rejection quality is going to matter a lot.

> **This is the most important thing I think we can save. This is not shared by anyone. No company in the world is ever going to share you this data, and it can only be generated while the model is being trained.**

### Q — Swati: "If Opus selected the sample, why did perplexity go [low]?"

> **Because it's a good example.** Now let's say [perplexity] 0.3 — and this happened while Opus was also there. **What does it mean? How did a 0.3 sample enter my training while Opus was there? That means there's a problem in the Opus proxy.** Again — it helps me debug.
>
> **But if I don't have these numbers, I will just see a loss curve — which is what we were seeing in the last training. And that tells us nothing.**

### The economics analogies for "aggregates hide reality"

> Let's say the inflation rate — things are falling, and we have no idea *why* it's falling. It may be falling because the dollar is depreciating now, or things are actually improving. If the INR improves against the dollar, everything is going to be cheap — and let's say 50% of our GDP import is oil and weapons. **But on the ground…**

He then pulls up the shrinking-packet example live:

> Four biscuits only. … **85 gram — that means even 100 would have been the number when you were born, and today it's 60.** In some way I'm happy because I eat less junk, but that is me telling myself the world is not bad.

**The teaching point:** the headline number (price, or average loss) is stable while the *unit* silently shrinks. Averages conceal. **Token-level records are the only defence.**

---

## 25. Determinism, replay and fork — why the ledger exists (02:25 – 02:28)

This is Vardhan's question, and the Admin says it's the one he forgot to answer explicitly.

> **Vardhan:** You mentioned the ledger as a record and for reproducibility — deterministic loss trace. But there is some indeterminacy, some randomness. How is that captured if you don't capture a seed?
>
> **Admin: Exactly the point, and exactly the problem the ledger is solving. Thank you for asking it. I forgot to tell — why are we making a ledger.**

The full answer:

> Let's say we have code and we make everything deterministic. What are we trying to solve? We're trying to say that from step 1 to step 10,000, if I run this code I'm going to get exactly these shards. **But if I run it again, I may get some differences.** Something might still creep in.
>
> **That is why: the ledger is going to take this run and get this graph back.** Right now, if I want to go back in history and run something, **I will not run the code — because I know some non-determinism can come in. I'm going to run the ledger.** That shard was sent. So I'm going to **read and send. I will not calculate it.**
>
> Shard number 7, then 13 was sent — there was a sequence based on the code. For some reason, if you run the code again, [something different] might happen. **We don't want to risk that. That is why I'm going to store the sequence in which this was sent. The moment I store it, it becomes a ledger.** Next, I want to run from this particular step — I will get shard 4, because 4 was written there.

**The one-sentence principle:**

> **Do not re-derive the data order. Record it, and replay it.**

> **Vardhan:** So these numbers are about the shards being used, or the hyperparameters at that point of training?
> **Admin: No — you're talking about a *training* ledger. Today we are only discussing the *data* ledger.**

*(This distinction matters for the assignment: you are building the data ledger.)*

### Why bit-identical replay is needed for experiments

> Model number 1 showed this kind of loss because it was being fed these shards. If I retrain my model — **I'm going to draw on top of this line, because everything is deterministic.** I'm drawing exactly on top of it. **That is how the behaviour should be.** But if your behaviour is not this and is something like *this* — do you think I can figure out what happened?

The reason this is not academic:

> When we start talking about **mixture of experts**, I'm going to say "I'm going to use a loss of type A" and then "a loss of type B." When you change the loss, when you change the **pressure of how much each expert should be used**, when you change the **learning rate**, when you change the **ratio of the null expert** — there are a lot of changes we will do, and we'll compare for a short sequence. We take the model, do something to the training loop, run it for 100 steps. Then strategy B, run 100 steps.
>
> **In that, if we are not getting exactly the same sequence to train on, we cannot compare** — because suddenly the first shard was simple English and the third was all CUDA kernels or drivers. The behaviour will be different.

**So determinism is not tidiness. It is the precondition for A/B testing anything during a 45-day run.**

### The three capabilities the final system must have

> We want a system that can allow us to **resume, replay, and fork into a different branch** — so we can do experiments but still go back to the same starting point and start training from there.

| Capability | Meaning |
|---|---|
| **Resume** | Crash at step 3,005 → restart at exactly step 3,005 with the same next shard |
| **Replay** | Re-run steps 1,000–2,000 and get the identical loss curve |
| **Fork** | Branch at step N, try strategy A for 100 steps, rewind, try strategy B for 100 steps, compare fairly |

---

## 26. Why exact resume is genuinely hard (02:15 – 02:20)

> When you give the kill command — for example your checkpoint is at 3,000 steps — **when you give the kill command it will not get killed at 3,000. It might get killed at 3,005.** So you still have to go back and trace.

The realistic list of what goes wrong:

- Sometimes the service **hangs** and won't respond to a kill.
- Sometimes you just **forget**. Sometimes you're sleeping. Sometimes you're out in the market. Sometimes the phone is [off].
- **Training time is not fixed.** "Each step might take 30 seconds, but it might take 32 seconds also. When you finally see the AWS log, you're going to see some GPUs are hot, some are cool. So approximately something that should have taken 3 hours might take 3 hours 12 minutes — **and that 12 minutes you are in your washroom.**"

### The reasons you deliberately stop

> You've trained, now you switch to a new machine. Why? You're taking some calls. You have to stop. You have to benchmark. The dataset is not ready. **The next sequence itself is different.** The training loop was only made for pre-training.
>
> **Are you assuming we'll have one training loop? No — you will have different trainings**: post-training, RL, SFT, annealing and everything else.
>
> You need to change the batch size, because a bigger machine is available cheaper. You need to increase the sequence length. **You need to stop the model and do modifications to the model, because we are scaling the model** — or you're going to remove the RoPE, your position embedding.
>
> There are a lot of things we need to do structurally as we are training. **So you have to stop, and when we restart we have to train from the point exactly where we left. What is the meaning of "exact"? Only the ledger can tell us.**

### The night-shift warning

> If you're involved in training — **please remember you may not be able to sleep. Every run is going to be something like 23 hours. We need someone to be awake at night.** Give me a thumbs up. **This is real training.**
>
> Don't expect a 9-to-6 Saturday kind of job. When the training happens, **you have to take leave from your jobs.** Don't cry "I want to be part of the training team, I promise to commit all the time" — that means **I'll call you at 2 a.m. and you need to be [available].** That is the meaning of training.

---

## 27. Operational reality: cold starts, kernels, images (01:35 – 01:45)

### How long does it take to start a GPU?

The Admin asks the room. Answers: "1 minute" (Raj), "5 minutes" (Harshvardhan). AJ asks about spot vs. on-demand — the Admin rules it out: *"No, let's say you've got the machine."*

Then the person who has actually done it answers:

> **Shwetha:** Before we optimised everything it used to take **more than 30 minutes**, because we have to download the data to the NVMe and it's a lot of data. After we optimised it came down to **10 minutes**. **I have seen two hours also.**
>
> **Admin:** The worst case we've seen — some libraries were missing — I remember something like **9 hours** where it's just not happening.

> **This is after taking a lot of precautions. So it takes hours sometimes — and you're sitting on an H200 where it is consuming money. It's a very painful job.**

What must happen on every cold start:
- Provision the machine
- Download the data
- Download the last checkpoint
- Initialise everything
- **Make sure the exact same CUDA libraries are there**

> Even at 10 minutes, that still means you're going to waste something like **₹4,000 just to start a new instance** before you start training.

### The first week is pure optimisation

> We'll put the model on an H200 and see that it takes around **128 seconds per batch**. We have not even started training. This is the first step. We don't care about loss right now — but forward and backward propagation takes 128 seconds.
>
> **I think the initial 7 days we're just going to spend making this something close to maybe 10 seconds.** That is the optimisation we need. And **we're going to use agents for this** — that's what we did last time. We'll run multiple agents for a full 3–4 days on different batch sizes, different sequences.

### The non-linear surprise

> You'll realise a model might take **10 seconds for an 8K sequence** and **3 seconds for 4K** — and you'll be confused. You're thinking if I change from 4 to 8, this might become 3 → 6. **But it's not 6, it's 10, because exponential things are also involved.**

### Kernel fusion and the version trap

> In your model you had a **linear layer followed by a normalisation layer and then an activation**. What if I take all of them and convert that into a **kernel**? The moment you do, your **10 seconds becomes 1 second** for that step.
>
> **But that kernel required a particular CUDA kernel or library — version 1.3.69 — and you need to remember this.** When you start with a default AWS image you install some default library. You may have 1.3.70 instead — **and the model crashes.**
>
> So **you need to maintain your own image.** *(aside: "…of how society perceives you.")* **Docker will definitely help, but things still change, and sometimes we face those problems.**

> **▸ Added context — what "converting into a kernel" means.** Each separate GPU operation (linear → normalise → activate) launches its own CUDA kernel, and each one **round-trips its intermediate tensor through HBM**. Since these ops are memory-bandwidth-bound, most of the wall-clock time is memory traffic, not arithmetic. **Kernel fusion** compiles the chain into a single kernel that keeps intermediates in fast on-chip registers/shared memory, eliminating the round-trips. That is the source of the 10×. The cost is that fused kernels (whether hand-written, from a library, or emitted by a compiler like `torch.compile` / Triton) are tightly bound to specific CUDA, driver and framework versions — hence the Docker-image discipline the Admin insists on.

### Q — Raj: "Can we save the config on a network volume?"

> **Admin: No.** *(then, jokingly)* I'm telling you, I'll add you to a team that does this, and then I'll ask you: "okay, you were saying in session six that it's just making a network volume, right?" **It is not about data.** Every time you get a server — **it's a blank server. It's like installing Windows and all of your applications and all your code.**

### Q — Pranabesh: "Can a CPU pre-upload the data somewhere, then start the GPU?"

> **Admin: The model is also not saved in a single file. The model needs to be downloaded in parallel.** We've gone through the process where we were downloading the dataset and we saw the time crash from 30 minutes to a few seconds. There are a lot of learnings there — we're definitely going to start from there. **Last time we ended at somewhere around 10 to 15 minutes each start.**

### The real V4 numbers

| Item | Size |
|---|---|
| 120 B model **checkpoint** | ~**256–298 GB** (Shwetha: "It's 256 GB / 298 GB around — that is the model file") |
| **Dataset** | ~**4 TB** |
| **CPU RAM to rebuild a checkpoint** | ~**1 TB** — "to build a checkpoint and to decompress the checkpoint" |

---

## 28. ZeRO — fitting a model that doesn't fit (02:20 – 02:22)

The Admin sets up the problem with a question:

> Our GPU is 80 GB. Our model is 10 GB. **Can I fit the model in the GPU? Yes or no?**
>
> **The answer is no** — because that's just the model size. We need to think about the **activation size** too, and activations depend on the sequence length. Let's say sequence length 4K, and that takes something like 100 GB.

His worked numbers (he simplifies mid-explanation):

```
model parameters + optimizer state :  20 GB
activations                        : 140 GB
                              total: 160 GB     ← does not fit on one 80 GB GPU

8 GPUs × 80 GB                     : 640 GB total
160 GB / 640 GB                    : 1/4

→ if each GPU holds only 1/4 of the state, it fits.
```

> So both cannot sit on one GPU. Clear? So what are we going to do? We remember we're training on **eight** GPUs. What is the total RAM I have? **640 GB.** So somehow, if I can **shard** [the state] — **the first half of the model saved in [GPU] 1, the second in GPU 2** — the data is going to flow like this. **That is what allows us to actually pack it.**

> There are many methods of doing it. **Last time we did use ZeRO-1, ZeRO-2 and ZeRO-3 — all three were required at different stages.** That is where this **rank partitioning** comes in.

> **▸ Added context — the three ZeRO stages, precisely.** ZeRO (Zero Redundancy Optimizer, from DeepSpeed) removes the redundancy of every GPU holding a full copy of training state, in three cumulative stages:
>
> | Stage | Shards across GPUs | Each GPU still holds a full copy of |
> |---|---|---|
> | **ZeRO-1** | Optimizer states (Adam m, v) | Gradients, parameters |
> | **ZeRO-2** | Optimizer states **+ gradients** | Parameters |
> | **ZeRO-3** | Optimizer states + gradients **+ parameters** | Nothing — parameters are gathered on demand per layer |
>
> Memory savings grow with each stage; so does communication traffic, because ZeRO-3 must all-gather parameters for each layer during forward and backward. Note this shards **state**, not **activations** — activation memory is attacked separately (activation checkpointing, sequence/context parallelism), which is why the Admin's 140 GB activation figure is the harder half of his example. **The Admin's transcript renders these as "0, 01, 02"; he means stages 1, 2 and 3.**

---

## 29. Keeping the GPU fed (02:22 – 02:23)

> Once that is done, someone needs to work on this: we're still a data loader at heart, we're doing so many things on it — **we still need to make sure it is fast enough to keep the GPU hungry.**
>
> **If the GPU is stopped, if the GPU is waiting for the data loader, that's a waste of a lot of money. We can't have that.** That's why we're going to have a dry run or sample run where we see whether the data loader can feed fast enough.

Levers that determine loader throughput:

| Lever | What it controls |
|---|---|
| **Shard size** | Bigger shards = fewer network round-trips, more local disk needed |
| **Compression** | Trade CPU decompress time against network transfer cost |
| **Storage bandwidth** | NVMe vs. network volume vs. S3 |
| **Local caching** | What is kept on the box between steps |
| **Prefetch depth** | *"How many samples I am keeping available in the RAM — and GPU-RAM transfer is fast, so we can fill it up"* |
| **Worker count** | *"How many of the 96 CPU threads can I use?"* |
| **Rank partitioning** | *"How have I partitioned my rank / model into different GPUs"* |

---

## 30. The dashboard and the four services (02:10 – 02:12, 02:23)

### Four separate servers

> There are going to be **three services** we need to run to make the training happen … **and we have another server on which the benchmark will run. I forgot the fourth one.**

| # | Server | Job |
|---|---|---|
| 1 | **Training server** | Runs the actual training loop. Nothing else — it must not be slowed down. |
| 2 | **Dashboard server** | Takes all data and displays loss, logs, gradients, learning rate. "We are going to be glued to that screen." |
| 3 | **Data ledger service** | Takes a shard, records the run on it, and updates: "this shard behaved like this." **"We didn't do that last time — everything was manual."** |
| 4 | **Benchmark server** | *"The moment a checkpoint is done it is automatically uploaded"* → benchmarks run there and push results back to the dashboard: *"on this particular shard the benchmark improved by these points, and these are the shards that helped."* |

Why benchmarking must be a separate box:

> **The server on which we are training cannot be used for [benchmarking].** Benchmarking takes time. **And we'll continue to train also** — we'll not wait for the benchmark to come back and say it's a bad checkpoint, because of time. That's how we pay for GPUs.

### The dashboard has a scale problem of its own

> Let's say our batch takes 30 seconds. Every 30 seconds we'll keep seeing [updates] — **and it will crash, because there are just too many steps. We have seen that also.**
>
> Let's say we have 10 million steps and we want to show 10 million points. You're going to say 10 million is still not enough — because you'll have, say, **450 heads × 1 million steps**, gradients of those [400] experts. So we have 400 million [series] again. Then you want to see perplexity data. **So you might be downloading a billion points at the same time just to look at a dashboard. That is a service we need to make.**

### What the dashboard must show

| Metric | Why |
|---|---|
| **Raw tokens** | Total tokens pushed through |
| **Useful loss-bearing tokens** | Raw minus pads minus masked — the honest number |
| **Accepted tokens** | Survived Opus |
| **GPU idle time** | Are we paying for nothing? |
| **Loader wait time** | Is the data pipeline the bottleneck? |
| **Cache rate** | Local cache hits |
| **Shard rate** | Shards consumed per unit time |
| **Packing utilisation** | What fraction of each sequence was real content |
| **Rejection rate** | Opus throughput |
| **Replay & resume latency** | *"How long does it take to stop and start again?"* |

> This is sort of a precursor of what the final dashboard just for the data loader will look like. What sequences are we training on right now? What global batch can we hit? What is the packing efficiency and Opus rejection? **This data is going to be super useful for us while we're training — and for the next batch, definitely.**

---

## 31. Why Indic — the Admin's argument (02:30 – 02:37)

This comes from Raj's question: *"How is this token/tokenizer work useful in other industries — insurance, healthcare, finance, structured SQL?"*

The Admin's first answer is blunt: **"That is not our target."** He then states the target precisely:

> The model needs to be **really good in Indic** — and the definition of "really good in Indic" is **fertility of 2 for all the languages**, so we are not wasting compute there.
> Second, we want the model to be **really good at agentic work**.
> Third, it needs to be **really good at coding**.
> **That is exactly the domains in which models are being used.**

And he rules out the time-series framing explicitly:

> There is no LLM anyone is training which is going to look at the time-series data of 1 million points and then take a call.

### The bias argument

> This week you should spend some time understanding **the biases people have towards India** — how we think and how the outside world thinks are very different.

He pulls up a live example: the US had just added tariffs on India, citing **forced-labour import enforcement failures**. He then shows a model's own comparative answer about which countries "respect human rights," noting it praises Norway's competitive elections and independent courts while framing others as having screened candidates and unelected institutions —

> …all of that is true for Saudi Arabia also, all of that is true for Pakistan also — **but look at how the US behaves. All of that mentality is part of how the model behaves.**

> For each country you can do SFT and make sure I don't get the answers US people are getting — **but it remains relevant fundamentally: if you come with a bias towards a particular country, you think for that country.**

And the sovereignty point:

> Another important reason: **we cannot ask Fable to write a model that can allow us to hack** — [but] the US can.

### The "who is Indic for?" argument

Raj pushes: is the Indic tokenizer useful in future?

> **Do you want Indians to use it? How many people speak English?**
>
> **All of you should remember that you represent only 0.1% of this country.** The remaining [99.9%] — you have no thoughts about them. What they do in life, where they spend their life, what language, how they use their phones, how they're connected to the country. **The main backbone of the country — the whole agriculture, and the people in factories actually working — they're off from this whole technology.**
>
> **The moment you say Indic, please do not think about yourself.** If you're thinking about yourself, we already have ChatGPT and better tools. **We're talking about the mass structure here** — where all discussions happen, legal documents, land documents. Everything is in a native language.
>
> If you think we should all learn English because that's the uniting language — then again the bias comes in. **Then why not Kannada be the national language?**

### The security argument

> Let's say there's a terrorist attack. Do you think they're going to speak English? When they communicate, do you think they'll make sure they use English? **When we get those documents, we can capture it. We can look at it.**
>
> Or do you want Pakistan to start speaking English so we can use our AI there? What do you want China to do — start using English? **Do you really want India to not be able to scan and understand any other language?** Urdu is one of the Indian languages.

*(He also references a Karnataka government circular on language during this segment; the transcript is too fragmentary to reconstruct it reliably.)*

---

## 32. The assignment (02:23 – 02:30, 02:40)

> **The assignment is dense. I request you to read it very, very carefully — because now all of you are separately required to write the whole data ledger that we spoke about today.**

> It's not going to be as intense as what we're finally looking for, but **you will catch the points I have missed**, and you and your agents will come up with ideas and strategies. **You yourself will see what it takes to train a model at the class of OpenAI and others.**

### The full path your system must implement

```
tokenized shards
  → manifest
  → mix schedule
  → packing
  → batches
  → training            ("just fake training — send to a loop and come back")
  → consumption ledger
  → checkpoint
  → crash
  → resume
  → replay
  → audit
```

### The nine things it must demonstrate

| # | Requirement | Admin's words |
|---|---|---|
| 1 | **Immutable tokenized shards with manifests** | *"Immutable means cannot be changed. Once you write it — same code, if I run it on a different day, I will get exactly the same shard."* |
| 2 | **Frozen tokenizer + content hashes** | *"…and that is what you're using for the next stage as well."* |
| 3 | **Packing policy per data type** | *"For coding it [differs]. For agentic it's going to be different."* |
| 4 | **Correct loss mask, attention mask, position IDs** | *"You want the loss calculated for these tokens, you don't [for those]."* |
| 5 | **Curriculum stages + lane weights + protected flows** | *"…what you already decided in the last assignment. Indic and agentic traces you always want to be there."* |
| 6 | **Evaluation / validation firewall** | *"Have you made sure your shard stores the tag 'I am for testing or evaluation' so I don't train on it?"* |
| 7 | **Opus acceptance / rejection / deferral + protected-flow override** | *"Can you override Opus rejection?"* |
| 8 | **Training consumption + learning ledger, complete** | Both directions — in and out |
| 9 | **Token-level AND sample-level loss tracking** | Both granularities |

### An open design question he poses to you

> Last time, let's say 10% is Indic. **The question is: do we make a shard where we have 10% [Indic] and 90% [other], and when we send it to Opus we say "we don't care, we're still going to take it" — or do we keep the Indic separately?**
>
> **That is one question I've not thought about**, because the tokenizer wasn't ready early enough for me to do the curriculum for Indic also. Maybe there are sentences that are easier to train on compared to others; difficult words can come later, where the content is more general discussion or is actually very scientific or very law-heavy.
>
> **A lot of RBI and Indian government documents are in Indic languages. We can use that. But should we first train on that, or should we focus on simple news articles?** So **Indic might also become a part of** [the staged curriculum].

### Deliverables

| Deliverable | Requirement |
|---|---|
| **GitHub repo** | Submit the link on the assignment page |
| **README.md** | *"A short README that explains the architecture and your design decisions"* |
| **One command** | *"I'm going to clone your repo, run `python run_demo.py`, and I should get exactly what you got."* |
| **Small dataset** | *"Keep your dataset small. Don't make me download a terabyte of dataset."* |
| **Logs** | Execution logs covering all of the above |
| **JSON** | Structured output |
| **EVIDENCE.md** | Proof the requirements are met |

### Scoring

> I think the assignment is **1,150** — the other [150] is just for the links you provide, but **the first 1,000 is how it will be evaluated.**

### What's deliberately out of scope

> **This is not the whole thing**, because you're not writing the service, the dashboard, the UI, or the speed/throughput work. **But I want you to experience the whole thing end to end**: download the dataset, tokenize it, shard it, fix the sequence length — let's say 4K — then think about documents: how do I mix, how do I match, how do I add it to a framework.

### And the reason he won't just give you the code

> **This code is actually very simple. But if I share it back with you, you'll not go through this experience of understanding.**

---

## 33. The Admin's closing statement (02:30)

> If you've understood what we've done from sessions 1, 2, 3, 4, 5 and today — **you are ready for the actual model training.** This is the stage we should be in **before we start talking about transformers at all.**
>
> Get these stages right and the model is on its way. It's like you're a parent thinking "I've given the best teacher, best food, everything best to my child — what can go wrong?" That's the stage we are in. We're planning the best life insurance, everything for your son or daughter to go outside and study. **Nothing can go wrong after this.**

---

## 34. Complete Q&A index

| Time | Who | Question | Answer, in one line |
|---|---|---|---|
| 00:40 | lakshmanarao | Does grad-accum of 4 store info across all 4 runs? | Yes — that is why it's called *accumulation*; you accumulate 1,2,3,4 then back-prop |
| 00:44 | M | What is the ledger? | It's a data loader that **also records what comes back from the model** |
| 00:55 | Mukund | Is EOS part of the loss mask policy? | EOS must be at the end of the document; it is a token the model must learn to predict |
| 00:55 | Chandrahaas | Do you mask the highest-loss tokens? | No — masking is structural (prompt vs. answer), never value-based |
| 00:55 | Suresh | Why does only the agentic batch have a loss mask policy? | Loss masks come from **SFT onwards**; there is nothing to mask in pretraining |
| 00:55 | Abhishek | EOS at end of document or sentence? | **End of the document — end of the context** |
| 00:55 | Sachin | Do we have both EOS and BOS? | Both exist; modern practice uses **EOS only** — BOS is wasted space |
| 00:55 | Sachin | Samples span multiple documents — how does the model know? | You will never find 4,096 continuous usable tokens; chop + EOS is unavoidable |
| 00:55 | Pratik | Repeat the loss mask policy | Loss is not computed on the question, only on the answer |
| 00:55 | lakshmanarao | Significance of the 24-token window? | It's just the on-screen stand-in for 4,096 — "I don't have an infinite screen" |
| 01:05 | lakshmanarao | Does sequence length matter at inference? | **Less is always fine; more is the risk.** Car at 60 km/h vs. 4,000 km/h |
| 01:05 | Mukund | Won't the model learn boundaries anyway? | Back-prop *enforces* it, but EOS is the **signal** it keys off |
| 01:10 | Udit | Batch-level padding vs. "all GPUs same length" | All GPUs run the same length **as each other**; length may vary step to step |
| 01:10 | Dattatreya | Greedy fill for a 4,096 context? | Deferred — that's the packing section |
| 01:10 | Avnish | Which padding logic is best/typical? | Deferred — answer is packing, not padding |
| 01:10 | Vardhan | Is that dynamic batching? | Deferred — same |
| 01:15 | Dattatreya | What data scale — 10 million? | **10 trillion tokens.** And yes, it's a knapsack/bin-packing problem |
| 01:15 | Swati | How does the model know the context ends at EOS? | Back-propagation punishes cross-boundary attention and it learns to stop |
| 01:20 | Umesh | Is document order unimportant? | 90% of the time yes — **but your curriculum from Session 5 decides** |
| 01:20 | Manjith | When structure-preserving over best-fit? | For SFT/agentic traces, where the trace must stay intact |
| 01:20 | Manjith | Can we use span ID / trace ID from distributed tracing? | The analogy breaks — a Wikipedia page has no spans |
| 01:20 | Dattatreya | Is context switching from back-prop or from attention? | **You're confusing model *memory* with discussion *context*** |
| 01:20 | Harini | If a document is chopped, is there EOS? | Yes, but documents get lost in the token sea — **best-fit avoids the chop** |
| 01:25 | Harini | Can we compress tokens? | That's **tokenizer fertility** — Telugu at 13 tokens/word forces chopping |
| 01:25 | gaurav | Why both EOS and BOS? | Strategy-dependent; modern practice: **EOS only** |
| 01:30 | Nikhil | Use the manifest to group similar batches? | Yes — the **parent manifest** encodes the curriculum stage |
| 01:35 | Raj / Shwetha | How long to start a GPU? | 30 min → 10 min optimised; 2 hours seen; **9 hours worst case** |
| 01:40 | Raj | Save config on a network volume? | It's not about data — **it's a blank server, like reinstalling Windows** |
| 01:40 | Pranabesh | Pre-upload data with a CPU? | The model isn't one file — it must be downloaded **in parallel** |
| 02:10 | Umesh | How do we use perplexity to filter, if it comes after the fact? | **You can't — for V5. That's the V6 gift.** |
| 02:10 | Sachin | Do you annotate shards by phase? | Yes — and perplexity later tells you whether the annotation was right |
| 02:10 | Sachin | Isn't that chicken-and-egg? | **Only the first time. After that, we know which came first.** |
| 02:10 | Soma | So you reposition shards to phases they weren't learned in? | **Exactly.** |
| 02:15 | Avnish | Do we start from the V4 model? | **No — different DNA** |
| 02:15 | Avnish | Isn't perplexity model-specific? | **All models are 90% the same**; the signal transfers |
| 02:15 | Swati | Opus selected it, so why did perplexity go low? | **That means the Opus proxy is broken — and now you can see it** |
| 02:25 | Vardhan | Non-determinism — how is it captured without a seed? | **That is exactly why the ledger exists. Don't recompute — replay.** |
| 02:25 | Vardhan | Are those shard numbers or hyperparameters? | Shards — **that's the data ledger; the training ledger is separate** |
| 02:30 | Raj | Is this useful in insurance/healthcare/SQL? | **Not our target.** Target = Indic + agentic + coding |
| 02:35 | Raj | What's the future advantage of an Indic tokenizer? | Mass access, sovereign document understanding, security |

---

## 35. Glossary

| Term | Definition |
|---|---|
| **Annealing** | The gradual overlap between curriculum stages; also the final low-learning-rate, high-quality-data phase of training |
| **Attention mask** | Per-token record of what the model was allowed to / did look at |
| **Best-fit packing** | Sort documents by length and place each into the tightest-fitting sequence, minimising sequence count and chops |
| **BOS / SOS** | Beginning-of-sequence token. Largely abandoned in favour of EOS alone |
| **Capability lane** | A named skill bucket (code, Indic, GK, agentic) with its own share of the mixture |
| **Checkpoint** | Weights + optimizer state + scheduler state + RNG state + loader state + ledger offset |
| **Concat-and-chop** | Crude packing: join documents, cut at the sequence boundary, discard the overflow |
| **Contamination** | Eval/test data leaking into training data, invalidating benchmarks |
| **Curriculum** | The ordered plan of what the model learns when |
| **Data ledger** | The double-entry record of what went in **and** what came back. The deliverable of this session |
| **Deduplication** | Removing repeated content so every step teaches something new |
| **EOS** | End-of-sequence token. Marks a **context switch**, placed at the end of a **document** |
| **Evaluation firewall** | Two-sided enforcement that eval/test data is never trained on |
| **Fertility** | Tokens emitted per word. Target ≈ 2 for Indic; 13 (Telugu) is a failure |
| **Fork** | Branch a run at step N to A/B-test a strategy, then rewind |
| **Global batch** | `GPUs × microbatch × grad_accum` — the samples behind one optimizer update |
| **Golden proxy** | A small, high-quality reference set used to build the Opus weight map |
| **Gradient accumulation** | Run k forward/backward passes, sum gradients, then do **one** optimizer step |
| **Greedy packing** | Place each document into the first sequence with room. Fast, leaves holes |
| **Learning ledger** | The *outbound* half: per-token loss, perplexity, gradient norms, Opus scores, model stage |
| **Loss mask** | Per-token flag: does this token contribute to the loss? |
| **Manifest** | The identity card of a shard — source, hashes, license, dedup/PII/eval status, stage |
| **Microbatch** | Samples processed on **one** GPU before gradient accumulation |
| **MMLU-Pro** | The general-knowledge + reasoning benchmark used as the GK target |
| **Opus selection** | Model-in-the-loop filter that keeps only samples that move the weights the golden proxy says need moving |
| **Padding** | Filling unused sequence positions with meaningless tokens. Right / left / batch-level / fixed |
| **Perplexity** | `e^loss` — how surprised the model is. Init value = vocabulary size |
| **Protected flow** | Data (Indic, agentic) that is never dropped, and overrides Opus rejection |
| **Provenance** | Where a document came from and who produced it |
| **Rank** | The index of a GPU/worker in a distributed run |
| **Replay** | Re-running a segment from the ledger to get a bit-identical loss curve |
| **Shard** | An immutable, pre-tokenized ~1 B-token unit — ~1,000 training steps |
| **Span** | A slice of a document that fits inside one sequence |
| **Structure-preserving packing** | Packing that refuses to merge unrelated traces — required for SFT/agentic |
| **Training step** | **One optimizer update.** Not one forward pass |
| **ZeRO** | Sharding optimizer state (1) / + gradients (2) / + parameters (3) across GPUs |

---

## 36. Analogy index (the Admin teaches almost entirely through these)

| Analogy | Teaches |
|---|---|
| **NRI family speaking the native language at home** | Protected Indic flows must run **throughout** training, not in one burst |
| **You don't speak your 4th language because you don't practise** | Frequency of exposure, not one-time exposure, drives retention |
| **English class → PhD English handover** | Annealing: stage transitions are ramps, not switches |
| **Indian parents biasing you towards JEE/NEET** | Opus selection = deliberately biasing data towards the target exam |
| **The model getting US citizenship** | Half-joke about the target being a globally competitive model |
| **A for apple, B for ball — at this stage?** | A rejected sample may just be *scheduled wrong*, not bad |
| **The third-class student who hacked the school marks** | Benchmark contamination — implausibly fast improvement means a leak |
| **NEET paper leak / "write your own question and answer"** | Loss masking: you can't grade a student on the question |
| **Lights, camera, action** | BOS as an explicit start marker |
| **Different phone calls: support / bank loan / principal** | Humans track context switches — EOS is how the model does |
| **School bell between English and Hindi class** | EOS is a **reset** |
| **A person speaking 15 minutes and saying nothing** | Fluency ≠ knowledge; perplexity distinguishes them |
| **"Speed of light is an electromagnetic wave, blah blah"** | The model avoiding the one token that carries information |
| **Someone waking you at 5 a.m. for a flight** | Long-context training must keep attention "super tight" |
| **Money can't buy love; for everything else, Mastercard** | Every design decision resolves to cost |
| **Being paid to sit on a chair for 8 hours** | Padding: attendance without work |
| **Packing 5 min before the airport vs. changing countries** | Greedy vs. best-fit packing |
| **Test after every page vs. after every chapter** | Gradient accumulation |
| **"Beta IIT clear" in the village** | The Admin's ego about needing gradient accumulation |
| **Car rated for 100 km/h, driven at 60** | Inferring below trained sequence length is safe; above is untested |
| **Do you know how much X earned? Not today's context** | Model **memory** vs. discussion **context** |
| **Shrinking Maggi packet: 100 g → 85 g → 60 g** | Averages hide reality; you need token-level records |
| **GDP stored quarterly, not just 1950 and 2026** | The mixture ratio must be recorded continuously |
| **A bank ledger: money in, money out, who got a loan** | The data ledger is double-entry accounting |
| **A good Python programmer who doesn't know elephants have 4 legs** | Why GK/common sense underwrites every other capability |
| **The GNU licence you memorised in sixth class** | Suspiciously low perplexity = boilerplate/contamination |
| **A parent who gave the best teacher, food and life insurance** | Sessions 1–6 are the preparation; nothing should go wrong after |

---

## 37. Checklist — build this

```
[ ] Ingest a small corpus with per-document provenance + quality metadata
[ ] Freeze the tokenizer; record its hash
[ ] Tokenize → token IDs → spans
[ ] Implement packing: pad-only / concat-chop / greedy / best-fit / structure-preserving
[ ] Add EOS at every DOCUMENT boundary (not sentence)
[ ] Emit loss mask, attention mask, position IDs per sequence
[ ] Write immutable shards + full manifests (hashes, license, dedup, PII, eval, stage, lane)
[ ] Enforce the minimum manifest bar — refuse to train on a shard missing any required field
[ ] Build the mixture schedule: stages, token budgets, lane weights, warm-up bands, annealing reserves
[ ] Mark and honour protected flows (Indic, agentic) — they override Opus rejection
[ ] Implement the evaluation firewall on BOTH sides (writer tags it; trainer rejects it)
[ ] Implement Opus accept / reject / defer / override with reason codes
[ ] Run a fake training loop that returns per-token loss
[ ] Record the CONSUMPTION ledger: run, batch, step, ckpt, rank, microbatch, sample,
    shard, tokens, masks, policies, mixture, stage, tokenizer, loader, opus decision
[ ] Record the LEARNING ledger: per-token loss + perplexity, perplexity at EOS,
    gradient norm, opus score, model stage, useful/harmful
[ ] Checkpoint: weights + optimizer + scheduler + RNG + loader state + LEDGER OFFSET
[ ] Crash mid-run (deliberately) and resume at the exact step
[ ] Replay a step range from the ledger and prove the loss trace is identical
[ ] Audit: dump shard-level and token-level loss; flag shards below the 1.2 cutoff
[ ] README.md — architecture + design decisions
[ ] run_demo.py — one command, small dataset, does all of the above
[ ] Logs + JSON + EVIDENCE.md
[ ] Push to GitHub; submit the link
```
