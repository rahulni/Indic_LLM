# What we're doing here, explained simply

No maths background needed. If you can read a clock and you've heard a piano chord,
you already know everything required.

---

## First: computers can't read

A computer doesn't see the word `cat`. It can only work with **numbers**.

So before an AI can read anything, every word has to be turned into a list of numbers.
That list is called an **embedding**. It's the word's secret code.

```
cat  ->  [0.3, -1.2, 0.8, 0.1, ... ]      (a list of numbers)
```

Normally the AI **learns** these codes by reading millions of sentences, slowly
adjusting the numbers until they're useful. That works, but it takes a gigantic
memory: one list of numbers for every word it knows. Millions of words, millions
of lists.

**The big question of this project:** what if we *build* the code cleverly instead of
making the AI memorise it?

---

## What already existed: the locker room

Someone already invented a clever way to build these codes without learning them.
It works like a school locker room.

Picture **32 lockers** in a row. To store a word, put its first letter in locker 1,
the second letter in locker 2, and so on.

```
c a t
|  |  |
1  2  3   ...  lockers 4 to 32 stay empty
```

This is neat because nothing has to be learned — you just put letters in lockers.

But it has two problems, and this project is about both of them:

**Problem 1: short words waste lockers.** The word `a` uses one locker and wastes 31.

**Problem 2: long words get chopped.** There are only 32 lockers. A 40-letter word?
Letters 33 to 40 have **nowhere to go**. They're thrown in the bin.

That means two different long words that happen to start the same look **completely
identical** to the computer. It genuinely cannot tell them apart. (We measured this —
they come out exactly, perfectly identical. Not "nearly". Identical.)

> **A detail that matters for Hindi and Marathi:** the lockers actually hold *bytes*,
> not letters, and Indic letters need about 3 bytes each. So:
>
> | word | letters | lockers used |
> |---|---|---|
> | `cat` | 3 | 3 |
> | `elephant` | 8 | 8 |
> | `मराठी` | 5 | **15** |
> | `नमस्ते` | 6 | **18** |
>
> An English word gets ~32 letters before being chopped. A Marathi word gets about 10.
> The chopping problem is **three times worse** for Indic languages.

---

## Track A: teaching the code to do maths (the clock trick)

### The idea

Right now, the AI has to *learn* that 9 means nine. What if the number's meaning were
built into its code — so built-in that when you **add two codes together, you
automatically get the code for the answer?**

`code(9) + code(9)` should just *be* `code(18)`. No learning. It just happens.

### How clocks do it

You already know how a clock works: it goes 1, 2, 3... up to 12, then wraps back to 1.

Now imagine you have **five** clocks, and they're weird sizes: 11 hours, 13 hours,
17 hours, 19 hours, and 23 hours.

To store the number **20**, set each clock to where 20 lands on it:

| clock size | 11 | 13 | 17 | 19 | 23 |
|---|---|---|---|---|---|
| **20** points at | 9 | 7 | 3 | 1 | 20 |

To store **30**:

| clock size | 11 | 13 | 17 | 19 | 23 |
|---|---|---|---|---|---|
| **30** points at | 8 | 4 | 13 | 11 | 7 |

Here's the magic. **To add 20 + 30, just turn each clock forward.** Turn the first
clock 30 more hours, the second clock 30 more hours, and so on:

| clock size | 11 | 13 | 17 | 19 | 23 |
|---|---|---|---|---|---|
| after turning | 6 | 11 | 16 | 12 | 4 |
| where **50** lands | 6 | 11 | 16 | 12 | 4 |

**Identical.** Turning the clocks *is* adding. Nobody did any adding — no carrying
digits, no working out. Just turn each clock and read off the answer.

And it isn't a coincidence or an approximation. We checked **millions** of cases and
it was right **every single time**, exactly.

### The catch (there's always a catch)

Multiply those five clock sizes: 11 × 13 × 17 × 19 × 23 = **1,062,347**.

The trick works perfectly for any number below that. Go above it and the clocks wrap
around and you get the wrong answer — like how a clock can't tell you the difference
between 3 o'clock today and 3 o'clock tomorrow.

We also found the clocks are **bad at one thing**: they can't tell you which of two
numbers is *bigger*. The clock positions for 5 and for 900,000 look equally random.
To compare, you have to decode both fully first.

### Did it help the AI?

This is the honest part. **Mostly no.**

We built a small AI and gave it the clock codes. Then we asked it to do sums it had
never seen — it practised on numbers up to 4 digits, then we tested it on 5 and 6 digits.

It failed. And here's the uncomfortable bit: an AI given **no** number-position help at
all did *better* than the one with our fancy clocks.

Later we found we'd left out an important training trick, added it, and then the AI
did generalise — but a plain *learned* code still beat our clever clock code
(94% vs 34%).

**The lesson:** giving something the answer isn't the same as teaching it to read the
answer. The clocks are perfect. The AI just couldn't make much use of them.

That's a real result, and we wrote it down rather than hiding it.

---

## Track B: turning words into chords instead of lockers

### The idea

The locker room wastes space and chops long words. What if instead of putting each
letter in a **separate box**, we mixed them all into **one sound**?

Think of a piano. Press one key: one note. Press five keys at once: a **chord**.

A chord takes up the same amount of "space" whether it's 3 notes or 40 notes — it's
still just *one sound*. Nothing gets thrown away because there are no boxes to run out of.

So: **give every letter its own musical note, and play the whole word as a chord.**

(There's a clever detail: each letter also gets told *which position it's in*, so `cat`
and `act` make different chords. Same notes, different arrangement.)

### The catch

Play 3 notes together and you can pick out each one easily.

Play 40 notes together and it turns to **mush**. You can't tell what's in there anymore.

So we measured exactly how mushy it gets. And — this is the part we're proud of — we
worked out the mushiness **with maths first**, wrote down our prediction, and *then*
ran the experiment. The prediction matched the measurement almost exactly.

Roughly, with a medium-sized "ear":

| letters in the word | how often we hear each letter correctly |
|---|---|
| 4 | 100% |
| 16 | 91% |
| 32 | 64% |
| 40 | 55% |

Bigger ear (more numbers in the code) = less mush. It's a genuine trade-off, not a
free win — and we say so.

### Did it beat the lockers?

**No — and this is the most honest thing in the whole project.**

We trained both on Shakespeare and asked them to guess the next word. The old locker
method won clearly.

Why? Because **the longest word in Shakespeare is 15 letters.** The locker room has 32
lockers. It never once ran out! Our chord idea solves a problem that Shakespeare
doesn't have.

Our chord method does keep two real advantages:
- it never throws letters away, no matter how long the word
- it uses **zero** memorised numbers, where the locker method needs over 1.5 million

But on the actual test, it lost.

---

## The most important part: we found our own mistake

Near the end we got hold of the original paper describing the locker method — and
discovered **we'd built the locker version slightly wrong.**

We'd left out one step that made the lockers work worse than they really should.

Think about what that means. Our mistake was making **the other team play badly**. Our
own idea looked better than it deserved to.

We fixed it. And when we did, our idea looked **even worse** than before — the gap
against us doubled.

We reported that anyway. We kept the old numbers on the page right next to the new
ones, so anyone can see exactly what changed and why.

**That's the real lesson of this project.** It's easy to hunt for mistakes when results
look disappointing. It's much harder to go looking when results look *great* — because
why would you? But that's exactly when you should look hardest.

---

## Want to poke at it yourself?

Both pages have things you can play with, no installing anything:

- **[The clock page](https://rahulni.github.io/Indic_LLM/7_embed_research/track_a_numeral_crt/submission_artifacts/dashboard.html)** —
  type in two numbers and watch the five clocks spin round to add them.
- **[The chord page](https://rahulni.github.io/Indic_LLM/7_embed_research/track_b_holographic_binding/submission_artifacts/dashboard.html)** —
  type a really long word and watch the locker method chop the end off in front of you.

---

## The whole thing in six lines

1. Computers turn words into lists of numbers so they can read.
2. There was a clever way to build those lists: 32 lockers, one letter each.
3. It wastes room on short words and chops long ones — three times worse in Marathi.
4. **Track A** built numbers as spinning clocks, so adding happens by itself. The maths
   works perfectly. The AI mostly couldn't use it.
5. **Track B** built words as chords, so nothing gets chopped. It works, with a
   measured amount of mush — but it lost to the lockers on Shakespeare.
6. We found a mistake in our own work that had been unfairly helping us, fixed it,
   published the worse result, and showed our workings.

Two ideas that mostly didn't win, explained honestly, with every claim checked.
That's what real research usually looks like.
