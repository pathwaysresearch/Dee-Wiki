---
type: synthesized
aliases: ["python-curriculum", "learn-python-data-science"]
tags: ["python", "pedagogy", "curriculum", "data-analysis", "programming"]
relationships:
  - target: anonymous-lambda-function
    type: extends
  - target: array-oriented-programming
    type: extends
  - target: array-based-computing
    type: extends
  - target: accumulation-methods-in-pandas
    type: extends
  - target: apply-method-pandas
    type: extends
---

# Python for Data Analysis: Ten-Stage Curriculum

# Python for Data Analysis: Ten-Stage Curriculum

A sequenced ten-stage progression from Python fundamentals to production-ready data analysis. Each stage builds on the last, and the ordering is deliberate: concepts introduced early reappear with greater depth later.

---

## Stage 1 — Python Foundations
**Core ideas:** variables, types, control flow, indentation.

Python's syntax is intentionally readable. Master `if/elif/else`, `for`, and `while` before anything else. These are the grammar before the literature.

---

## Stage 2 — Data Structures
**Core ideas:** lists, tuples, dicts, sets, comprehensions.

Python's built-in containers are expressive. List comprehensions (`[x**2 for x in range(10)]`) foreshadow the array-oriented style you will use constantly in NumPy and pandas.

---

## Stage 3 — Functions
**Core ideas:** defining functions, scope, return values, default arguments, docstrings.

Functions are the unit of reusable logic. A function packages a computation so it can be named, tested, and composed. This is the foundation for everything: lambda functions are anonymous functions, `apply` passes functions to data frames, and generators are functions with memory.

**Key habit:** write small, single-purpose functions. If it needs more than one screen, decompose it.

---

## Stage 4 — Anonymous (Lambda) Functions
**Core ideas:** `lambda` syntax, one-liner transformations, passing functions as arguments.

Lambda functions (`lambda x: x * 2`) are functions without names. They are not more powerful than regular functions — they are more *concise* for short, throwaway operations.

**When to use:** passing a simple transformation to `sorted()`, `map()`, `filter()`, or pandas `apply()`. When logic grows beyond one expression, write a named function instead.

---

## Stage 5 — Generators and Iterators
**Core ideas:** `yield`, lazy evaluation, memory efficiency, generator expressions.

A generator is a function that remembers where it paused. Instead of building a full list in memory, it produces values one at a time. This matters enormously when data is large.


def squares(n):
    for i in range(n):
        yield i ** 2


Generator expressions (`(x**2 for x in range(n))`) mirror list comprehensions but stay lazy.

---

## Stage 6 — NumPy and Array-Oriented Programming
**Core ideas:** ndarrays, vectorized operations, replacing loops with array expressions.

Array-oriented programming means expressing logic over *whole arrays* rather than element-by-element loops. NumPy operations dispatch to compiled C code, making them orders of magnitude faster.


import numpy as np
a = np.array([1, 2, 3, 4])
print(a * 2)          # [2 4 6 8]  — no loop written
print(a[a > 2])       # [3 4]      — boolean indexing


The mental shift: stop thinking "iterate over elements" and start thinking "describe the transformation on the whole structure."

---

## Stage 7 — Array-Based Computing
**Core ideas:** broadcasting, `np.where`, `np.vectorize`, replacing conditionals with array logic.

Array-based computing extends Stage 6 by encoding conditional logic inside array operations.


np.where(a > 2, 'big', 'small')  # ['small','small','big','big']


Broadcasting allows arithmetic between arrays of different shapes by automatically stretching dimensions — a subtlety worth studying carefully because it enables concise code but produces silent errors if misunderstood.

---

## Stage 8 — pandas: Series and DataFrames
**Core ideas:** labeled indexing, alignment, reading/writing data, basic exploration.

pandas wraps NumPy arrays in labeled structures. Alignment by index is the killer feature: operations between two Series automatically match on labels, not position.

Start with: `pd.read_csv()`, `.head()`, `.info()`, `.describe()`, `.value_counts()`.

---

## Stage 9 — Reduction vs. Accumulation in pandas
**Core ideas:** aggregation methods that collapse data vs. methods that preserve shape.

Two families of operations are easily confused:

| Family | Shape change | Examples |
|---|---|---|
| **Reduction** | Many rows → one scalar | `.sum()`, `.mean()`, `.max()`, `.groupby().agg()` |
| **Accumulation** | Same shape, running result | `.cumsum()`, `.cumprod()`, `.cummax()` |

Reductions answer "what is the total?" Accumulations answer "what was the running total at each point?"

The same distinction applies to NumPy: `np.sum(a)` vs. `np.cumsum(a)`.

**GroupBy pattern:**

df.groupby('category')['revenue'].sum()    # reduction per group
df.groupby('category')['revenue'].cumsum() # accumulation per group


---

## Stage 10 — Pipelines, Apply, and Production Patterns
**Core ideas:** `apply()`, `applymap()`, chaining, scikit-learn `Pipeline`, code organization.

At this stage, all prior concepts converge:
- **Lambda functions** (Stage 4) are passed to `apply()`.
- **Array-oriented thinking** (Stage 6–7) guides vectorized alternatives to `apply()` when speed matters.
- **Generators** (Stage 5) appear in custom iterators for large datasets.
- **Reductions and accumulations** (Stage 9) compose naturally in `groupby().apply()` workflows.

scikit-learn `Pipeline` chains preprocessing and modeling into a single object, preventing [[data-leakage-in-applied-data-science|data leakage]] by ensuring transformations are fit only on training data.

---

## Pedagogical Notes

**The core progression follows a single thread:** *name a computation → make it anonymous → make it lazy → apply it to whole arrays → apply it to labeled arrays → compose it in pipelines.*

Stages are not waterfall phases — revisit earlier stages with new eyes. After Stage 6, re-read Stage 3: functions become more interesting once you can pass them to vectorized operations.

**Common traps by stage:**
- Stage 4: over-using lambda where a named function would be clearer.
- Stage 7: broadcasting errors from wrong axis assumptions.
- Stage 9: confusing `.transform()` (accumulation-like, group-preserving) with `.agg()` (reduction).
- Stage 10: fitting transformers on the full dataset before the train/test split — the canonical leakage error.
