# Transformer Interpretability: Attention Heads & Induction Circuits

A hands-on coding assignment exploring transformer interpretability through implementing attention mechanisms and induction heads from scratch.

## Overview

This assignment is part of **CS182: Deep Learning** and focuses on mechanistic interpretability—understanding *how* transformers work internally rather than just *what* they do.

You will implement:

1. **Single Attention Head** - The core attention mechanism with causal masking
2. **Induction Copy Head** - A two-head circuit that enables in-context learning

By the end, you'll understand how transformers can learn to recognize patterns like "A B ... A" and predict that "B" comes next.

## Background

### What are Induction Heads?

Induction heads are a key discovery in transformer interpretability research. They implement a simple but powerful algorithm:

> "If I've seen token A followed by token B before, and I now see token A again, predict B."

This is implemented through two attention heads working together:
- **Previous Token Head**: Copies information about what token came before each position
- **Copying Head**: Matches the current token with historical patterns and copies what followed

### References

- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (Olsson et al., 2022)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Elhage et al., 2021)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

## Getting Started

### Prerequisites

- Python 3.9+
- NumPy
- pytest (for running tests)

## Assignment Structure

```
transformer_interpretability_template/
├── core/
│   ├── attention.py          # TODO: Implement single_attention_head
│   └── induction_heads.py    # TODO: Implement induction_copy_head
├── utils/
│   ├── numerical.py          # Provided: softmax, create_causal_mask
│   └── constants.py          # Provided: EPSILON, VOCAB_SIZE, etc.
└── tests/
    ├── test_attention.py     # 54 tests for attention head
    ├── test_induction_heads.py # 21 tests for induction head
    └── test_numerical.py     # 12 tests for utilities
```

## Your Task

### Part 1: Single Attention Head

Implement `single_attention_head()` in `core/attention.py`.

**Input:**
- `attn_input`: Token embeddings of shape `(seq_len, d_model)`
- `wqk`: Pre-multiplied query-key matrix `(d_model, d_model)`
- `wov`: Pre-multiplied output-value matrix `(d_model, d_model)`

**Steps to implement:**
1. Convert inputs to numpy arrays
2. Compute pre-attention scores
3. Apply causal mask (use provided `create_causal_mask`)
4. Apply softmax (use provided `softmax`)
5. Compute value projections
6. Return attention-weighted output

### Part 2: Induction Copy Head

Implement three functions in `core/induction_heads.py`:

1. **`_build_previous_token_matrices()`** - Creates weight matrices for attending to the previous position
2. **`_build_copying_matrices()`** - Creates weight matrices for pattern matching
3. **`induction_copy_head()`** - Orchestrates the full induction mechanism

## Running Tests

Run the corresponding cells in the provided notebook!

## Academic Integrity

This is an individual assignment. You may:
- Discuss concepts with classmates
- Use the provided references and course materials
- Ask clarifying questions on Ed

You may not:
- Share code with other students
- Copy solutions from online sources

## License

This assignment is part of CS182 course materials. For educational use only.
