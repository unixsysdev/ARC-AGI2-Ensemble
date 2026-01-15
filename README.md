# ARC-AGI-2 Solver: System 1 / System 2 Architecture

A solver for the [ARC-AGI-2 benchmark](https://github.com/arcprize/ARC-AGI-2) using a dual-system architecture with visual verification.

## Table of Contents
- [Overview](#overview)
- [How ARC Works](#how-arc-works)
- [Architecture](#architecture)
- [The Learning Loop](#the-learning-loop)
- [Models Used](#models-used)
- [Setup](#setup)
- [Usage](#usage)

---

## Overview

This solver implements the **System 1 / System 2** cognitive architecture inspired by Daniel Kahneman's work and recent AI research (HRM, TRM papers). Instead of a single model trying to solve puzzles, we have:

| System | Role | Analogy |
|--------|------|---------|
| **System 1** | Fast, parallel "workers" that generate solutions | The hands doing the work |
| **System 2** | Slow, deliberate "manager" that decides strategy | The brain directing effort |
| **Visual Critic** | Quality control via vision-language model | The eyes checking the result |

---

## How ARC Works

### Data Format

Each ARC task is a JSON file containing:
```json
{
  "train": [
    {"input": [[0,1,2], [3,4,5]], "output": [[5,4,3], [2,1,0]]},
    {"input": [...], "output": [...]}
  ],
  "test": [
    {"input": [[...]], "output": [[...]]}  // output hidden during competition
  ]
}
```

- **Grids**: 2D arrays of integers 0-9 (colors), max 30×30
- **Training pairs**: 2-5 examples showing the transformation rule
- **Test pairs**: 1-2 inputs where you must produce the output

### The Challenge

The solver must:
1. **Observe** the training input→output pairs
2. **Infer** the transformation rule (pattern, logic, algorithm)
3. **Apply** that rule to test inputs
4. **Submit** 2 guesses per test input (ARC allows 2 attempts)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM 2: The Manager                              │
│  ┌──────────────┐    ┌─────────────────┐    ┌────────────────────────────┐  │
│  │ TaskState    │───▶│ HeuristicPolicy │───▶│ Action Dispatcher          │  │
│  │ - attempts   │    │ - decide()      │    │ - FAST_GUESS_CODE          │  │
│  │ - best_score │    │ - explain()     │    │ - FAST_GUESS_INSTRUCTION   │  │
│  │ - complexity │    └─────────────────┘    │ - DEEP_REFINE_*            │  │
│  └──────────────┘                           │ - POOLED_REFINE            │  │
│                                             │ - VLM_VERIFY               │  │
│                                             │ - SUBMIT / GIVE_UP         │  │
└─────────────────────────────────────────────┴─────────────┬──────────────┘
                                                            │
                    ┌───────────────────────────────────────┼───────────────┐
                    │                                       │               │
                    ▼                                       ▼               ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────────────┐
│  SYSTEM 1: Code Workers         │  │  SYSTEM 1: Instruction Workers          │
│  ┌─────────────────────────────┐│  │  ┌─────────────────────────────────────┐│
│  │ CodeGenerator               ││  │  │ InstructionGenerator                ││
│  │ - generate(task, n=30)      ││  │  │ - generate(task, n=30)              ││
│  │ - revise(task, code, error) ││  │  │ - revise_individual(inst, errors)   ││
│  │ Model: Qwen3-Coder-480B     ││  │  │ - revise_pooled(top_instructions)   ││
│  └─────────────────────────────┘│  │  │ Model: DeepSeek-V3.2-Speciale       ││
│  ┌─────────────────────────────┐│  │  └─────────────────────────────────────┘│
│  │ CodeExecutor (Sandbox)      ││  │  ┌─────────────────────────────────────┐│
│  │ - execute(code, input_grid) ││  │  │ InstructionExecutor                 ││
│  │ - test_on_training()        ││  │  │ - execute(instruction, input_grid)  ││
│  │ - timeout: 5s               ││  │  │ - test_on_training()                ││
│  │ - restricted namespace      ││  │  │ Model: DeepSeek-V3.2-Speciale       ││
│  └─────────────────────────────┘│  │  └─────────────────────────────────────┘│
└─────────────────────────────────┘  └─────────────────────────────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                          VISUAL CRITIC (VLM)                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ GridVisualizer → PNG → Qwen3-VL-235B → "VALID" or "INVALID: reason"     │  │
│  │                                                                         │  │
│  │ Catches: broken symmetry, missing objects, random noise, visual errors │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## The Learning Loop

Unlike traditional ML training, ARC-AGI uses **test-time compute** - the "learning" happens at inference time for each specific task.

### Phase 1: Initial Exploration (Attempts 1-30)

The Policy alternates between code and instruction generation:

```
for attempt in range(30):
    if attempt % 2 == 0:
        → FAST_GUESS_CODE: Generate 5 Python solutions
           └─ CodeGenerator.generate(task, n=5)
           └─ CodeExecutor.test_on_training() → score 0.0-1.0
    else:
        → FAST_GUESS_INSTRUCTION: Generate 5 English instructions
           └─ InstructionGenerator.generate(task, n=5)
           └─ InstructionExecutor.test_on_training() → score 0.0-1.0
    
    → Update TaskState with best solutions
    → If score == 1.0: SUBMIT immediately
```

### Phase 2: Refinement (Attempts 31-80)

Once we have candidates, we refine the best ones:

```
→ DEEP_REFINE_CODE: Take top-3 code solutions
   └─ Get the error messages from failed training pairs
   └─ CodeGenerator.revise(code, error) → improved code
   └─ Re-test on training pairs

→ DEEP_REFINE_INSTRUCTION: Take top-3 instructions
   └─ Get (input, expected, actual) error tuples
   └─ InstructionGenerator.revise_individual(inst, errors)
   └─ Re-test on training pairs

→ POOLED_REFINE: Synthesize from top-5 instructions
   └─ Show all 5 with their scores
   └─ Ask model to create a BETTER instruction
   └─ InstructionGenerator.revise_pooled([(inst, score), ...])
```

### Phase 3: Verification (When score > 0.9)

Before submitting, verify with visual critic:

```
→ VLM_VERIFY:
   └─ GridVisualizer.render_comparison(input, candidate_output)
   └─ VLMCritic.verify(image) → "VALID" or "INVALID: broken symmetry"
   └─ If INVALID: penalize score by 50%
   └─ If VALID: boost score by 10%
```

### Final Submission

```
→ SUBMIT: Take best 2 unique solutions
   └─ For each test input:
       └─ Execute best code/instruction on test input
       └─ Return top-2 distinct outputs
```

---

## Scoring: How Solutions Are Evaluated

### Training Time Scoring
Solutions are scored on training pairs using **cell-wise accuracy**:

```python
def score_grid(candidate: Grid, expected: Grid) -> float:
    correct_cells = sum(c == e for row in zip(candidate, expected) for c, e in row)
    total_cells = height * width
    return correct_cells / total_cells  # 0.0 to 1.0
```

### Leave-One-Out Cross Validation
For code solutions:
```python
score, errors = CodeExecutor.test_on_training(code, task.train)
# Runs code on ALL training inputs
# Returns average score across pairs
```

A score of **1.0** means perfect on ALL training pairs - this solution is likely correct.

---

## Models Used

All models accessed via [Chutes API](https://llm.chutes.ai) (OpenAI-compatible):

| Model | Role | Why This Model |
|-------|------|----------------|
| `deepseek-ai/DeepSeek-V3.2-Speciale-TEE` | **Primary Reasoner** | Best open-source reasoning for abstract patterns |
| `moonshotai/Kimi-K2-Instruct` | **Fallback Reasoner** | Strong thinking model, used when DeepSeek fails |
| `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE` | **Code Generator** | SOTA for Python synthesis |
| `Qwen/Qwen3-VL-235B-A22B-Instruct` | **Visual Critic** | Multimodal verification |

### Model Fallback Chains

The solver automatically tries alternate models if the primary fails:

```
Reasoner: DeepSeek → Kimi K2 → Qwen-Coder
Coder:    Qwen-480B → Qwen-30B
```

### Why Two Approaches?

**Python Code** (System 1 - Code):
- ✅ Deterministic - same input always gives same output
- ✅ Fast to test - no LLM call needed after generation
- ✅ Precise - no interpretation errors
- ❌ Brittle - complex patterns hard to express

**English Instructions** (System 1 - Instructions):
- ✅ Flexible - can describe abstract transformations
- ✅ Easier to revise - natural language feedback
- ❌ Non-deterministic - LLM may interpret differently
- ❌ Slower - requires LLM call to execute

The hybrid approach uses code when possible, instructions when patterns are too abstract.

---

## Setup

### Prerequisites
- Python 3.11+
- Chutes API key (configured in `.env`)

### Installation
```bash
cd arc_solver

# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Configuration
The `.env` file is pre-configured:
```bash
CHUTES_API_KEY=cpk_...
CHUTES_BASE_URL=https://llm.chutes.ai/v1
MAX_CONCURRENCY=20
```

---

## Usage

### Single Task
```bash
# Verbose mode to see all actions
python -m src.run --task-id a47bf94d -v
```

### Batch Evaluation
```bash
# Run first 10 evaluation tasks
python -m src.run --split evaluation --limit 10

# Run all 120 evaluation tasks
python -m src.run --split evaluation

# Run with offset (for resuming)
python -m src.run --split evaluation --offset 50 --limit 20
```

### Training Data (for development)
```bash
python -m src.run --split training --limit 5
```

### Output
Results are saved to:
- `attempts/<task_id>.json` - Per-task results
- `attempts/arc-agi_evaluation_attempts.json` - Competition submission format

---

## File Structure

```
src/
├── run.py                    # CLI entry point
├── config.py                 # API keys, model settings
├── models/
│   └── task.py               # Grid, Task, Pair data structures
├── llms/
│   ├── chutes_client.py      # Async OpenAI-compatible client
│   └── prompts.py            # All LLM prompt templates
├── system1/                  # "Workers"
│   ├── code_generator.py     # Python solver synthesis
│   ├── code_executor.py      # Sandboxed execution (5s timeout)
│   ├── instruction_generator.py  # English instruction synthesis
│   └── instruction_executor.py   # LLM-as-computer
├── system2/                  # "Manager"
│   ├── state.py              # TaskState, Action enum
│   ├── policy.py             # HeuristicPolicy decision logic
│   └── controller.py         # Main orchestration loop
├── critic/                   # Visual verification
│   ├── visualizer.py         # Grid → PNG renderer
│   └── vlm_critic.py         # Qwen3-VL verification
└── utils/
    └── grid_diff.py          # ASCII diff for debugging
```

---

## 🚀 Hybrid Mode (Local + Remote)

The solver supports a **hybrid approach** combining fast local exploration with quality remote refinement:

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HYBRID LOCAL-REMOTE MODE                              │
│                                                                              │
│  Phase 1: LOCAL EXPLORATION (Fast)                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ vLLM (GPT-OSS-20B) → 64 candidates @ 256 tok/s → Filter to viable ones  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    ▼                                         │
│  Phase 2: SANDBOX TESTING                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Execute all candidates → Score on training pairs → Keep top 5           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    ▼                                         │
│  Phase 3: REMOTE REFINEMENT (Quality)                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Qwen-480B refines top candidates with detailed error feedback            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    ▼                                         │
│  Phase 4: VLM VERIFICATION + VLM-IN-THE-LOOP                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Qwen3-VL verifies visually → If rejected, send reason to remote → Fix   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Local vLLM Setup (AMD ROCm)

```bash
# Start vLLM server (64 concurrent for max throughput)
export PYTORCH_TUNABLEOP_ENABLED=1
export HIP_FORCE_DEV_KERNARG=1

vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
```

### Running Hybrid Mode

```bash
# Basic hybrid run (64 candidates, 1 batch)
python -m src.run --split training --task-id 0c786b71 --hybrid

# Custom settings
python -m src.run --hybrid --local-concurrency 64 --hybrid-candidates 64

# Full verbose logging
python -m src.run --hybrid -v
```

### Voting & Diversity (Poetiq-style)

The solver groups identical outputs and ranks by vote count:

```
VOTING - 15 unique outputs from 64 candidates
  Group 1: 12 votes, best score 0.95  ← Most likely correct!
  Group 2: 8 votes, best score 0.90
  Group 3: 5 votes, best score 0.85
```

**Key insight**: If 12/64 candidates produce the *same* output, it's probably correct!

### VLM-in-the-Loop Revision

When VLM rejects a candidate, the rejection reason is sent to the remote LLM for targeted fixes:

```
VLM: "INVALID: output dimensions don't match expected pattern"
   ↓
Remote receives: "VLM visual inspection found: output dimensions don't match expected pattern"
   ↓
Remote fixes the specific issue in the code
   ↓
Re-test and re-verify
```

---

## CLI Reference

```bash
python -m src.run [OPTIONS]

# Dataset selection
--split {training,evaluation}  # Which dataset (default: evaluation)
--limit N                      # Max tasks to solve
--offset N                     # Skip first N tasks
--task-id ID                   # Solve specific task

# Model selection
--reasoner {deepseek,kimi,qwen-reasoning}  # Reasoner model
--coder {qwen-coder,qwen-coder-small}      # Coder model

# Local LLM
--use-local                    # Use local vLLM only
--local-url URL                # vLLM server URL (default: http://localhost:8000)

# Hybrid mode
--hybrid                       # Enable hybrid local+remote mode
--local-concurrency N          # Max parallel requests (default: 32)
--hybrid-candidates N          # Candidates to generate (default: 64)

# Other
--list-models                  # Show available models
-v, --verbose                  # Debug logging
--dry-run                      # Load only, don't solve
```

---

## Performance Tuning

### AMD Strix Halo (128GB Unified Memory)

| Setting | Value | Reasoning |
|---------|-------|-----------|
| `--max-num-seqs` | 32 | Sweet spot for throughput vs memory |
| `--gpu-memory-utilization` | 0.85 | Leave headroom for spikes |
| `local_concurrency` | 32 | Match vLLM setting |
| `hybrid_candidates` | 64 | 2 batches of 32 |

**Expected throughput**: ~256 tok/s with 32 concurrent

### Environment Variables

```bash
export PYTORCH_TUNABLEOP_ENABLED=1  # Enable kernel tuning
export PYTORCH_TUNABLEOP_TUNING=1   # First run creates tunableop_results0.csv
export HIP_FORCE_DEV_KERNARG=1      # ROCm optimization
export TORCH_BLAS_PREFER_HIPBLASLT=1  # Use hipBLASLt for GEMM
```

---

## Future Enhancements

1. **RL-Trained Policy**: Replace `HeuristicPolicy` with PPO-trained neural network
2. **Test-Time Training (TTT)**: Fine-tune model on each task's training examples
3. **Ensemble Voting**: Run multiple strategies and vote on best answer
4. **Synthetic Data**: Pre-train on millions of generated ARC-like tasks

---

## References

- [ARC-AGI-2 Dataset](https://github.com/arcprize/ARC-AGI-2)
- [Jeremy Berman's arc-lang](https://github.com/jerber/arc-lang-public) - Inspiration for evolutionary approach
- [HRM Paper](https://arxiv.org/abs/2506.21734) - Hierarchical Reasoning Model
- [TRM Paper](https://arxiv.org/abs/2510.04871) - Tiny Recursive Model

