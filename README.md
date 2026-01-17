# ARC-AGI-2 Solver

Two approaches to solving [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2):

| Solver | Approach | Best For |
|--------|----------|----------|
| **Primitives** | DSL + VLM verification | Pattern recognition, object manipulation |
| **Systems** | Python code synthesis + execution | Complex logic, counting, math |

## Quick Navigation
- [DSL Primitives Reference](#-dsl-primitives-reference)
- [Systems Solver (Code)](#-systems-solver-code-synthesis)
- [Primitives Architecture](#-primitives-architecture)
- [Usage](#-usage)
- [Setup](#-setup)

---

## 🧩 DSL Primitives Reference

The solver uses a Domain-Specific Language (DSL) with **primitives** that can be chained together. 
You can use **natural language** to describe arguments - the interpreter will convert them to structured parameters.

### Selection Primitives

| Primitive | Description | Examples |
|-----------|-------------|----------|
| `select(criteria="color", value=N)` | Select all cells of color N (0-9) | `select(criteria="color", value=3)` |
| `select(criteria="connected")` | Find all connected components | `select(criteria="connected")` |
| `select(criteria="largest")` | Select the largest object | `select(criteria="largest")` |
| `select(criteria="smallest")` | Select the smallest object | `select(criteria="smallest")` |
| `select(criteria="unique", value="colors")` | Find object with unique color set | `select(criteria="unique", value="colors")` |
| `select(criteria="unique", value="size")` | Find object with unique size | `select(criteria="unique", value="size")` |
| `select(criteria="size_rank", value=N)` | Select by size rank (0=smallest, -1=largest) | `select(criteria="size_rank", value=0)` |
| `select(criteria="enclosed", enclosing_color=N)` | Find regions enclosed by color N | `select(criteria="enclosed", enclosing_color=1)` |

**Free-form Selection** (NEW):
```python
select("the small colorful object that differs from the noise")
select("the largest connected component")
select("objects containing blue and green colors")
```

### Filter Primitives

| Primitive | Description | Example |
|-----------|-------------|---------|
| `filter(condition="area_eq", value=N)` | Keep objects with exactly N cells | `filter(condition="area_eq", value=9)` |
| `filter(condition="area_lt", value=N)` | Keep objects smaller than N cells | `filter(condition="area_lt", value=10)` |
| `filter(condition="area_gt", value=N)` | Keep objects larger than N cells | `filter(condition="area_gt", value=50)` |
| `filter(condition="has_colors", value=[...])` | Keep objects with ALL specified colors | `filter(condition="has_colors", value=[1,3,4])` |
| `filter(condition="touches_border")` | Keep objects touching grid edge | `filter(condition="touches_border")` |

**Free-form Filtering** (NEW):
```python
filter("keep only the smallest one")
filter("keep objects with area around 9 cells")
filter("keep the one with multiple colors")
```

### Painting Primitives

| Primitive | Description | Example |
|-----------|-------------|---------|
| `paint(color=N)` | Paint selected cells with color N | `paint(color=4)` |
| `replace(source_color=A, target_color=B)` | Replace all A with B everywhere | `replace(source_color=0, target_color=5)` |

### Transformation Primitives

| Primitive | Description |
|-----------|-------------|
| `transform(action="rotate_90")` | Rotate selection 90° clockwise |
| `transform(action="rotate_180")` | Rotate selection 180° |
| `transform(action="rotate_270")` | Rotate selection 270° clockwise |
| `transform(action="flip_horizontal")` | Flip selection horizontally |
| `transform(action="flip_vertical")` | Flip selection vertically |
| `gravity(direction="down")` | Drop objects in direction (down/up/left/right) |

### Extraction & Composition

| Primitive | Description | Example |
|-----------|-------------|---------|
| `extract()` | Crop grid to selection bounding box | `extract()` |
| `flood_fill(color=N, start_position=P, target_color=T)` | Fill from position with color | `flood_fill(color=5, start_position="border", target_color=0)` |
| `composite(source_sel, target_sel, mode="overlay")` | Combine selections | `composite(sel1, sel2, mode="overlay")` |

### Color Reference

| Code | Color |
|------|-------|
| 0 | Black (background) |
| 1 | Blue |
| 2 | Red |
| 3 | Green |
| 4 | Yellow |
| 5 | Grey |
| 6 | Pink |
| 7 | Orange |
| 8 | Cyan |
| 9 | Brown |

---

## 🔧 Systems Solver (Code Synthesis)

The **Systems** solver generates Python code to transform grids, then executes it in a sandbox.

### Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM 2: Manager                               │
│  HeuristicPolicy decides: FAST_GUESS_CODE or FAST_GUESS_INSTRUCTION       │
└────────────────────────────────────────┬──────────────────────────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────────┐
          ▼                              ▼                                  ▼
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│  Code Generator         │  │  Instruction Generator  │  │  VLM Critic             │
│  LLM → Python code      │  │  LLM → English steps    │  │  Visual verification    │
│  Sandboxed execution    │  │  LLM executes steps     │  │  "VALID" / "INVALID"    │
└─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘
```

### How It Works

1. **Code Generation**: LLM generates Python `def transform(input_grid) -> output_grid`
2. **Sandbox Execution**: Code runs in restricted namespace with 5s timeout
3. **Scoring**: Test on training pairs, score by cell-wise accuracy
4. **Refinement**: Failed code gets error message, LLM fixes it
5. **VLM Verification**: Visual check catches broken patterns

### Usage

```bash
python run.py --solver systems --task-id 00d62c1b
```

### When to Use Systems vs Primitives

| Task Type | Best Solver |
|-----------|-------------|
| Object extraction | Primitives |
| Color replacement | Primitives |
| Counting, math | Systems |
| Complex logic | Systems |
| Pattern matching | Both work |

---

## 🏗️ Primitives Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        ENSEMBLE PLANNER (VLM + LLM)                        │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌────────────────────┐  │
│  │ VLM Visual Planner  │  │ LLM Symbolic Planner│  │ Meta-Reviewer      │  │
│  │ (sees grid images)  │  │ (sees grid arrays)  │  │ (picks best plan)  │  │
│  └─────────────────────┘  └─────────────────────┘  └────────────────────┘  │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      FREE-FORM INTERPRETER (NEW)                           │
│  select("colorful object") → select(criteria="unique", value="colors")     │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        PRIMITIVE TRANSLATOR                                │
│  English → DSL Primitives: select() → filter() → extract()                │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        PRIMITIVE INTERPRETER                               │
│  ExecutionState: grid + selections → apply primitives → new state          │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     UNIFIED JUDGE (VLM + Logic)                            │
│  ┌─────────────────────┐    ┌───────────────────────────────────────────┐  │
│  │ VLM Verifier        │    │ Logic Judge                               │  │
│  │ Filmstrip → VLM     │    │ Compare output vs expected                │  │
│  │ "Does this look     │    │ Cell-by-cell accuracy                     │  │
│  │  correct?"          │    │                                           │  │
│  └─────────────────────┘    └───────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Ensemble Planning** | VLM + LLM generate plans in parallel, Meta-Reviewer picks best |
| **Free-Form Arguments** | Describe selections in natural language |
| **Expert Alternation** | If same expert fails 2x, automatically switch to other |
| **Dual Verification** | Both VLM and Logic Judge validate each step |
| **Feedback Loop** | Previous failures inform next attempt |
| **Filmstrip Rendering** | Visual step-by-step execution trace |

---

## 🚀 Usage

### Basic Run
```bash
python run.py --solver primitives --task-id 00d62c1b --preset balanced
```

### Ensemble Mode (Recommended)
```bash
python run.py --solver primitives --task-id 0a1d4ef5 \
  --vlm-model "Qwen/Qwen3-VL-235B-A22B-Instruct" \
  --ensemble --attempts 5
```

### All Options
```bash
python run.py --solver primitives [OPTIONS]

# Task selection
--task-id ID              # Solve specific task
--split {training,evaluation}  # Dataset

# Model selection
--vlm-model MODEL         # VLM for visual planning
--llm-model MODEL         # LLM for symbolic planning
--preset {fast,balanced,quality}  # Model preset

# Ensemble mode
--ensemble                # Enable VLM+LLM dual-path planning

# Feedback control
--attempts N              # Max retry attempts (default: 5)
--feedback-limit N        # Limit feedback to last N failures
--no-feedback {vlm,llm}   # Disable feedback for specific planner

# Output
--visual-planning         # Enable visual (VLM) planning
-v, --verbose             # Debug logging
```

### Model Presets

| Preset | Text Model | VLM | Speed |
|--------|------------|-----|-------|
| `fast` | Nemotron-30B | Qwen2.5-VL-32B | Fastest |
| `balanced` | Kimi-K2 | InternVL3-78B | Medium |
| `quality` | DeepSeek-V3.2 | Qwen3-VL-235B | Best |

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- Chutes API key

### Installation
```bash
cd arc_solver

# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Configuration
Create `.env`:
```bash
CHUTES_API_KEY=cpk_...
CHUTES_BASE_URL=https://llm.chutes.ai/v1
```

---

## Example Workflows

### Extract Unique Object
```dsl
1. select("the small multi-colored pattern that differs from the large blocks")
2. extract()
```

### Paint Connected Components
```dsl
1. select(criteria="connected")
2. filter(condition="touches_border")
3. paint(color=5)
```

### Find and Crop Smallest
```dsl
1. select(criteria="connected")
2. select(criteria="smallest")
3. extract()
```

---

## References

- [ARC-AGI-2 Dataset](https://github.com/arcprize/ARC-AGI-2)
- [Chutes API](https://chutes.ai) - Model inference
