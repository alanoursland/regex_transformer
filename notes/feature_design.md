# Feature Design Document - OUTLINE

## 1. Goals

### 1.1 Phase 1 Objectives

The primary goal of Phase 1 is to **build working infrastructure and validate that the approach works**. We need a complete end-to-end system before we can do interpretability analysis.

#### Core Deliverables

**1. Regex/FSM System**:
- Build a system that can parse regex patterns into minimal finite state machines
- FSM must support both recognition (classify strings) and generation (produce valid strings)
- Need complete FSM operations: state tracing, transition enumeration, serialization
- Must work correctly and reliably - this is foundational for everything else

**2. Data Generation Pipeline**:
- Generate labeled training data from FSMs
- Each training example has:
  - Input string (character sequence)
  - Next-token labels (what character comes next at each position)
  - State-class labels (what state class we're in at each position)
- Ensure good coverage of FSM states and transitions
- Support train/validation/test splits with reproducible seeding

**3. Transformer Model**:
- Implement single-layer transformer with dual prediction heads:
  - Next token prediction (character-level language model)
  - State classification (predict FSM state class)
- Character-level tokenization (one token = one character)
- Configurable architecture (embedding dim, number of attention heads, etc.)
- Instrumentation to extract attention weights and activations (for future analysis)

**4. Training Pipeline**:
- Multi-task training (next-token + state-class losses)
- Standard training loop with checkpointing
- Metrics tracking: accuracy for both tasks, per-class performance
- Save trained models and training logs

**5. Experimental Framework**:
- Define experiments in Python code (not config files)
- Each experiment: regex definition → FSM → data → trained model → metrics
- Organized results directory structure
- Reproducible experiments with seed control

#### Success Criteria

**Primary Success**: Model achieves high accuracy on both tasks
- Next-token prediction accuracy > 95% on test set
- State-class prediction accuracy > 95% on test set
- Model actually learns the pattern, not memorizing

**Starting Simple**:
- Begin with simplest patterns: `a*`, `a*b*`, `a+b+`
- Small alphabets: 2-3 characters
- Short sequences: 10-20 characters
- Prove it works on simple cases before scaling up

**Validation**:
- Model can overfit a tiny dataset (sanity check)
- Model generalizes to held-out test data
- Predictions make sense (not random)
- FSM correctly implements the regex (test against known examples)

#### What This Phase Accomplishes

By the end of Phase 1, we will have:
- A working regex → FSM → training data → trained model pipeline
- Evidence that transformers CAN learn finite state machines (at least simple ones)
- Infrastructure to run experiments systematically
- Baseline performance numbers
- Trained models ready for interpretability analysis in Phase 2

This is **infrastructure building**, not research. We're setting up the testbed.

---

### 1.2 Capacity Experiments

Once we have a working system, we want to understand **how much model capacity is actually required** to learn FSMs of different sizes.

#### Research Question

Given an FSM with:
- |Q| states
- |Σ| alphabet size
- |δ| transitions

What are the minimum transformer parameters (embedding dimension, number of heads) needed for the model to learn it successfully?

#### Why This Matters

**Theoretical Insight**:
- Does capacity scale linearly with FSM size? Logarithmically? Something else?
- Is there a minimum threshold below which learning fails?
- How much excess capacity helps (or hurts)?

**Practical Value**:
- Informs model sizing for future experiments
- Tests whether our capacity heuristics are correct
- Reveals if model is learning efficiently or wastefully

**Interpretability Preparation**:
- Undersized models might learn cleaner, more compressed representations
- Oversized models might spread information across many dimensions
- Understanding capacity helps us know where to look for learned structure

#### Experimental Design

For each regex pattern (e.g., `a*b*` with 3 states):

**1. Undersized Models**:
- Embedding dimension < |Q| (not enough dimensions to represent all states)
- Very few attention heads (1-2)
- **Question**: Does it still learn? How does performance degrade?
- **Hypothesis**: Might fail or learn approximate solutions

**2. Properly-Sized Models**:
- Embedding dimension ≈ |Q| or slightly larger
- Attention heads based on transition structure (4-8)
- **Question**: What's the minimum that achieves high accuracy?
- **Hypothesis**: Should learn successfully with appropriate capacity

**3. Oversized Models**:
- Embedding dimension >> |Q| (many extra dimensions)
- Many attention heads (8-16)
- **Question**: Does extra capacity improve learning or just add noise?
- **Hypothesis**: Might learn faster but representations could be diffuse

#### Metrics

For each capacity configuration, measure:
- Training convergence speed (epochs to reach 95% accuracy)
- Final test accuracy (next-token and state-class)
- Model size (number of parameters)
- Whether model converges at all

**Capacity vs Performance Curves**:
- Plot test accuracy vs embedding dimension
- Identify minimum viable capacity
- Identify saturation point (where more capacity doesn't help)

#### Expected Outcomes

**Best Case**: Clear relationship between FSM complexity and required capacity
- Can derive formula: `embedding_dim = f(|Q|, |Σ|)`
- Minimum threshold is predictable
- Excess capacity has diminishing returns

**Worst Case**: Highly variable, no clear pattern
- Required capacity depends on unpredictable factors
- Need to over-provision to be safe
- Still valuable to know this!

**Most Likely**: Somewhere in between
- Rough heuristics work most of the time
- Some patterns are easier/harder than expected
- Enough data to make educated guesses

#### Value Proposition

Even if we don't get clean theoretical results, capacity experiments tell us:
- Whether the model is learning efficiently
- How much headroom we need for reliable learning
- If our architecture choices make sense
- What to expect as we scale to more complex patterns

This grounds our interpretability work: we'll know if we're looking at a model that's barely fitting the data vs. one with lots of spare capacity.

---

### 1.3 Future Work (Out of Scope for Phase 1)

Phase 1 is about **building and validating**. Phase 2 is about **understanding**.

#### Deferred to Phase 2: Interpretability Analysis

Once we have trained models from Phase 1, we can analyze what they learned:

**Attention Analysis**:
- Do attention patterns correlate with FSM state transitions?
- Can we identify which attention heads track which types of transitions?
- Does attention create a graph structure matching the FSM?

**Activation Analysis**:
- Do MLP activations encode FSM states?
- Can we decode "current state" from hidden representations?
- Are states linearly separable in embedding space?

**Correspondence Finding**:
- Map learned representations to ground-truth FSM structure
- Identify which model components correspond to which FSM components
- Measure how closely the learned structure matches the true structure

**Structure Extraction**:
- Can we extract an explicit FSM from the learned weights?
- How accurate is the extracted FSM compared to ground truth?
- Can we compile the learned model into a symbolic program?

#### Why Wait?

**Need Working Models First**:
- Can't analyze what doesn't exist yet
- Need to confirm models actually learn before asking how they learn
- Avoid premature optimization / analysis

**Infrastructure Comes First**:
- Analysis tools need working FSMs and trained models to operate on
- Visualization and comparison tools depend on complete pipeline
- Get the basics right before diving deep

**Iterative Process**:
- First build it, then study it
- Analysis will reveal what to build next
- Don't guess what analysis we'll need - let the data tell us

#### Explicit Non-Goals for Phase 1

**Do NOT**:
- Build FSM extraction algorithms (we don't know how yet)
- Implement attention visualization dashboards (overkill for now)
- Try to prove theoretical claims about transformers (need data first)
- Optimize for performance or scale (keep it simple)
- Support complex regex features (start minimal)

**Do Instead**:
- Focus on correctness and clarity
- Build with analysis in mind (instrumentation hooks)
- Keep it simple and modifiable
- Document everything
- Make it work, make it right, THEN make it fast/fancy

#### The Path Forward

Phase 1 deliverable: **Working system that proves transformers can learn FSMs**

Phase 2 question: **HOW do transformers learn FSMs? Can we extract the learned structure?**

We're doing research, which means we don't know Phase 2's answers yet. That's fine. Phase 1 gives us the tools to find those answers.

**First make it exist, then make it interpretable.**

## 2. Regex/FSM

### 2.1 Regex Definition

A **regex definition** is a tuple that fully specifies the regex matching problem:

```
RegexDefinition = (alphabet, patterns)

where:
  alphabet: Set[char]  - the set of valid input characters
  patterns: List[(pattern_string, class_name)]  - regex patterns and their classifications
```

#### Components

**Alphabet (Σ)**:
- Explicitly defines all valid input characters
- Keeps vocabulary manageable (e.g., `{'a', 'b', 'c'}` or `{'0', '1', '2', ..., '9'}`)
- Allows use of `.` in regex without requiring full Unicode
- Example: `{'a', 'b'}` for simple binary patterns

**Patterns**:
- List of `(regex_pattern, class_name)` tuples
- Each pattern defines strings that belong to a particular class
- Multiple patterns can define different classes within the same system
- Example:
  - `[("a*b*", "accept")]` - simple accept/reject classification
  - `[("[a-z][a-z0-9]*", "identifier"), ("[0-9]+", "number")]` - token classification

**Implicit Reject State**:
- Any transition not defined by the patterns leads to an implicit reject state
- Reject state loops to itself on all input characters
- Strings that don't match any pattern are classified as "reject"
- Eliminates need to explicitly define rejection patterns

#### Examples

**Simple Binary Pattern**:
```python
RegexDefinition(
    alphabet={'a', 'b'},
    patterns=[("a*b*", "accept")]
)
```
- Strings like "aabb", "b", "aaabbb" → "accept"
- Strings like "ba", "aba", "abab" → "reject"

**Multi-Class Token System**:
```python
RegexDefinition(
    alphabet={'a','b','c','d','e','f','g','h','i','j','k','l','m',
              'n','o','p','q','r','s','t','u','v','w','x','y','z',
              '0','1','2','3','4','5','6','7','8','9'},
    patterns=[
        ("[a-z][a-z0-9]*", "identifier"),
        ("[0-9]+", "number"),
        ("[a-z]+", "keyword")
    ]
)
```
- "foo123" → "identifier"
- "42" → "number"
- "abc" → could match "identifier" or "keyword" (need precedence rules)

**State-Based Classification**:
```python
RegexDefinition(
    alphabet={'a', 'b', 'c'},
    patterns=[
        ("a+", "A"),
        ("b+", "B"),
        ("c+", "C")
    ]
)
```
- Useful for testing if transformer learns to distinguish different states
- "aaa" → "A", "bbb" → "B", "ccc" → "C"
- "abc" → "reject"

#### Special Considerations

**Incomplete vs Reject**:
- An "incomplete" class represents strings that are prefixes of valid patterns
- A "reject" class represents strings that can never lead to acceptance
- We may want to distinguish these in our FSM
- Implementation detail: should incomplete be explicit or derived?

**Ambiguity Handling**:
- If multiple patterns match the same string, need precedence rules
- Options: first-match, longest-match, explicit priority
- For initial implementation: keep patterns non-overlapping

### 2.2 FSM Representation

A **finite state machine** is the formal tuple:

```
FSM = (Q, Σ, δ, q₀, C)

where:
  Q: Set[State]                    - finite set of states
  Σ: Set[char]                     - alphabet (input symbols)
  δ: (State, char) → State         - transition function
  q₀: State                        - start state (q₀ ∈ Q)
  C: State → class_name            - classification function
```

#### Components Explained

**Q - States**:
- Finite set of states in the machine
- Can be represented as integers (0, 1, 2, ...) or objects
- Includes all reachable states from q₀
- May include an explicit reject state or leave it implicit

**Σ - Alphabet**:
- Same as the alphabet from the regex definition
- Defines all valid input symbols
- Transition function δ is only defined for characters in Σ

**δ - Transition Function**:
- Maps (current_state, input_char) → next_state
- Can be represented as:
  - Dictionary: `{(state, char): next_state}`
  - 2D array/matrix indexed by state and character
  - Function that computes next state
- Undefined transitions implicitly go to reject state

**q₀ - Start State**:
- Initial state when processing begins
- Every string trace starts from q₀
- Usually denoted as state 0 or "start"

**C - Classification Function**:
- Maps each state to its class name
- Examples:
  - `C(state_5) = "accept"`
  - `C(state_3) = "incomplete"`
  - `C(state_reject) = "reject"`
- Multiple states can have the same classification
- Critical for training labels

#### Minimal FSM Requirement

We need a **minimal FSM** - the FSM with the fewest states that correctly implements the regex.

**Why minimal?**
- Simpler structure to learn
- Clearer correspondence between states and transformer representations
- More interpretable
- Smaller embedding dimensions needed

**Minimization approach**:
- Construct NFA from regex (Thompson's construction or similar)
- Convert NFA to DFA (subset construction)
- Minimize DFA (Hopcroft's algorithm or similar)
- Preserve state classifications during minimization

#### Reject State Handling

**Option 1: Implicit Reject State**
- Undefined transitions automatically go to reject
- Reject state is not explicitly in Q
- Saves memory and simplifies FSM
- Transition function returns None or special value for reject

**Option 2: Explicit Reject State**
- Add reject state to Q
- All undefined transitions explicitly map to reject state
- Reject state has self-loops for all alphabet characters
- Cleaner for some algorithms

**Decision**: Start with explicit reject state for clarity, optimize later if needed.

#### Example FSM

For regex `a*b*` with alphabet `{a, b}`:

```
Q = {q0, q1, q2, q_reject}
Σ = {a, b}
q₀ = q0

δ = {
  (q0, 'a'): q0,     # stay in q0 while reading a's
  (q0, 'b'): q1,     # transition to q1 on first b
  (q1, 'b'): q1,     # stay in q1 while reading b's
  (q1, 'a'): q_reject,  # can't go back to a's
  # all other transitions go to q_reject
}

C = {
  q0: "accept",       # empty string or just a's
  q1: "accept",       # a's followed by b's
  q_reject: "reject"
}
```

### 2.3 FSM Operations

The FSM needs to support the following operations:

#### 2.3.1 Basic Transition

**`step(state, char) → next_state`**

Single-step state transition.

- **Input**: current state, input character
- **Output**: next state after consuming character
- **Purpose**: Core FSM operation, building block for all other operations
- **Behavior**:
  - Look up δ(state, char)
  - Return next_state
  - Return reject state if transition undefined

**Example**:
```python
state = step(q0, 'a')  # → q0
state = step(q0, 'b')  # → q1
state = step(q1, 'a')  # → q_reject
```

#### 2.3.2 State Classification

**`classify(state) → class_name`**

Get the classification of a given state.

- **Input**: state
- **Output**: class name (string)
- **Purpose**: Determine what class a state belongs to
- **Behavior**: Return C(state)

**Example**:
```python
classify(q0)  # → "accept"
classify(q1)  # → "accept"
classify(q_reject)  # → "reject"
```

#### 2.3.3 String Tracing

**`trace(input_string) → List[State]`**

Trace the sequence of states visited while processing a string.

- **Input**: string of characters
- **Output**: list of states (including start state)
- **Purpose**: Generate training labels - need state at each position
- **Behavior**:
  - Start at q₀
  - For each character, apply step() and record state
  - Return full state sequence
- **Length**: len(states) = len(input_string) + 1 (includes initial state)

**Example**:
```python
trace("ab")  # → [q0, q0, q1]
# Position 0: in q0 (before seeing anything)
# Position 1: in q0 (after 'a')
# Position 2: in q1 (after 'b')
```

**Note on labeling**: For training data at position i:
- Current state = states[i] (before consuming next token)
- Next token = input_string[i] (or EOS if i == len(input_string))
- State class = C(states[i])

#### 2.3.4 String Classification

**`classify_string(input_string) → class_name`**

Classify an entire input string.

- **Input**: string of characters
- **Output**: class name of final state
- **Purpose**: Verify FSM correctness, validate generated data
- **Behavior**:
  - Trace string through FSM
  - Return classification of final state

**Example**:
```python
classify_string("aabb")  # → "accept"
classify_string("ba")    # → "reject"
```

#### 2.3.5 Forward Generation

**`next_possibilities(state) → Set[(char, next_state, class)]`**

Enumerate all valid transitions from a given state.

- **Input**: current state
- **Output**: set of (character, next_state, next_class) tuples
- **Purpose**: Enable forward data generation by random walk
- **Behavior**:
  - For each char in alphabet Σ
  - Compute next_state = δ(state, char)
  - Compute next_class = C(next_state)
  - Return tuple (char, next_state, next_class)

**Example**:
```python
next_possibilities(q0)  # → {('a', q0, 'accept'), ('b', q1, 'accept')}
next_possibilities(q1)  # → {('a', q_reject, 'reject'), ('b', q1, 'accept')}
```

**Usage for generation**:
- Sample randomly from possibilities
- Can filter by desired next_class (e.g., only non-reject transitions)
- Build strings incrementally

#### 2.3.6 Backward Generation (Optional)

**`prev_possibilities(state) → Set[(char, prev_state, prev_class)]`**

Enumerate all states that can transition TO the given state.

- **Input**: current state
- **Output**: set of (character, prev_state, prev_class) tuples
- **Purpose**: Enable backward data generation from accept states
- **Behavior**:
  - For each state s in Q
  - For each char in Σ
  - If δ(s, char) == current_state, include (char, s, C(s))

**Example**:
```python
prev_possibilities(q1)  # → {('b', q0, 'accept'), ('b', q1, 'accept')}
# q1 can be reached from q0 via 'b' or from q1 via 'b'
```

**Usage**:
- Start from accept state
- Walk backward to start state
- Guarantees generating accepting strings
- More complex to implement (requires reverse index)
- **Decision**: Optional for initial implementation, add if needed

#### 2.3.7 Reset

**`reset() → State`**

Return to the start state.

- **Input**: none
- **Output**: start state q₀
- **Purpose**: Convenient for generation, reset between strings
- **Behavior**: Return q₀

#### 2.3.8 Additional Utilities

**`is_accept_state(state) → bool`**
- Check if state is classified as "accept"
- Useful for termination conditions in generation

**`is_reject_state(state) → bool`**
- Check if state is classified as "reject"
- Useful for filtering during generation

**`get_states_by_class(class_name) → Set[State]`**
- Return all states with given classification
- Useful for targeted generation

### 2.4 FSM Properties

The FSM should expose metadata useful for model configuration and debugging:

#### 2.4.1 Structural Properties

**State Count**: `|Q|`
- Number of states in the FSM
- Informs embedding dimension (need to represent this many states)
- Example: `a*b*` has 3 states (q0, q1, q_reject)

**Transition Count**: Number of defined transitions
- Size of transition function δ
- Measure of FSM complexity
- Dense vs sparse transition structure

**Alphabet Size**: `|Σ|`
- Number of valid input characters
- Determines vocabulary size for transformer
- Informs embedding dimension

#### 2.4.2 Classification Properties

**Class Distribution**: `{class_name: count}`
- How many states have each classification
- Check for imbalanced classifications
- Example: `{"accept": 2, "reject": 1}`

**Classes**: `Set[class_name]`
- All distinct class names
- Determines output dimension for classification head

#### 2.4.3 Reachability

**Reachable States**: States reachable from q₀
- Some states might be unreachable (artifact of construction)
- Only reachable states matter for learning
- Should be all states in a properly minimized FSM

**States by Class**: Map of class → states
- Which states have which classifications
- Useful for targeted data generation

#### 2.4.4 Usage for Model Sizing

These properties inform model hyperparameters:

- **Embedding dimension**: Should be ≥ |Q| to represent all states
  - Heuristic: `embedding_dim = max(|Q|, |Σ|) * multiplier`

- **Vocabulary size**: `|Σ| + special_tokens`
  - Need tokens for each character plus EOS, possibly PAD

- **Classification output**: `|Classes|`
  - Output dimension for state classification head

- **Number of attention heads**: Related to transition structure
  - Could be based on branching factor or alphabet size
  - Heuristic: 4-8 heads for small FSMs

### 2.5 FSM Serialization

We need to save and load FSMs for reproducibility and inspection.

#### 2.5.1 Serialization Requirements

**What to save**:
- All FSM components: Q, Σ, δ, q₀, C
- Metadata: state count, class distribution, etc.
- Original regex definition (for reference)
- Creation timestamp, version info

**Format**: JSON (human-readable, standard, easy to inspect)

**Structure**:
```json
{
  "regex_definition": {
    "alphabet": ["a", "b"],
    "patterns": [
      {"pattern": "a*b*", "class": "accept"}
    ]
  },
  "fsm": {
    "states": [0, 1, 2],
    "alphabet": ["a", "b"],
    "start_state": 0,
    "transitions": {
      "0,a": 0,
      "0,b": 1,
      "1,b": 1,
      "1,a": 2
    },
    "classifications": {
      "0": "accept",
      "1": "accept",
      "2": "reject"
    }
  },
  "metadata": {
    "state_count": 3,
    "transition_count": 4,
    "classes": ["accept", "reject"],
    "created": "2025-11-07T15:30:00Z"
  }
}
```

#### 2.5.2 Operations

**`save_fsm(fsm, filepath)`**
- Serialize FSM to JSON
- Save to specified file path
- Include all necessary information for reconstruction

**`load_fsm(filepath) → FSM`**
- Load FSM from JSON file
- Reconstruct FSM object
- Validate loaded data

**`fsm_to_dict(fsm) → dict`**
- Convert FSM to dictionary (for JSON serialization)

**`dict_to_fsm(data) → FSM`**
- Reconstruct FSM from dictionary

### 2.6 FSM Visualization

Visual representations of FSMs for debugging and analysis.

#### 2.6.1 Graph Representation

**Format**: Directed graph where:
- Nodes = states
- Edges = transitions labeled with characters
- Node colors = state classifications
- Special marking for start state (arrow or double circle)

**Library Options**:
- **NetworkX**: Python graph library, easy to use
- **Graphviz**: Professional graph visualization via DOT format
- **Matplotlib**: Direct drawing (more control, more code)

**Recommendation**: NetworkX + Graphviz for best results

#### 2.6.2 Visualization Features

**Color Coding**:
- Different color for each class
- Example: green for "accept", red for "reject", yellow for "incomplete"
- Helps visually distinguish state types

**State Labels**:
- Show state ID/name
- Show state class
- Example: "q0 (accept)"

**Transition Labels**:
- Show character(s) that trigger transition
- Multiple characters on same edge if they go to same next state
- Example: "a,b,c" if all three lead to same state

**Start State Marking**:
- Arrow pointing to start state from outside
- Or double circle
- Makes entry point clear

**Layout**:
- Automatic layout algorithm (spring, hierarchical, etc.)
- Readable even for moderately complex FSMs
- May need manual adjustment for publication-quality

#### 2.6.3 Operations

**`visualize_fsm(fsm, output_path=None, format='png')`**
- Generate visual representation of FSM
- Save to file or display interactively
- Supported formats: PNG, PDF, SVG, DOT

**`fsm_to_dot(fsm) → str`**
- Convert FSM to DOT format (Graphviz)
- Can be rendered separately or saved for manual editing

**Example visualization needs**:
- Debug FSM construction
- Verify minimization worked correctly
- Include in experiment reports
- Compare ground truth FSM to learned structure (future work)

## 3. Transformer Model

### 3.1 Architecture Overview

Single-layer transformer with character-level tokenization and multiple prediction heads.

**High-level structure**:
```
Input Tokens (+ optional Goal)
    ↓
Token Embedding (factorized with sparsity)
    ↓
Position Embedding (learned)
    ↓
Transformer Block:
    - Multi-head Self-Attention (causal mask)
    - MLP/Feed-forward
    - Residual connections
    - LayerNorm
    ↓
Three Output Heads:
    - Next Token (vocab_size logits)
    - Current Class (num_classes logits)
    - FSM State (|Q| logits)
```

**Why single layer?**
- Maximum interpretability
- Clear attribution of learned structure
- Sufficient for finite state machines (hypothesis to test)
- Can scale to multiple layers later if needed

---

### 3.2 Input Representation

#### 3.2.1 Token Vocabulary

**Vocabulary = Alphabet + Special Tokens**

- **Alphabet characters**: All characters from FSM alphabet Σ
- **Special tokens**:
  - `<EOS>`: End of sequence (always included)
  - `<PAD>`: Padding token (for batching variable-length sequences)
  - `<GOAL>`: Goal class token (optional, for goal-conditioned experiments)

**Vocabulary size**: `vocab_size = |Σ| + num_special_tokens`

**Example**: For alphabet `{a, b}`:
- Vocab = `[a, b, <EOS>, <PAD>]` → vocab_size = 4
- With goal conditioning: vocab_size = 4 + num_classes (one goal token per class)

#### 3.2.2 Token Embedding (Factorized with Sparsity)

**Purpose**: Learn embeddings that encourage feature superposition and interpretability.

**Architecture**:
```
Factorized embedding: E = U @ V

where:
  U ∈ ℝ^(vocab_size × k)  - token-to-feature matrix
  V ∈ ℝ^(k × embedding_dim) - feature-to-embedding matrix
  k - number of intermediate features (k << vocab_size typically)
```

**Token embedding process**:
```python
# For token t:
e_t = U[t, :] @ V  # shape: (embedding_dim,)

# U[t, :] is a k-dimensional feature vector
# Each token is represented as a sparse combination of k features
```

**Sparsity regularization**:
- Apply L1 penalty on U during training
- Encourages each token to use only a few features
- Creates interpretable feature sharing across tokens
- Example: tokens that transition similarly might share features

**Why factorized?**
- **Standard embedding** (lookup table): Each token gets independent vector
  - No explicit structure sharing
  - Harder to interpret which tokens are "similar"

- **Factorized embedding**: Tokens share a common feature space
  - Token similarity emerges from shared features
  - Sparsity makes it interpretable (which tokens use which features)
  - Smaller parameter count when k < embedding_dim

**Hyperparameters**:
- `k`: Number of intermediate features (4-16 typical)
- `l1_weight`: Strength of L1 penalty on U
- Both configurable for experiments

**Ablation variants** (for experiments):
- **Variant A (default)**: Factorized U @ V with L1(U)
- **Variant B (baseline)**: Standard lookup table (no factorization)
- **Variant C (structured)**: Feature-coded embedding with hand-designed features

#### 3.2.3 Position Embedding

**Learned position embeddings** (default):
```python
pos_emb ∈ ℝ^(max_seq_length × embedding_dim)

# For position i:
p_i = pos_emb[i, :]  # shape: (embedding_dim,)
```

**Combined input**:
```python
# At position i with token t:
input_i = token_emb[t] + pos_emb[i]
```

**Why learned?**
- Simpler than sinusoidal for short sequences
- Let model learn position representation
- Easy to inspect what positions encode

**Alternative (optional)**: Sinusoidal / RoPE
- Can add rotary position embeddings (RoPE) as alternative
- Test if position encoding method matters
- Defer unless needed

**Sequence length**:
- `max_seq_length`: Maximum input sequence length
- Configured based on FSM and data generation
- Typical: 10-50 for simple patterns

#### 3.2.4 Goal Conditioning (Optional)

**Purpose**: Enable goal-directed generation ("generate string reaching class X")

**Implementation**:
```python
# Option 1: Goal token at start (recommended)
sequence = [<GOAL=accept>, 'a', 'a', 'b', 'b', <EOS>]

# Goal token gets special embedding
goal_emb = goal_token_embeddings[goal_class]

# Option 2: FiLM conditioning (alternative)
# Compute scale/shift from goal, apply to residual stream
gamma, beta = goal_network(goal_embedding)
h_conditioned = gamma * h + beta
```

**Default**: Goal token at position 0 (simplest)

**When to use**:
- Test if model can do goal-directed generation
- Probe whether model learns class-conditional distributions
- Compare goal-conditioned vs unconditional learning

**Not required for basic experiments** - can start without it and add later.

---

### 3.3 Transformer Block

Single transformer block with standard components.

#### 3.3.1 Multi-Head Self-Attention

**Purpose**: Learn relationships between positions in sequence

**Architecture**:
```python
# For input sequence H ∈ ℝ^(seq_len × embedding_dim)

# Per head h:
Q_h = H @ W_Q_h  # Query
K_h = H @ W_K_h  # Key
V_h = H @ W_V_h  # Value

# Attention scores with causal mask
scores_h = (Q_h @ K_h.T) / sqrt(head_dim)
scores_h = scores_h.masked_fill(causal_mask, -inf)
attn_h = softmax(scores_h, dim=-1)

# Attention output
out_h = attn_h @ V_h

# Concatenate all heads and project
attn_out = concat([out_1, out_2, ..., out_H]) @ W_O
```

**Causal masking** (always applied):
- Position i can only attend to positions ≤ i
- Prevents information leakage from future tokens
- Necessary for next-token prediction
- Keeps all training objectives aligned

**Why causal everywhere?**
- Next-token prediction requires it
- Even for classification, prevents "cheating" by looking ahead
- Makes experiments comparable across objectives
- Models reality (when generating, we don't know future)

**Number of heads**: `num_heads`
- Configurable based on FSM complexity
- Must divide `embedding_dim` evenly
- Typical: 4-8 heads for small FSMs
- Each head can specialize on different transition types (hypothesis)

**Head dimension**: `head_dim = embedding_dim / num_heads`

#### 3.3.2 MLP / Feed-Forward Network

**Purpose**: Nonlinear transformation after attention

**Architecture**:
```python
# Standard transformer MLP
mlp_out = MLP(layer_norm(attn_out))

# MLP structure:
def MLP(x):
    h = Linear(embedding_dim → mlp_dim)(x)
    h = activation(h)  # ReLU or GELU
    h = Dropout(h)
    h = Linear(mlp_dim → embedding_dim)(h)
    return h
```

**MLP dimension**: `mlp_dim`
- Typical: `mlp_dim = 4 * embedding_dim` (standard transformer)
- Configurable for capacity experiments
- Can test smaller/larger ratios

**Activation**: ReLU or GELU (standard choices)

#### 3.3.3 Residual Connections and LayerNorm

**Standard transformer structure with pre-norm**:
```python
# Attention block
h1 = x + Attention(LayerNorm(x))

# MLP block
h2 = h1 + MLP(LayerNorm(h1))

# Final output
output = h2
```

**Pre-norm vs post-norm**:
- Use **pre-norm** (current standard)
- More stable training
- LayerNorm before each sub-block

**Why this matters for interpretability**:
- Residual stream accumulates information
- Each layer adds to representation
- Can decompose: `output = input + attn_contribution + mlp_contribution`
- Useful for analysis in Phase 2

---

### 3.4 Output Heads

Three separate prediction heads operating on the final hidden states.

#### 3.4.1 Next Token Prediction Head

**Purpose**: Predict the next character in the sequence

**Architecture**:
```python
# For each position i, predict next token
logits_next = Linear(embedding_dim → vocab_size)(h[i])

# Prediction: argmax(logits_next)
# Loss: CrossEntropy(logits_next, true_next_token[i])
```

**Output dimension**: `vocab_size` (alphabet + special tokens)

**Target**: At position i, target is token at position i+1 (or `<EOS>` if at end)

**Weight tying** (optional but recommended):
```python
# Tie output head weights to input embedding
# Makes learning more efficient
next_token_head.weight = token_embedding.weight.T
```

#### 3.4.2 Current Class Head

**Purpose**: Predict the FSM state class at current position (before consuming next token)

**Architecture**:
```python
# For each position i, predict current state class
logits_class = Linear(embedding_dim → num_classes)(h[i])

# Prediction: argmax(logits_class)
# Loss: CrossEntropy(logits_class, true_class[i])
```

**Output dimension**: `num_classes` (number of distinct state classifications)

**Target**: At position i, target is `C(state[i])` where `state[i]` is FSM state before consuming token i

**Example**: For `"ab"` with FSM states `[q0, q0, q1]`:
- Position 0: predict class of q0 (before seeing 'a')
- Position 1: predict class of q0 (before seeing 'b')
- Position 2: predict class of q1 (before seeing EOS)

#### 3.4.3 FSM State Head

**Purpose**: Predict the exact FSM state at current position (more fine-grained than class)

**Architecture**:
```python
# For each position i, predict current FSM state
logits_state = Linear(embedding_dim → num_states)(h[i])

# Prediction: argmax(logits_state)
# Loss: CrossEntropy(logits_state, true_state[i])
```

**Output dimension**: `|Q|` (number of FSM states)

**Target**: At position i, target is state index (integer 0 to |Q|-1)

**Relationship to class head**:
- State prediction is more specific than class prediction
- Multiple states can have same class
- State → Class is a many-to-one mapping
- If model learns states, it implicitly knows classes

**Why both?**
- State head: Tests if model learns full FSM structure
- Class head: Tests if model learns high-level abstractions
- Useful to compare: does model learn states or just classes?
- Class may be easier to learn (fewer categories)

---

### 3.5 Training Modes

**Multi-task learning** with configurable loss weights.

#### 3.5.1 Loss Function

**Combined loss**:
```python
loss_total = λ_next * loss_next_token
           + λ_class * loss_class
           + λ_state * loss_state

where:
  loss_next_token = CrossEntropy(logits_next, targets_next)
  loss_class = CrossEntropy(logits_class, targets_class)
  loss_state = CrossEntropy(logits_state, targets_state)

  λ_next, λ_class, λ_state ∈ {0, 1}  # on/off switches
```

**Training modes** (via λ settings):
1. **Next-token only**: (1, 0, 0)
2. **Class only**: (0, 1, 0)
3. **State only**: (0, 0, 1)
4. **Next-token + Class**: (1, 1, 0)
5. **Next-token + State**: (1, 0, 1)
6. **All objectives**: (1, 1, 1)

**Why configurable?**
- Test which objectives help learning
- Ablate to understand what model needs
- State prediction might be easier than next-token (or vice versa)
- Multi-task might improve or hurt (empirical question)

**Default for initial experiments**: Mode 4 (next-token + class)
- Both objectives are meaningful
- Class is interpretable
- Next-token ensures model learns sequence structure

#### 3.5.2 Per-Position Masking

**Padding mask**:
- Don't compute loss on padded positions
- Mask out padding tokens in loss calculation

**Example**:
```python
# Sequence: ['a', 'b', <EOS>, <PAD>, <PAD>]
# Valid positions: [0, 1, 2]
# Compute loss only for positions 0, 1, 2
```

---

### 3.6 Model Sizing

**Hyperparameters informed by FSM properties**.

#### 3.6.1 Embedding Dimension

**Goal**: Sufficient capacity to represent FSM states

**Heuristic**:
```python
embedding_dim = max(|Q|, |Σ|) * multiplier

where:
  |Q| = number of FSM states
  |Σ| = alphabet size
  multiplier = 2-4 (configurable)
```

**Rationale**:
- Need at least |Q| dimensions to represent all states (in theory)
- Alphabet size matters if learning character relationships
- Multiplier adds headroom for robustness

**Capacity experiments**:
- **Undersized**: embedding_dim < |Q|
- **Properly-sized**: embedding_dim ≈ |Q| to 2*|Q|
- **Oversized**: embedding_dim >> |Q|

**Minimum**: 32 or 64 (practical minimum for stability)

#### 3.6.2 Number of Attention Heads

**Heuristic**:
```python
num_heads = min(max(4, |Σ|), 8)

# Or based on branching factor:
num_heads = average_transitions_per_state
```

**Constraints**:
- Must divide `embedding_dim` evenly
- Minimum: 1 (for undersized experiments)
- Typical: 4-8

**Rationale**:
- Each head could specialize on a transition type
- More heads allow more specialization
- Too many heads might over-parameterize small FSMs

#### 3.6.3 MLP Dimension

**Standard**: `mlp_dim = 4 * embedding_dim`

**Adjustable for capacity experiments**:
- Can try 2x, 4x, 8x multipliers
- Affects parameter count significantly

#### 3.6.4 Sequence Length

**Based on data generation**:
```python
max_seq_length = expected_string_length + buffer

# Typical: 16-64 for simple patterns
```

#### 3.6.5 Example Configurations

**For FSM `a*b*` with |Q|=3, |Σ|=2**:

**Config 1 (Undersized)**:
- embedding_dim: 2 (< |Q|)
- num_heads: 1
- mlp_dim: 8
- Total params: ~50-100

**Config 2 (Properly-sized)**:
- embedding_dim: 8 (≈ 3 * 2.5)
- num_heads: 4
- mlp_dim: 32
- Total params: ~300-500

**Config 3 (Oversized)**:
- embedding_dim: 64 (>> |Q|)
- num_heads: 8
- mlp_dim: 256
- Total params: ~10k+

---

### 3.7 Instrumentation

**Purpose**: Extract internal representations for future analysis (Phase 2)

**Required hooks**:

#### 3.7.1 Attention Weights

```python
# For each head h, save attention matrix:
attention_weights[h] ∈ ℝ^(seq_len × seq_len)

# Where attention_weights[h][i, j] = attention from position i to j
```

**What to save**:
- All heads, all layers
- For sample inputs (don't save everything)
- Store in experiment results

**Use cases**:
- Visualize attention patterns
- Compare to FSM transition matrix
- Identify which heads track which transitions

#### 3.7.2 MLP Activations

```python
# Save MLP intermediate activations:
mlp_pre ∈ ℝ^(seq_len × mlp_dim)   # before activation function
mlp_post ∈ ℝ^(seq_len × embedding_dim)  # after second linear layer
```

**Use cases**:
- Train linear probes to decode FSM state
- Visualize activation space
- Check if states are linearly separable

#### 3.7.3 Residual Stream

```python
# Save hidden states at each point:
h_input ∈ ℝ^(seq_len × embedding_dim)      # after embedding
h_post_attn ∈ ℝ^(seq_len × embedding_dim)  # after attention block
h_post_mlp ∈ ℝ^(seq_len × embedding_dim)   # after MLP block (final)
```

**Use cases**:
- Decompose information flow
- Measure layer contributions
- Verify residual stream hypothesis

#### 3.7.4 Implementation

**Using PyTorch hooks**:
```python
# Register forward hooks to capture activations
def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model.attention.register_forward_hook(save_activation('attn'))
model.mlp.register_forward_hook(save_activation('mlp'))
```

**Storage**:
- Don't save activations for every training step (too expensive)
- Save for validation set or specific test examples
- Store in experiment results directory

---

### 3.8 Model Configuration

**All hyperparameters in a config object**:

```python
ModelConfig:
    # Vocabulary
    vocab_size: int

    # Architecture
    embedding_dim: int
    num_heads: int
    mlp_dim: int
    max_seq_length: int

    # Factorized embedding
    use_factorized_embedding: bool
    factorization_rank: int  # k
    embedding_l1_weight: float

    # Output heads
    num_classes: int
    num_states: int

    # Training objectives (loss weights)
    lambda_next_token: float
    lambda_class: float
    lambda_state: float

    # Optional features
    use_goal_conditioning: bool

    # Regularization
    dropout: float

    # Activation function
    activation: str  # 'relu' or 'gelu'
```

**Auto-sizing helper**:
```python
def auto_size_model(fsm, multiplier=2, num_heads=4):
    """Generate model config from FSM properties"""
    embedding_dim = max(fsm.num_states, fsm.alphabet_size) * multiplier

    # Adjust to make divisible by num_heads
    embedding_dim = (embedding_dim // num_heads) * num_heads

    return ModelConfig(
        vocab_size=fsm.alphabet_size + num_special_tokens,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        mlp_dim=4 * embedding_dim,
        num_classes=len(fsm.classes),
        num_states=fsm.num_states,
        ...
    )
```

## 4. Data Generation

### 4.1 Generation Strategy

The data generator operates directly on the compiled FSM for each regex.
The FSM defines all legal transitions δ and terminal classes (Accept, Incomplete, Reject).
We treat the FSM as a search space and produce sequences by guided exploration rather than naïve random walks.
	•	Targeted sampling: Each sample begins with a target (class, length) drawn from configurable distributions.
This provides direct control of length and class balance and eliminates geometric-length bias.
	•	Backward and forward search:
	•	Use backward walks from terminal states to q₀ to hit exact length/class targets efficiently.
	•	Fall back to best-first/beam search (A*-style) when backward generation is infeasible or when additional coverage/diversity is desired.
	•	Both modes support coverage weighting so rarely used (q, a) edges are favored.
	•	Coverage control: Maintain per-state and per-edge hit counts; exploration bonuses bias search toward under-represented regions of the FSM.
	•	Negative construction: Reject examples are generated through structured perturbations—illegal steps, overruns, premature stops, and wrong-branch alternations—ensuring diverse failure modes.
	•	Extendable frontier: Every partial sequence (search node) can be re-enqueued to extend later, enabling curriculum growth or active rebalancing.

### 4.2 Data Characteristics

The resulting dataset contains sequences covering all reachable behaviors of the FSM:
	•	Accept strings: Complete paths ending in A (accept states).
	•	Incomplete strings: Prefixes ending in non-accept states from which A remains reachable.
	•	Reject strings: Paths entering the reject sink or containing illegal transitions.
	•	Balanced distribution: The generator enforces (or reports) class and length balance and tracks coverage statistics.
	•	Configurable difficulty: Parameters (length range, class ratios, coverage targets) can be tuned to shape task complexity and support ablation or curriculum studies.

### 4.3 Labeling

All labels are regenerated on-the-fly from the FSM at load time; nothing except the token sequence is stored.
	•	Per-position labels:
	•	Next token: the following character or EOS.
	•	State class: the FSM class of the current state before reading the next token.
	•	Label generation: Trace each string through δ(q,a) to produce the state sequence in O(L).
	•	Alignment: State class labels align with transformer input positions, supporting both next-token and state-prediction objectives.

### 4.4 Dataset Splits

Datasets are divided deterministically by hashing the raw string, guaranteeing no leakage between splits.
	•	Training set – majority of samples.
	•	Validation set – used for hyperparameter tuning and convergence checks.
	•	Test set – held-out evaluation only.
	•	Configurable sizes: Split proportions and optional OOD partitions (held-out lengths, edges, regex families) are set in configuration.

### 4.5 Reproducibility and Telemetry
	•	Deterministic generation: All randomness is seeded; given the same FSM and seed, data regenerate bit-for-bit.
	•	Audit metadata: Each shard records the RNG seed, generation mode, and configuration hash.
	•	Telemetry reports: For each shard the generator logs realized class/length histograms, state/edge coverage, negative subtype ratios, and generation failures—providing quantitative guarantees of dataset quality.

## 5. Symbolic Transformer Analogue (Idealized Reference Model)

### 5.1 Purpose

Before training any learned transformer, we construct a **symbolic analogue** — an idealized reference model that formalizes how attention operations could represent FSM transitions if the transformer learned the FSM perfectly.

**This is NOT an learned transformer block.** It's an analytic surrogate that describes the mathematical structure of an idealized attention computation. It is implementable directly from the FSM definition.

This reference model acts as:
- **A conceptual hypothesis**: What kind of computation would a transformer perform if its hidden states perfectly encoded FSM states?
- **A diagnostic tool**: Ground-truth attention patterns for comparing to learned attention matrices
- **An interpretability guide**: Expected structure that trained models might converge toward

Where the FSM defines discrete transitions δ(q, a) → q', the symbolic analogue describes what Q, K, V structures would reproduce those transitions through attention, assuming perfect state representation.

---

### 5.2 Conceptual Overview

**Traditional FSM evaluation is sequential**:
```
q_t = δ(q_{t-1}, x_t)
```

**The transformer performs parallel, differentiable message-passing** across all token positions using self-attention:
```
H' = softmax((QK^T / √d_k) + M_causal) V
```

The symbolic analogue asks: "If we could design Q, K, V matrices directly from δ, what would they look like?"

It describes the mathematical structure of attention operations that would reproduce δ's semantics — effectively **"compiling" the FSM into transformer form**.

**Important**: This is not a faithful reconstruction of transformer data flow. In a real transformer:
- Q_i = W_Q h_i depends only on the current position's hidden state
- K_j = W_K h_j depends only on another position's hidden state
- The hidden states h themselves emerge from learned embeddings and attention

The symbolic analogue abstracts away this complexity by assuming we already know the FSM state at each position. It describes the **ideal fixed point** that training could converge to, not the mechanism of convergence itself.

---

### 5.3 Construction

#### Inputs
- Finite-state machine M = (Q, Σ, δ, q₀, C)
- Input token sequence x₁, ..., x_T

#### Derived components
1. **States** → one-hot basis e_q ∈ ℝ^|Q|
2. **Tokens** → one-hot basis e_a ∈ ℝ^|Σ|
3. **Transition function δ** → defines valid triples (q, a, q')

---

### 5.4 Symbolic Q, K, V Definitions

We define query, key, and value tables so that the attention mechanism exactly enforces δ:

**Keys: represent current FSM state**
```
K[q] = e_q
```

**Queries: represent token-conditioned state matches**
```
Q[a,q] = {
  1  if δ(q,a) defined
  0  otherwise
}
```
⇒ A query for token a "attends" only to keys for predecessor states that can transition under a.

**Values: encode next-state embeddings**
```
V[a,q] = e_{δ(q,a)}
```

Each token x_i = a retrieves a weighted combination of values from previous positions j < i whose states q_j satisfy δ.

**Note on abstraction**: The formulation Q[a,q] intentionally bakes in both the current token (a) and the previous state (q) into the query structure. In a real transformer, Q_i would depend only on h_i (the current position's representation), not directly on both token identity and prior state. This is an **intentional abstraction** that collapses the dependencies to describe the ideal matching rule: "position i (with token a) should attend to position j if δ(q_j, a) is defined." A faithful mechanistic account would require explaining how h_i encodes sufficient information about both x_i and the query pattern, which is what the learned transformer must discover through training.

---

### 5.5 Symbolic Attention Operation

For each position i:
```
A_{ij} = {
  1  if δ(q_j, x_i) is defined and j < i
  0  otherwise
}
```

Then the symbolic "attention update" is:
```
h_i = Σ_{j<i} A_{ij} V[x_i, q_j]
```

Since δ is deterministic, each row of A is effectively one-hot:
```
h_i = e_{δ(q_{i-1}, x_i)}
```

**This describes a fixed point, not a forward pass**: The construction assumes we already know q_j at each position j to build the attention matrix A. But in a real transformer's causal forward pass, h_i depends on earlier hidden states through attention — there's a circularity here.

What this formulation really describes is: **"If the transformer's internal representations perfectly encoded FSM states, then the attention patterns that preserve δ would look like this."**

It's describing the **ideal attractor** that training could converge to, not the step-by-step mechanism of how the model reaches that attractor.

**Algebraic form vs. computational parallelism**: When we say "parallel" or "one-pass", we mean the attention matrix A can be written in matrix form (mathematical simultaneity), not that it can be computed in a single forward pass without knowing the states. The symbolic analogue expresses FSM transitions as linear algebraic operations, which is the same mathematical framework transformers use — even though the learned transformer must discover this structure implicitly through gradient descent.

---

### 5.6 Multi-Head Analogue (Optional)

To emulate a multi-head transformer, δ can be partitioned into edge subsets:
```
E_h ⊆ {(q,a,q')}
```

Each head h defines its own Q_h, K_h, V_h from E_h.

**Heads may specialize for**:
- Self-loops / stay transitions
- Class-changing transitions
- Long-range or reset edges

The outputs from all heads are concatenated and linearly projected, mirroring the learned transformer block.

---

### 5.7 Evaluation Characteristics

- **Matrix form (not computational parallelism)**: Expresses FSM transitions as attention matrices A, which can be written in algebraic form. "One-pass" means mathematical simultaneity (matrix operations), not that it can be computed in one forward pass without prior state knowledge.
- **Exact execution**: Reproduces the FSM's δ transitions without error (assuming perfect state representations)
- **Interpretability**: Every attention edge in A corresponds to a valid FSM transition δ(q,a) → q'
- **Diagnostic reference**: Serves as the ground-truth pattern that the trained transformer is hypothesized to approximate. Used for comparison, not as an implementable model.

---

### 5.8 Implementation Sketch

```python
def build_symbolic_qkv(fsm):
    """Construct symbolic Q, K, V matrices from FSM"""
    Q = np.zeros((len(fsm.alphabet), len(fsm.states)))
    V = np.zeros((len(fsm.alphabet), len(fsm.states), len(fsm.states)))

    for a_idx, a in enumerate(fsm.alphabet):
        for q_idx, q in enumerate(fsm.states):
            if (q, a) in fsm.delta:
                q_next = fsm.delta[(q, a)]
                Q[a_idx, q_idx] = 1
                V[a_idx, q_idx, q_next] = 1

    K = np.eye(len(fsm.states))
    return Q, K, V

def symbolic_attention_step(Q, K, V, seq_tokens, init_state):
    """Execute FSM using symbolic attention (sequential for clarity)"""
    states = [init_state]
    for a in seq_tokens:
        q_prev = states[-1]
        q_next = np.argmax(V[a, q_prev])
        states.append(q_next)
    return states
```

**Note**: The implementation above is **sequential** (iterating token by token) for clarity and simplicity. The **theory** is parallel/matrix-based: the full attention matrix A for a sequence can be constructed in matrix form. However, actually computing A requires knowing all states, which brings us back to the circularity mentioned in Section 5.5.

In practice, for the diagnostic reference, we can:
- Use the sequential implementation to trace FSM execution and generate ground-truth state sequences
- Construct the ideal attention matrix A^symbolic post-hoc from those traces
- Compare A^symbolic to learned attention matrices A^learned

This symbolic attention layer provides the expected evaluation pattern that the real transformer should learn to approximate.

---

### 5.9 What's Abstracted Away

The symbolic analogue deliberately omits several components present in a real transformer:

**Token and Position Embeddings**:
- Real transformers map discrete tokens to continuous embeddings: e(x_i) ∈ ℝ^d
- Position information is added: h_i^(0) = e(x_i) + p_i
- The symbolic analogue assumes identity mappings: tokens and states are already one-hot vectors
- **Implication**: In the learned model, embeddings supply the continuous basis for Q/K/V projections that implicitly realize δ

**Hidden State Evolution**:
- Real transformers compute h_i through attention over previous hidden states
- Each h_i is a mixture of information from earlier positions
- The symbolic analogue assumes h_i = e_q (pure state encoding) is achieved
- **Implication**: The learned model must discover how to encode FSM states in its continuous hidden representations

**Residual Connections and Layer Normalization**:
- Real transformers accumulate information via residuals: h^(l+1) = h^(l) + Attn(h^(l))
- LayerNorm stabilizes training
- The symbolic analogue describes only the attention operation itself
- **Implication**: Analysis must account for how residuals and norms affect the learned FSM representation

**Why abstract these away?**
The symbolic analogue focuses on the core question: "Can attention operations encode FSM transitions?"

The answer is: "Yes, if hidden states represent FSM states and Q/K/V are structured appropriately."

The embeddings, residuals, and norms are important for **how the learned model achieves this**, but not for **whether it's possible** — which is what the symbolic analogue demonstrates.

---

### 5.10 Role in Experiments

The symbolic analogue is **not a trained model** — it's a diagnostic reference.

It allows us to:
- **Visualize** what an ideal attention pattern would look like
- **Compare** learned attention matrices A_learned to the ideal A^symbolic
- **Quantify convergence**: how closely does the learned transformer's attention or residual dynamics align with the symbolic evaluator's structure?

**Usage in Phase 2 (Interpretability)**:
- Generate ground-truth attention patterns for comparison
- Compute correlation metrics between learned and symbolic attention
- Identify which heads in the learned model correspond to which transition types
- Test if the learned model is actually approximating the FSM or solving the task differently

**Summary**: The symbolic analogue provides a theoretical target — the mathematical structure that a perfect FSM-implementing transformer would have. The learned transformer may converge toward this structure, approximate it loosely, or solve the task through an entirely different mechanism. Phase 2 analysis will reveal which scenario occurs.

## 6. Experimental Framework

### 6.1 Experiment Definition
Experiments are defined entirely in Python code to maximize clarity, composability, and reproducibility.

- **Direct instantiation:** Each experiment is a small Python class or script that directly assigns parameters — no JSON/YAML translation layer.  
- **Executable specification:** Running the experiment object produces the complete pipeline (FSM → data → model → evaluation).  
- **Programmatic generation:** New experiments (e.g., sweeps, ablations, curriculum sequences) can be created dynamically through Python loops or inheritance.  
- **Explicit provenance:** All parameters are defined in code and stored alongside results for full traceability.

### 6.2 Experiment Components
Each experiment specifies five core components:

1. **Regex definition**  
   - The formal expression or pattern family to compile into an FSM.  
   - May include composed or parameterized regexes for grouped studies.

2. **Data generation parameters**  
   - Controls dataset balance, sequence length distribution, and coverage targets.  
   - Connects directly to the FSM for consistent labeling and reproducibility.

3. **Model architecture parameters**  
   - Transformer depth, width, embedding dimensions, positional encoding type, and other structural settings.  
   - Enables controlled capacity studies and scaling experiments.

4. **Training hyperparameters**  
   - Optimizer type, learning rate, batch size, dropout, gradient clipping, and training schedule.  
   - Tuned for stability and comparability across datasets.

5. **Random seed**  
   - Shared across Python, NumPy, PyTorch, and the data generator to guarantee deterministic runs.

### 6.3 Experiment Execution
Each experiment follows a consistent sequence:

1. **Compile regex → FSM**  
   - Convert the regex definition into an FSM with all states, transitions, and terminal classes.

2. **Generate data**  
   - Produce training, validation, and test splits using the configured data generator.  
   - Verify balance and coverage via telemetry reports.

3. **Build and train model**  
   - Instantiate the transformer model with specified parameters.  
   - Train until convergence or early-stopping criteria are met.

4. **Evaluate model performance**  
   - Compute accuracy for next-token prediction, FSM-state prediction, and acceptance classification.  
   - Optionally include interpretability diagnostics (e.g., attention–FSM correlation, structure extraction).

5. **Save all artifacts**  
   - Persist configuration, FSM, datasets, checkpoints, logs, metrics, and visualizations.  
   - Record random seed and Git commit hash for reproducibility.

### 6.4 Results Organization
All experiment outputs are stored under `src/results/` (git-ignored).

**Directory structure:**
```
src/results/
└── <experiment_id>/
├── config.py
├── fsm.pkl
├── data/
├── checkpoints/
├── logs/
├── metrics.json
├── visualizations/
└── telemetry.json
```
- **Experiment ID:** Derived from a descriptive name and timestamp or UUID.  
- **Isolation:** Each experiment directory is self-contained; re-running the same configuration regenerates identical outputs.

### 6.5 Experiment Configuration
- **Lightweight and scriptable:** No external experiment manager; all configuration handled in Python.  
- **Explicit provenance:** Every parameter, including defaults and seeds, is saved with results.  
- **Easy iteration:** New experiments require only a few lines of code; parameter sweeps are simple Python loops.  
- **Composable utilities:** Shared modules manage FSM compilation, data generation, model building, and evaluation.  
- **Research-grade simplicity:** Emphasizes transparency and flexibility over automation overhead while preserving full reproducibility.


## 7. Evaluation/Metrics/Visualization

## 7. Evaluation and Metrics

### 7.1 Evaluation Goals
The evaluation framework measures both **task performance** and **model interpretability**.  
Metrics are designed to quantify how well the model learns the formal structure defined by the FSM, not merely token prediction accuracy.

Core goals:
- Verify the model’s ability to reproduce FSM behavior (correct transitions and acceptance).  
- Characterize generalization across lengths, held-out edges, and regex families.  
- Assess alignment between internal representations (attention, activations) and FSM structure.

### 7.2 Evaluation Workflow
Each trained model is evaluated through a structured, multi-stage process:

1. **Load experiment artifacts**  
   - Load FSM, data splits, and trained model checkpoint.  
   - Reconstruct tokenizers and state labelers to ensure alignment.

2. **Run primary evaluations**  
   - Compute metrics on the held-out test set for next-token and state prediction.  
   - Optionally include additional splits (e.g., length-OOD, edge-OOD, or unseen regexes).

3. **Collect interpretability diagnostics**  
   - Analyze attention heads, hidden representations, and linear probes.  
   - Compare learned structure to FSM topology and transition dynamics.

4. **Aggregate and visualize results**  
   - Summarize metrics in tables and plots.  
   - Export visualizations for interpretability analyses (attention maps, probe correlations, graph overlays).

### 7.3 Core Metrics
The following metrics are computed for all experiments:

- **Token accuracy** – Standard next-token prediction accuracy on the test set.  
- **State classification accuracy** – Accuracy of predicting the correct FSM state class (Accept / Incomplete / Reject) per position.  
- **Acceptance decision accuracy** – Accuracy of full-string classification (whether the model predicts a sequence as accepted).  
- **Sequence-level loss** – Average negative log-likelihood over full sequences.  
- **Length generalization gap** – Performance difference between in-distribution and out-of-distribution sequence lengths.  
- **Edge coverage accuracy** – Fraction of FSM transitions correctly modeled during generation or inference.

### 7.4 Interpretability Metrics
To evaluate whether the model’s internal representations reflect the FSM structure:

- **Attention–transition correlation**  
  - Compute the correlation between attention weights and true FSM transition adjacency matrices.  
  - Quantifies whether attention focuses on structurally relevant tokens.

- **State probe accuracy**  
  - Train linear probes on hidden states to predict the FSM state class.  
  - Measures how explicitly FSM structure is encoded in representations.

- **Transition probe accuracy**  
  - Predict next-state IDs or edge types from hidden activations.  
  - Evaluates whether transitions are locally linearizable in representation space.

- **Representational similarity**  
  - Compute cosine similarity or CKA between hidden states grouped by FSM state.  
  - High intra-state similarity indicates structural abstraction.

- **Graph reconstruction score**  
  - Attempt to reconstruct the FSM graph from learned attention or probe connectivity; compare precision/recall against ground truth.

### 7.5 OOD Generalization Tests
Generalization is tested under controlled out-of-distribution regimes:

- **Length-OOD:** Evaluate on sequences longer than seen in training.  
- **Edge-OOD:** Exclude certain transitions during training and test on them.  
- **Regex-OOD:** Train on a subset of regex families and test on structurally similar but unseen ones.  
- **Noise robustness:** Optionally inject token corruption or perturbations to test FSM stability.

### 7.6 Reporting and Visualization
Each experiment produces a standardized results package:

- **metrics.json** – Numerical summaries of all metrics (per split and aggregate).  
- **visualizations/** –  
  - Attention heatmaps  
  - State-probe confusion matrices  
  - Length generalization plots  
  - Graph overlays comparing attention structure to FSM transitions  
- **analysis_notebook.ipynb** – Optional per-experiment notebook for deeper inspection and qualitative analysis.

### 7.7 Reproducibility and Comparison
- All evaluations are deterministic given a fixed random seed and checkpoint.  
- Each experiment stores its metrics and visualizations in its own result directory.  
- Comparison utilities allow aggregation across experiments for tables, charts, and statistical summaries.  
- Consistent evaluation protocols ensure comparability across regexes, model sizes, and data regimes.

## 8. Testing

### 8.1 What Needs Testing

#### 8.1.1 Regex/FSM Implementation
- **Why:** The FSM layer encodes core formal logic and is the foundation for data generation and labeling. Even small errors can propagate through all experiments.  
- **What to test:**  
  - **State transitions:** Verify that δ(q, a) returns the correct next state for all valid (q, a) pairs.  
  - **Classification:** Confirm that each state’s class (Accept, Incomplete, Reject) matches expected semantics.  
  - **Generation:** Ensure that generated strings accepted by the FSM correspond to the true regex language.  
- **How:**  
  - Unit tests for FSM construction, δ-transition mapping, and class assignment.  
  - Round-trip validation of regex ↔ FSM equivalence using small reference patterns (e.g., `a+`, `(ab)*`, `a|b`).  
  - Property-based tests to check closure properties (e.g., concatenation, alternation, repetition).

#### 8.1.2 Experimental Framework
- **Why:** Experiments are the central pipeline; correctness and reproducibility depend on deterministic orchestration.  
- **What to test:**  
  - **Data generation:** Deterministic outputs under fixed seeds, balanced sampling, and class distributions.  
  - **Experiment execution:** End-to-end pipeline (regex → FSM → data → model → results) runs without errors.  
  - **Results saving:** All expected artifacts (config, FSM, checkpoints, metrics, logs) are written correctly.  
- **How:**  
  - Integration tests that instantiate a minimal regex (e.g., `a+`) and run a full training-and-evaluation cycle.  
  - Hash comparisons of generated datasets to confirm reproducibility.  
  - File-existence and schema checks for result directories.

#### 8.1.3 Metrics / Visualization (if custom)
- **Why:** Custom metrics and visualization routines must yield accurate, interpretable results to support analysis.  
- **What to test:**  
  - Metric computation correctness for token accuracy, state prediction, and acceptance classification.  
  - Consistency between numerical outputs and visual summaries (e.g., confusion matrices).  
  - Plot generation sanity (no crashes, correct axis labels, proper scaling).  
- **How:**  
  - Unit tests with small synthetic data and known expected metric values.  
  - Golden-file comparisons for generated plots when applicable.

---

### 8.2 What Doesn’t Need Testing

#### 8.2.1 Transformer Model
- The transformer implementation relies entirely on PyTorch primitives and standard library modules.  
- These components are thoroughly tested upstream; reproducing those tests adds no value.  
- Validation occurs through **training behavior** (loss decrease, overfitting checks), not dedicated unit tests.

---

### 8.3 Testing Approach
- **Unit tests for core logic:**  
  Validate all deterministic functions (FSM transitions, label tracing, metrics).  

- **Sanity checks:**  
  Ensure the model can overfit a tiny dataset (e.g., 10 samples) as a minimal correctness signal.  

- **End-to-end validation:**  
  Run a complete experiment on a trivial regex (e.g., `a+`) to confirm the pipeline operates and results are saved.  

- **Research-grade testing:**  
  Emphasis is on **correctness, determinism, and scientific reproducibility**, not full production coverage.  
  Focus testing effort where implementation complexity or mathematical correctness matters most.

## 9. Code Guidelines

### 9.1 Core Principles
- **PyTorch-based:** Use PyTorch for all neural components; avoid custom autograd unless absolutely necessary.
- **Low overhead:** Prefer simple functions and small modules over abstractions and metaprogramming.
- **Research-grade:** Optimize for readability and editability, not production durability.
- **Fail early:** Prefer assertions and explicit checks over silent fallbacks.
- **Minimal deps:** Standard library + PyTorch; add only when a dependency removes substantial complexity.

### 9.2 Dependencies
- **Required:** `python>=3.10`, `torch`.
- **Optional (allowed):** `numpy` (array ops), `matplotlib` (basic plots), `networkx`/`graphviz` (FSM viz), `tqdm` (progress).
- **Discouraged/forbidden:** Heavy orchestration frameworks, remote experiment trackers, general-purpose “transformers” libraries, bespoke logging stacks.
  - Rationale: keep the conceptual surface small and reproducible; avoid vendor lock-in and hidden state.

### 9.3 Code Organization
- **Modules, not monoliths:** Keep files short and single-purpose (e.g., `fsm.py`, `data_gen.py`, `model.py`, `train.py`, `eval.py`, `experiments/`).
- **No cyclical imports:** Keep dependencies acyclic; pass objects explicitly.
- **Pure-ish utilities:** Make helpers side-effect-free; return values instead of mutating global state.
- **Tight coupling where warranted:** If two functions are always changed together, colocate them. Avoid “generic layers” that obscure the math.

### 9.4 Configuration
- **Code == config:** Experiments are Python objects; no YAML/JSON parsers.
- **Explicit defaults:** Set defaults in constructors; avoid hidden env vars.
- **Provenance:** Store the exact experiment object (repr/pretty-print) with results.
- **Param changes are code changes:** If a setting matters, it lives in code and lands in version control.

### 9.5 Error Handling
- **Aggressive validation:** Assert regex/alphabets are consistent; check FSM totality where required; validate shapes and dtypes.
- **Clear messages:** Assertions should say *what* failed and *why it matters*.
- **No silent casts:** Be explicit about device/dtype transfers; never auto-float or auto-cuda.
- **Boundary checks:** When sampling by target (class, length), verify feasibility before generating; raise informative errors otherwise.

### 9.6 Documentation
- **Docstrings for public APIs:** One-liners are fine; include argument/return semantics and invariants.
- **Comments for non-obvious math:** Especially in FSM construction/minimization and attention/metric routines.
- **README:** How to run an experiment end-to-end; expected outputs; troubleshooting tips.
- **Keep it brief:** Documentation should accelerate editing, not encase it.

---

### 9.7 Reproducibility & Seeds
- **Single seed source:** Thread a single integer seed through Python/NumPy/PyTorch and the data generator.
- **Deterministic flags:** Where practical, set PyTorch determinism options; if you disable them for speed, note it in results.
- **Record context:** Save seed, `torch.__version__`, CUDA info, and current Git commit hash with each experiment.

### 9.8 Logging & Telemetry (Lightweight)
- **Print + tiny logger:** Use `print` or Python’s `logging` with INFO/DEBUG; avoid heavy log infra.
- **Structured crumbs:** Save `metrics.json` and `telemetry.json` (data distributions, coverage, retry rates).
- **No PII/no uploads:** Results are local files under `src/results/`; no network side effects.

### 9.9 Performance & Numerics
- **Default dtypes:** Use `float32` unless a test explicitly needs `float64` (e.g., numeric stability checks).
- **Devices:** Accept `device` as an argument; don’t read global CUDA state inside functions.
- **Batch shapes first:** Prefer batch-first tensors; assert shapes at module boundaries.
- **Micro-optimizations last:** Only optimize hotspots you’ve measured; don’t preempt clarity.

### 9.10 File I/O & Serialization
- **Results live in** `src/results/<experiment_id>/`.
- **Serialization:** Use simple formats—`json` for metrics/telemetry, `pickle/pt` for FSMs/checkpoints (with version stamps).
- **Atomic writes:** Write to temp then rename to avoid partial files on crash.
- **Path hygiene:** No hard-coded absolute paths; use `pathlib`.

### 9.11 Style & Linting (Pragmatic)
- **Type hints:** Helpful for public functions; optional elsewhere.
- **Formatting:** If you want consistency, use `black` (optional); don’t block progress on style nits.
- **Small functions:** Prefer short, testable functions over giant blocks.

### 9.12 Notebooks Policy
- **Optional, not canonical:** Use notebooks for one-off analysis only.
- **Export results:** Any insight from a notebook should be captured by code in `eval/` or saved as artifacts (plots/metrics) under the experiment directory.
- **No hidden state:** Don’t keep training loops or data generation logic only in notebooks.

### 9.13 Testing Scope (Right-Sized)
- **Unit tests:** FSM ops, tracing, metrics math, and data-feasibility checks.
- **Smoke tests:** One end-to-end experiment on a trivial regex (e.g., `a+`) to validate the pipeline.
- **Sanity over coverage:** We are not chasing coverage percentages; we’re guarding against silent wrongness.

### 9.14 API Design Do/Don’t
- **Do:** Pass objects explicitly; return data rather than mutating globals; keep function signatures small and explicit.
- **Don’t:** Use singletons, implicit registries, or magical global configs; hide control flow in decorators/metaclasses.

### 9.15 When to Add a Dependency
- **Add if:** It removes ≥100 lines of tricky code or eliminates a maintenance risk.
- **Don’t add if:** It’s for convenience UX, dashboards, or cloud hooks that complicate reproducibility.

### 9.16 Deprecation & TODOs
- **TODO tags:** Short, dated comments (`# TODO(2025-11-07): reason`) are fine; prune regularly.
- **Deprecation:** Prefer delete/replace over long-lived deprecation layers; the codebase is research, not a public API.

### 9.17 Security & Licensing (Minimal)
- **No secrets:** Never read env secrets or external tokens; keep everything local.
- **Licenses:** Respect third-party tool licenses (Graphviz, etc.); include attributions where required.

---
**Principle of least sophistication:** if a simpler implementation communicates the idea, choose it.

## 10. Non-Features / Out of Scope

The following items are intentionally excluded from the current phase of development.  
They may be revisited in later stages once the core framework is stable and scientifically validated.

- **Multi-layer transformers** – Phase 1 focuses on a single-layer transformer to simplify analysis and interpretability.  
- **Complex regex features** – Backreferences, lookahead/lookbehind, and other non-regular constructs are out of scope; only pure regular languages are supported.  
- **Large-scale experiments** – Dataset sizes, model scales, and compute budgets are kept small for fast iteration and precise diagnostics.  
- **Production deployment** – No serving, packaging, or runtime optimization; experiments run locally or in controlled research environments.  
- **Web interfaces** – No web dashboards or front-end visualization layers; results are local plots and static artifacts.  
- **Distributed or multi-GPU training** – Single-process, single-device operation; scaling deferred to future work.  
- **Interpretability analysis** – Deferred to Phase 2 once the core model and data pipeline are validated.  
- **FSM extraction algorithms** – Reverse-engineering or graph extraction from model weights is future research, not part of this phase.  
- **Automated hyperparameter tuning** – All hyperparameters are manually chosen; no AutoML or sweeping frameworks.

### Rationale
This scope keeps the project lean, interpretable, and reproducible.  
Phase 1’s goal is to **validate the framework**, not to optimize scale or interface polish.  
Every feature above adds either engineering overhead or conceptual noise without advancing the central research question.