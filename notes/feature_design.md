# Feature Design Document - OUTLINE

## 1. Goals

### 1.1 Phase 1 Objectives
- Build regex/FSM system
- Generate training data from FSM
- Train transformer model on regex patterns
- Verify model can learn the task
- Establish baseline performance

### 1.2 Capacity Experiments
- Test model learning with minimal capacity (below theoretical requirements)
- Test model learning with excess capacity
- Understand relationship between FSM complexity and model requirements
- Inform model sizing heuristics

### 1.3 Future Work (Out of Scope for Phase 1)
- Interpretability analysis
- FSM extraction
- Attention pattern analysis
- *(Deferred until we have working trained models)*

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

### 3.1 Architecture
- Single transformer block
- Multi-head self-attention
- MLP/feed-forward layer
- Residual connections
- LayerNorm
- Position embeddings

### 3.2 Input/Output
- **Input:** character sequences (token-level)
- **Output heads:**
  - Next token prediction (vocab_size logits)
  - State classification (num_classes logits)
  - State class prediction does NOT include next token in context

### 3.3 Vocabulary
- Character-level tokens
- One character = one token
- Special tokens: EOS, possibly PAD

### 3.4 Model Sizing
- Embedding dimension informed by FSM properties
- Number of attention heads informed by FSM complexity
- Configurable, not hardcoded
- Support for capacity experiments (undersized/oversized models)

### 3.5 Instrumentation
- Ability to extract attention weights
- Ability to extract MLP activations
- Ability to extract residual stream values
- Support future interpretability analysis (but don't implement analysis yet)

## 4. Data Generation

### 4.1 Generation Strategy
- Use FSM to generate valid strings
- Sample transitions to create diverse examples
- Control string length distribution
- Ensure coverage of different state classes

### 4.2 Data Characteristics
- Strings reaching accept states
- Strings in incomplete states (mid-FSM)
- Strings in reject states (illegal transitions)
- Balanced distribution across classes (or configurable)

### 4.3 Labeling
- **Per-position labels:**
  - Next token (the following character or EOS)
  - State class (current FSM state class, before next token)
- Trace string through FSM to generate labels
- Alignment with transformer input positions

### 4.4 Dataset Splits
- Training set
- Validation set
- Test set
- Configurable sizes

### 4.5 Reproducibility
- Seeded random generation
- Deterministic data creation

## 5. Experimental Framework

### 5.1 Experiment Definition
- Define experiments in Python code (not JSON/YAML)
- Direct value assignment, no translation overhead
- Easy to create new experiments programmatically

### 5.2 Experiment Components
- Regex definition
- Data generation parameters
- Model architecture parameters
- Training hyperparameters
- Random seed

### 5.3 Experiment Execution
1. Take regex definition → generate FSM
2. Generate train/validation/test data from FSM
3. Build and train transformer model
4. Evaluate model performance
5. Save all artifacts

### 5.4 Results Organization
- Results go to `src/results/` (gitignored)
- Organized by experiment name/ID
- **Each experiment folder contains:**
  - Experiment configuration
  - Generated FSM
  - Trained model checkpoints
  - Training logs
  - Evaluation metrics
  - Visualizations

### 5.5 Experiment Configuration
- Lightweight, research-grade approach
- Easy to modify and iterate
- No heavy frameworks
- Clear parameter provenance

## 6. Evaluation/Metrics/Visualization

### 6.1 Training Metrics
- Loss (total, next-token component, state-class component)
- Accuracy (next-token, state-class)
- Per-class accuracy for state classification
- Training curves over epochs

### 6.2 Evaluation Metrics
- Test set accuracy (next-token, state-class)
- Confusion matrix for state classification
- Per-class performance breakdown
- Comparison to baseline (random, simple heuristic)

### 6.3 Capacity Analysis Metrics
- Performance vs model size
- Minimum capacity for convergence
- Performance saturation point

### 6.4 Visualizations
- Training/validation loss curves
- Accuracy curves
- Confusion matrices
- FSM graph visualization
- *(Future: attention heatmaps, activation visualizations)*

### 6.5 Outputs
- Plots saved to experiment results directory
- Summary metrics in logs
- Easy to compare across experiments

## 7. Testing

### 7.1 What Needs Testing

#### 7.1.1 Regex/FSM Implementation
- **Why:** Complex logic, easy to get wrong
- **What:** State transitions, classification, generation
- Unit tests for FSM operations
- Validation against known regex semantics

#### 7.1.2 Experimental Framework
- **Why:** Ensure reproducibility and correctness
- **What:** Data generation, experiment execution, results saving
- Integration tests for end-to-end workflow

#### 7.1.3 Metrics/Visualization (if custom)
- **Why:** Ensure accurate measurement
- **What:** Metric calculations, plot generation
- Unit tests for metric computation

### 7.2 What Doesn't Need Testing

#### 7.2.1 Transformer Model
- Standard PyTorch implementation
- Trust PyTorch primitives
- Validate through training, not unit tests

### 7.3 Testing Approach
- Unit tests for core logic
- Sanity checks (can model overfit tiny dataset?)
- End-to-end validation on simple regex
- Research-grade testing (not production-grade)

## 8. Code Guidelines

### 8.1 Core Principles
- **PyTorch-based:** Use PyTorch for all neural network components
- **Low overhead:** Minimal abstractions, direct implementations
- **Research-grade code:** Readable, modifiable, not production-hardened
- **Early failure preference:** Don't hide errors, fail fast and loud
- **Minimal dependencies:** Standard library + PyTorch + minimal extras

### 8.2 Dependencies
- **Required:** PyTorch, Python standard library
- **Optional:** matplotlib (plots), networkx/graphviz (FSM viz), numpy (if needed)
- **Forbidden:** Heavy frameworks, experiment tracking services, transformers library

### 8.3 Code Organization
- Clear module structure
- Self-contained components
- Easy to understand and modify
- Avoid over-engineering

### 8.4 Configuration
- Experiments defined in code, not config files
- Direct Python objects
- No config parsers or translators
- Easy to version control and diff

### 8.5 Error Handling
- Validate inputs aggressively
- Assert assumptions explicitly
- Clear error messages
- Fail rather than silently produce wrong results

### 8.6 Documentation
- Docstrings for public APIs
- Comments for non-obvious logic
- README for setup and usage
- Research-appropriate level of documentation

## 9. Non-Features / Out of Scope

- Multi-layer transformers (start with 1 layer)
- Complex regex features (backreferences, lookahead/lookbehind)
- Large-scale experiments (keep it small and interpretable)
- Production deployment
- Web interfaces
- Distributed training
- Interpretability analysis (deferred to Phase 2)
- FSM extraction algorithms (future work)
- Automated hyperparameter tuning