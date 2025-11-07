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
- Tuple specification: (alphabet, patterns)
- Pattern-to-class mapping
- Implicit reject state for undefined transitions
- Multiple patterns/classes in single definition

### 2.2 FSM Representation
- FSM tuple: (Q, Σ, δ, q₀, C)
- Minimal FSM (fewest states)
- State classification function
- Explicit vs implicit reject state

### 2.3 FSM Operations

#### 2.3.1 Recognition/Tracing
- Classify input string (what class does it match?)
- Trace state sequence for input string
- State-by-state classification during trace

#### 2.3.2 Forward Generation
- From current state, enumerate next possibilities
- Output: (character, next_state, next_class) tuples
- Enable data generation by walking FSM

#### 2.3.3 Backward Generation (Optional)
- From current state, enumerate previous possibilities
- Start from accept states, work backward
- Enable targeted generation of accepting strings

### 2.4 FSM Properties
- State count, class distribution
- Transition count
- Reachability information
- Metadata for model sizing decisions

### 2.5 FSM Serialization
- Save/load FSM to disk
- Human-readable format
- Reproducibility support

### 2.6 FSM Visualization
- Render as directed graph
- Color states by class
- Show transitions with labels
- Export to image formats

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