# Regex/Transformer Experiment Project

## Overview

An experiment to train a transformer on regular expression patterns and analyze whether the model learns interpretable graph structures that parallel the underlying regex state machines. This serves as a testbed for the broader theory that transformers build relational graphs rather than opaque representations.

## Motivation

### The Core Question
Can we prove that transformers learn interpretable relational structures by training them on a domain where we have complete ground truth?

### Why Regular Expressions?
- **Known ground truth**: Regex patterns have explicit, well-defined state machines
- **Finite state machines**: No recursion complications, clear local state transitions
- **Simple but non-trivial**: Complex enough to be meaningful, simple enough to fully analyze
- **Infinite training data**: Easy to generate arbitrary amounts of valid/invalid strings
- **Visualizable**: State transition graphs can be directly compared to learned structures

### Advantages Over Alternatives
- **Simpler than Lisp**: Avoids recursive structure complications
- **Cleaner than natural language**: Unambiguous semantics
- **Better than toy tasks**: Real computational structure to learn

## Theoretical Foundation

### The Relational Graph Theory of Transformers

**Core Hypothesis**: Transformers operate by building hierarchical relational graphs where:

1. **Attention finds edges**: Attention patterns identify which tokens relate to which others
2. **MLPs compute edge meanings**: MLPs transform attention-weighted values into embeddings that encode "this token has these relationships"
3. **Residual stream performs superposition**: Each layer adds new relationship information to token representations
4. **LayerNorm creates recency bias**: Normalization progressively dilutes older relationship information, creating natural hierarchical abstraction

### What This Predicts for Regex

If the theory is correct, we should observe:

1. **Edge discovery in attention**: Attention patterns should recover the state transition graph
   - Can we identify heads that track "currently in state X"?
   - Do attention patterns match transitions between regex components?

2. **Hierarchical emergence**: 
   - Early layers: character-level transitions
   - Late layers: clause-level structure (kleene star, alternation, etc.)
   - Recency bias should make low-level state info fade while high-level "which regex component am I in" dominates

3. **MLP edge encoding**: 
   - After attention identifies state relationships, MLP outputs should encode positional state information
   - Should be able to decode "current state" from MLP activations

4. **Superposition verification**: 
   - Residual stream should show: `position_embedding = original_character + state_info_layer1 + clause_info_layer2 + ...`
   - Recent layer contributions should have larger magnitude

## Experimental Design

### Training Setup

**Model Architecture**:
- Small transformer (2-4 layers recommended)
- Keep it interpretable - need to trace what each layer does
- Standard architecture (attention + MLP + residual + LayerNorm)

**Training Task Options**:
1. Next character prediction
2. Binary classification: "does this string match the regex?"
3. Both (multi-task)

**Regex Patterns**:
- Start simple: `a*b*`, `a+b+`
- Progress to complex: nested structures, alternation, character classes
- Generate training data by:
  - Creating strings that match the pattern
  - Creating strings that don't match
  - Ensuring good coverage of state space

### Analysis Pipeline

**1. Attention Analysis**:
- Visualize attention patterns for known regex states
- Compare attention adjacency matrices to state transition graphs
- Identify which heads correspond to which types of transitions
- Measure correlation between learned attention and ground truth transitions

**2. MLP Probing**:
- Train linear probes on MLP outputs to predict current state
- Check if state information is linearly decodable
- Examine which layers encode which levels of state abstraction

**3. Residual Stream Analysis**:
- Decompose token representations into layer contributions
- Measure magnitude of contributions from each layer
- Verify recency bias (recent layers dominate)
- Check if early layer info degrades in late layers

**4. Extraction Pipeline**:
- Attempt to extract the learned state machine
- Compare extracted graph to ground truth regex FSM
- Measure extraction accuracy and completeness

### Success Criteria

**Strong Success**: 
- Attention patterns clearly correlate with state transitions (r > 0.8)
- Can extract interpretable state machine from learned weights
- Extracted machine matches ground truth structure
- All four theoretical components (attention/MLP/residual/LayerNorm) show predicted behavior

**Moderate Success**:
- Attention shows some correlation with transitions (r > 0.5)
- Can identify which heads track which state types
- MLP activations encode state information (linear probe accuracy > 80%)
- Hierarchical structure emerges across layers

**Failure (but still informative)**:
- Transformer solves the task but in a completely different way
- No correlation between attention and state transitions
- Can't decode state from internals
- This would indicate the theory needs major revision

## Broader Research Context

### Connection to "Zero as Recognition"
The regex experiment tests whether neural networks fundamentally operate through relation recognition:
- State transitions ARE relations between positions
- Each layer recognizes increasingly abstract relational patterns
- The "zero" (distinction/absence) enables recognition of state boundaries

### Path to Symbolic LLMs
If successful, this experiment proves we can:
1. Train a transformer on data
2. Extract the relational graph it learned
3. Compile that graph into symbolic code

This enables the broader vision:
- **Learning flexibility**: Use transformers to discover algorithms from data
- **Execution efficiency**: Run extracted symbolic algorithms instead of matrix operations
- **Interpretability**: The graph IS the explanation
- **Verifiability**: Restrict edge sets to create provably safe models

### Applications Beyond Regex

Success here opens paths to:
- **Parsers**: Extract learned grammar rules
- **Theorem provers**: Formalize learned inference patterns  
- **Planning algorithms**: Extract decision procedures
- **Program synthesis**: Distill code generation patterns

Any domain where transformers work but are computational overkill.

## Implementation Roadmap

### Phase 1: Setup (Week 1)
- [ ] Design regex patterns (simple to complex)
- [ ] Build training data generator
- [ ] Implement small transformer (2-4 layers)
- [ ] Set up training pipeline
- [ ] Create visualization tools for attention patterns

### Phase 2: Training (Week 2)
- [ ] Train on simple patterns (a*b*, a+b+)
- [ ] Verify model actually learns the task
- [ ] Collect activation data during inference
- [ ] Generate attention visualizations

### Phase 3: Analysis (Weeks 3-4)
- [ ] Analyze attention patterns vs ground truth
- [ ] Train linear probes on MLP outputs
- [ ] Decompose residual stream contributions
- [ ] Measure layer-wise magnitude trends

### Phase 4: Extraction (Weeks 5-6)
- [ ] Develop graph extraction algorithms
- [ ] Compare extracted FSM to ground truth
- [ ] Measure extraction accuracy
- [ ] Test on increasingly complex patterns

### Phase 5: Iteration (Weeks 7-8)
- [ ] Identify where theory breaks or needs refinement
- [ ] Try variations (different architectures, training objectives)
- [ ] Scale to more complex regex patterns
- [ ] Document negative results

## Expected Challenges

### Technical Challenges
- **Attention may be diffuse**: Not all heads may correspond to clean state transitions
- **Superposition is messy**: Multiple types of information mixed in same dimensions
- **Extraction is hard**: Going from weights to explicit graph structure is non-trivial

### Interpretation Challenges
- **Correlation ≠ Causation**: Attention patterns correlating with transitions doesn't prove the model uses them that way
- **Multiple solutions**: Model might solve the task differently than expected
- **Emergent behavior**: Higher-level patterns may not be predictable from lower layers

### Practical Challenges
- **Hyperparameter sensitivity**: Results may depend heavily on architecture choices
- **Training instability**: Small models can be finicky
- **Visualization complexity**: Hard to visualize high-dimensional spaces

## Success Metrics

### Quantitative Metrics
- **State prediction accuracy**: Can we decode current state from activations?
- **Attention-transition correlation**: How well do attention patterns match ground truth?
- **Extraction fidelity**: How closely does extracted FSM match ground truth?
- **Hierarchical emergence**: Do early vs late layers show predicted specialization?

### Qualitative Metrics
- **Interpretability**: Can a human look at the learned structure and understand it?
- **Generalization**: Does the model learn the pattern or memorize examples?
- **Robustness**: Does the structure persist across training runs?

## Timeline

**Optimistic (2 months)**:
- Month 1: Setup, training, initial analysis
- Month 2: Deep analysis, extraction, iteration

**Realistic (3-4 months)**:
- Month 1: Setup and debugging
- Month 2: Training and basic analysis
- Month 3: Deep analysis and extraction attempts
- Month 4: Iteration and refinement

**Pessimistic (6+ months)**:
- Debugging takes longer than expected
- Need multiple architectural iterations
- Extraction proves very difficult
- Theory needs significant revision

## Documentation Requirements

### Code Documentation
- Clear README with setup instructions
- Well-commented training code
- Reproducible experiments (fixed seeds, saved configs)
- Visualization notebooks

### Scientific Documentation
- **Experiment log**: What was tried, what worked, what didn't
- **Negative results**: Document failures - they're valuable!
- **Surprising findings**: Unexpected patterns, anomalies
- **Theoretical implications**: What does this mean for the broader theory?

### Public Artifacts
- GitHub repository with all code
- Blog post or paper describing findings
- Visualization examples
- Trained model checkpoints

## Why This Matters

This isn't just about regex. It's about answering fundamental questions:

1. **Do transformers learn interpretable structure?** Or are they fundamentally opaque?

2. **Can we extract algorithms from neural networks?** This is the key to making AI systems verifiable and trustworthy.

3. **Is the relational graph theory correct?** If yes, it changes how we think about AI. If no, we need a better theory.

4. **Can we build better AI systems?** Symbolic extraction could give us the best of both worlds: learning from data + efficient execution + interpretability.

The regex experiment is small and tractable, but it tests big ideas. If it works, it's the foundation for verifiable AI, efficient algorithms, and trustworthy systems.

If it doesn't work as expected, we learn something equally valuable about the limits of current theory and what needs to change.

Either way: it's good science.

---

## References to Past Conversations

Key insights from previous discussions:
- [Transformer models as relational graph builders](https://claude.ai/chat/1f3ffbad-5b1e-471b-a5cd-50d7efa8d968)
- Discussion of attention finding edges, MLPs computing meanings
- LayerNorm as forgetting mechanism creating recency bias
- Connection to "Zero as Recognition" framework
- Path from regex → symbolic LLMs → verifiable AI systems

## Next Steps

1. Review this document and refine experimental design
2. Set up development environment
3. Start with simplest possible version (single regex pattern, tiny model)
4. Build incrementally, validating each step
5. Document everything - negative results matter!

The key is to start small, measure carefully, and build understanding iteratively. This isn't a race - it's foundational research that needs to be done right.