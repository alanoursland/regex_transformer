"""FSM to QKV construction - direct encoding of finite state machines into attention weights."""

from typing import Dict, List, Tuple
from fsm.dfa import FSM


def construct_qkv_from_fsm(fsm: FSM) -> Dict[str, List]:
    """
    Construct QKV matrices that exactly implement an FSM.
    
    Theory:
        - d_model = num_states (one-hot state encoding)
        - n_heads = len(alphabet) (one head per input symbol)
        - Each position i encodes "current state after processing input[:i]"
        - Head h processes symbol alphabet[h]
        - Attention: position i looks back to position i-1
        - Value: encodes transition δ(state[i-1], symbol[i]) = next_state
    
    Args:
        fsm: Deterministic finite automaton
        
    Returns:
        Dictionary with:
            - 'V': Value matrices as nested lists [n_heads][from_state][to_state]
                   V[head][state][next] = 1.0 if δ(state, symbol[head]) = next, else 0.0
            - 'initial_state': One-hot encoding of start state as list
    """
    num_states = fsm.states
    num_heads = len(fsm.alphabet)
    
    # Initialize value matrices (one per head)
    # V[head][from_state] = one-hot vector of next_state
    V = []
    
    for head_idx in range(num_heads):
        token_id = head_idx
        
        # Value matrix for this head
        head_V = []
        for state in range(num_states):
            # One-hot encoding of next state
            next_state = fsm.step(state, token_id)
            one_hot = [0.0] * num_states
            one_hot[next_state] = 1.0
            head_V.append(one_hot)
        
        V.append(head_V)
    
    # Initial state encoding (start state as one-hot)
    initial_state = [0.0] * num_states
    initial_state[fsm.start] = 1.0
    
    return {
        'V': V,
        'initial_state': initial_state,
    }


def matrix_vector_mult(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Multiply matrix by vector."""
    result = []
    for row in matrix:
        result.append(sum(row[i] * vector[i] for i in range(len(vector))))
    return result


def argmax(vector: List[float]) -> int:
    """Return index of maximum value."""
    return max(range(len(vector)), key=lambda i: vector[i])


def fsm_forward_pass(
    qkv: Dict[str, List],
    tokens: List[int],
    fsm: FSM,
) -> Tuple[List[List[float]], List[int]]:
    """
    Execute constructed QKV matrices on an input sequence.
    
    Args:
        qkv: Dictionary with V and initial_state
        tokens: List of token IDs (indices into alphabet)
        fsm: The FSM (needed for reference)
        
    Returns:
        - states_encoding: List of one-hot state encodings [seq_len+1][d_model]
        - predicted_states: List of state IDs predicted by attention
    """
    V = qkv['V']
    initial_state = qkv['initial_state']
    
    # State encodings at each position
    states = [initial_state]
    predicted_state_ids = [fsm.start]
    
    for token_id in tokens:
        # Select the value matrix for this token (head = token_id)
        head_V = V[token_id]
        
        # Get previous state encoding
        prev_state = states[-1]
        
        # Apply value transform: next_state = V @ prev_state
        next_state_encoding = matrix_vector_mult(head_V, prev_state)
        
        states.append(next_state_encoding)
        
        # Decode to state ID (argmax of one-hot)
        predicted_state_id = argmax(next_state_encoding)
        predicted_state_ids.append(predicted_state_id)
    
    return states, predicted_state_ids


def test_equivalence(
    fsm: FSM,
    test_strings: List[str],
    verbose: bool = True
) -> bool:
    """
    Test if FSM and QKV construction produce identical results.
    
    Args:
        fsm: The finite state machine
        test_strings: List of strings to test
        verbose: Print detailed comparison
        
    Returns:
        True if all tests pass, False otherwise
    """
    qkv = construct_qkv_from_fsm(fsm)
    
    all_passed = True
    
    for test_str in test_strings:
        # FSM execution
        tokens = fsm.tokens_from_string(test_str) if test_str else []
        fsm_states = fsm.trace(tokens)
        fsm_final_class = fsm.classify_name(fsm_states[-1])
        
        # QKV execution
        qkv_encodings, qkv_states = fsm_forward_pass(qkv, tokens, fsm)
        qkv_final_class = fsm.classify_name(qkv_states[-1])
        
        # Compare
        match = fsm_states == qkv_states
        
        if verbose:
            status = "✓" if match else "✗"
            print(f"{status} '{test_str}'")
            print(f"  FSM states: {fsm_states}")
            print(f"  QKV states: {qkv_states}")
            print(f"  FSM class:  {fsm_final_class}")
            print(f"  QKV class:  {qkv_final_class}")
            if not match:
                print(f"  MISMATCH!")
            print()
        
        if not match:
            all_passed = False
    
    return all_passed