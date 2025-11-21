#!/usr/bin/env python3
"""
Gated Constant Error Carousel (CEC) Implementation
This implements a CEC with input gating (g^i) and feedback gating (g^f)
"""

import numpy as np
import matplotlib.pyplot as plt

class GatedCEC:
    """
    Gated CEC with input and feedback gates.

    The network operates as:
    - Input gate g^i(t) controls how much of the input p(t) enters the memory
    - Feedback gate g^f(t) controls how much of the previous state is retained
    - Memory state: s(t) = g^f(t) * s(t-1) + g^i(t) * p(t)
    - Output: a(t) = s(t)
    """

    def __init__(self, initial_state=0.0):
        """
        Initialize the Gated CEC.

        Args:
            initial_state: Initial memory state (default: 0.0)
        """
        self.initial_state = initial_state
        self.state = initial_state

    def forward(self, input_signal, input_gate, feedback_gate):
        """
        Process sequences through the gated CEC.

        Args:
            input_signal: p(t) - input sequence
            input_gate: g^i(t) - input gating signal (0 blocks, 1 passes)
            feedback_gate: g^f(t) - feedback gating signal (0 clears, 1 retains)

        Returns:
            output_sequence: a(t) - network output
            state_sequence: internal memory states at each time step
        """
        # Convert to numpy arrays
        p = np.array(input_signal, dtype=float)
        gi = np.array(input_gate, dtype=float)
        gf = np.array(feedback_gate, dtype=float)

        # Check that all sequences have the same length
        length = len(p)
        assert len(gi) == length, "Input gate must have same length as input"
        assert len(gf) == length, "Feedback gate must have same length as input"

        # Initialize output and state sequences
        output_sequence = np.zeros(length)
        state_sequence = np.zeros(length + 1)
        state_sequence[0] = self.initial_state

        # Process each time step
        for t in range(length):
            # Update state: s(t) = g^f(t) * s(t-1) + g^i(t) * p(t)
            self.state = gf[t] * self.state + gi[t] * p[t]

            # Output equals current state
            output_sequence[t] = self.state
            state_sequence[t + 1] = self.state

        return output_sequence, state_sequence

    def reset_state(self):
        """Reset the internal state to initial value."""
        self.state = self.initial_state


def explain_gates():
    """Print explanation of how each gate affects network operation."""
    print("=" * 80)
    print("GATED CEC OPERATION EXPLANATION")
    print("=" * 80)
    print()
    print("The Gated Constant Error Carousel (CEC) uses two gating mechanisms:")
    print()
    print("1. INPUT GATE g^i(t):")
    print("   - Controls how much of the input p(t) is added to the memory")
    print("   - When g^i(t) = 1: Input is fully added to memory")
    print("   - When g^i(t) = 0: Input is blocked from memory")
    print("   - Allows the network to selectively store information")
    print()
    print("2. FEEDBACK GATE g^f(t):")
    print("   - Controls how much of the previous state is retained")
    print("   - When g^f(t) = 1: Previous state is fully retained (perfect memory)")
    print("   - When g^f(t) = 0: Previous state is erased (memory reset)")
    print("   - Allows the network to forget or retain past information")
    print()
    print("MEMORY UPDATE EQUATION:")
    print("   s(t) = g^f(t) * s(t-1) + g^i(t) * p(t)")
    print()
    print("   where:")
    print("   - s(t) is the current memory state")
    print("   - s(t-1) is the previous memory state")
    print("   - p(t) is the current input")
    print()
    print("OUTPUT:")
    print("   a(t) = s(t)")
    print("   The output directly reflects the current memory state")
    print()
    print("=" * 80)
    print()


def plot_gated_cec(input_signal, input_gate, feedback_gate, output, states):
    """
    Create comprehensive visualization of the gated CEC operation.

    Args:
        input_signal: p(t)
        input_gate: g^i(t)
        feedback_gate: g^f(t)
        output: a(t)
        states: internal states
    """
    time_steps = np.arange(len(input_signal))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Input signal p(t)
    ax1 = axes[0, 0]
    ax1.stem(time_steps, input_signal, linefmt='b-', markerfmt='bo', basefmt='k-', label='p(t)')
    ax1.set_title('Input Signal p(t)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step t', fontsize=12)
    ax1.set_ylabel('p(t)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim(-0.2, 1.2)

    # Plot 2: Input gate g^i(t)
    ax2 = axes[0, 1]
    ax2.stem(time_steps, input_gate, linefmt='g-', markerfmt='go', basefmt='k-', label='$g^i(t)$')
    ax2.set_title('Input Gate $g^i(t)$', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Step t', fontsize=12)
    ax2.set_ylabel('$g^i(t)$', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim(-0.2, 1.2)

    # Plot 3: Feedback gate g^f(t)
    ax3 = axes[1, 0]
    ax3.stem(time_steps, feedback_gate, linefmt='m-', markerfmt='mo', basefmt='k-', label='$g^f(t)$')
    ax3.set_title('Feedback Gate $g^f(t)$', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time Step t', fontsize=12)
    ax3.set_ylabel('$g^f(t)$', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    ax3.set_ylim(-0.2, 1.2)

    # Plot 4: Output a(t)
    ax4 = axes[1, 1]
    ax4.stem(time_steps, output, linefmt='r-', markerfmt='ro', basefmt='k-', label='a(t)')
    ax4.set_title('Network Output a(t)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time Step t', fontsize=12)
    ax4.set_ylabel('a(t)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('gated_cec_signals.png', dpi=150, bbox_inches='tight')
    print("Saved figure: gated_cec_signals.png")
    plt.show()

    # Create a second figure showing all signals together
    fig2, ax = plt.subplots(figsize=(15, 8))

    # Use stem plots for all signals
    markerline1, stemlines1, baseline1 = ax.stem(time_steps, input_signal, linefmt='b-', markerfmt='bo', basefmt=' ', label='p(t) - Input')
    markerline2, stemlines2, baseline2 = ax.stem(time_steps, input_gate, linefmt='g-', markerfmt='go', basefmt=' ', label='$g^i(t)$ - Input Gate')
    markerline3, stemlines3, baseline3 = ax.stem(time_steps, feedback_gate, linefmt='m-', markerfmt='mo', basefmt=' ', label='$g^f(t)$ - Feedback Gate')
    markerline4, stemlines4, baseline4 = ax.stem(time_steps, output, linefmt='r-', markerfmt='ro', basefmt=' ', label='a(t) - Output')

    # Set transparency for better visualization
    stemlines1.set_alpha(0.6)
    stemlines2.set_alpha(0.6)
    stemlines3.set_alpha(0.6)
    stemlines4.set_alpha(0.8)
    markerline4.set_markersize(8)

    ax.set_title('Gated CEC: All Signals', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time Step t', fontsize=13)
    ax.set_ylabel('Signal Value', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    ax.axhline(y=0, color='k', linewidth=0.8)  # Add baseline at y=0

    plt.tight_layout()
    plt.savefig('gated_cec_combined.png', dpi=150, bbox_inches='tight')
    print("Saved figure: gated_cec_combined.png")
    plt.show()


def main():
    """Main function to demonstrate the gated CEC."""

    # Print explanation first
    explain_gates()

    # Create the sequences from the PDF diagram
    print("SIMULATING THE EXAMPLE FROM NEW_CEC.PDF")
    print("=" * 80)
    print()

    # Define the sequences based on the PDF figures
    # p(t): all ones from t=0 to t=12
    p_t = np.ones(13)

    # g^i(t): ones from t=0 to t=4, zeros from t=5 to t=6, ones from t=7 to t=12
    gi_t = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1])

    # g^f(t): ones from t=0 to t=6, zeros at t=7, ones from t=8 to t=10, zeros at t=11 to t=12
    gf_t = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0])

    print("Input signal p(t):")
    print(f"  {p_t}")
    print()
    print("Input gate g^i(t):")
    print(f"  {gi_t}")
    print()
    print("Feedback gate g^f(t):")
    print(f"  {gf_t}")
    print()

    # Create and run the gated CEC
    cec = GatedCEC(initial_state=0.0)
    output, states = cec.forward(p_t, gi_t, gf_t)

    print("Network output a(t):")
    print(f"  {output}")
    print()
    print("Memory states s(t) [including initial state]:")
    print(f"  {states}")
    print()

    # Analyze specific time steps
    print("=" * 80)
    print("STEP-BY-STEP ANALYSIS:")
    print("=" * 80)
    print()

    for t in range(len(p_t)):
        prev_state = states[t]
        curr_state = states[t + 1]
        print(f"Time t={t}:")
        print(f"  p({t}) = {p_t[t]}, g^i({t}) = {gi_t[t]}, g^f({t}) = {gf_t[t]}")
        print(f"  s({t}) = g^f({t}) * s({t-1 if t > 0 else 'init'}) + g^i({t}) * p({t})")
        print(f"  s({t}) = {gf_t[t]} * {prev_state} + {gi_t[t]} * {p_t[t]} = {curr_state}")
        print(f"  a({t}) = {output[t]}")

        # Add interpretation
        if gi_t[t] == 0 and gf_t[t] == 1:
            print(f"  --> Input blocked, memory retained")
        elif gi_t[t] == 1 and gf_t[t] == 0:
            print(f"  --> Input added, memory cleared")
        elif gi_t[t] == 0 and gf_t[t] == 0:
            print(f"  --> Both blocked, memory reset to 0")
        elif gi_t[t] == 1 and gf_t[t] == 1:
            print(f"  --> Input added, memory retained (accumulation)")
        print()

    # Create visualizations
    plot_gated_cec(p_t, gi_t, gf_t, output, states)

    print("=" * 80)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()