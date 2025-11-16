#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class CEC:
    
    def __init__(self):
        self.IW11 = 1.0
        self.LW11 = 1.0
        self.bias = 0.0
        self.delay = 0.0
        
    def forward(self, input_sequence):
        output_sequence = []
        delay_states = [self.delay]
        
        current_delay = self.delay
        
        for input_val in input_sequence:
            new_delay = current_delay + self.IW11 * input_val
            output = new_delay
            
            output_sequence.append(output)
            delay_states.append(new_delay)
            
            current_delay = new_delay
            
        return np.array(output_sequence), np.array(delay_states)
    
    def reset_delay(self):
        self.delay = 0.0

def main():
    
    print("=== CEC Implementation ===")
    print(f"IW1,1 = 1")
    print(f"LW1,1 = 1")
    print(f"Initial Delay = 0")
    print()
    
    cec = CEC()
    
    sequence_length = 10
    input_sequence = np.ones(sequence_length)
    
    print(f"Input sequence (p): {input_sequence}")
    
    output_sequence, delay_states = cec.forward(input_sequence)
    
    print(f"Output sequence: {output_sequence}")
    print(f"Delay states: {delay_states}")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    time_steps = np.arange(len(input_sequence))
    plt.plot(time_steps, input_sequence, 'bo-', linewidth=2, markersize=8, label='Input p')
    plt.title('Input Sequence')
    plt.xlabel('Time Step')
    plt.ylabel('Input Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(time_steps, output_sequence, 'ro-', linewidth=2, markersize=8, label='Output')
    plt.title('Output Sequence')
    plt.xlabel('Time Step')
    plt.ylabel('Output Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    delay_time_steps = np.arange(len(delay_states))
    plt.plot(delay_time_steps, delay_states, 'go-', linewidth=2, markersize=8, label='Delay/Memory States')
    plt.title('Delay/Memory States')
    plt.xlabel('Time Step')
    plt.ylabel('Delay Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(time_steps, input_sequence, 'bo-', linewidth=2, markersize=8, label='Input p', alpha=0.7)
    plt.plot(time_steps, output_sequence, 'ro-', linewidth=2, markersize=8, label='Output', alpha=0.7)
    plt.title('Input vs Output Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Testing CEC with different sequences ===")
    
    cec.reset_delay()
    
    test_sequences = [
        [1, 1, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0]
    ]
    
    plt.figure(figsize=(15, 5))
    
    for i, seq in enumerate(test_sequences):
        cec.reset_delay()
        seq = np.array(seq, dtype=float)
        output, delays = cec.forward(seq)
        
        plt.subplot(1, 3, i+1)
        time_steps = np.arange(len(seq))
        plt.plot(time_steps, seq, 'bo-', linewidth=2, markersize=6, label='Input', alpha=0.7)
        plt.plot(time_steps, output, 'ro-', linewidth=2, markersize=6, label='Output', alpha=0.7)
        plt.title(f'Test Sequence {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        print(f"Test {i+1} - Input: {seq}")
        print(f"Test {i+1} - Output: {output.round(3)}")
        print()
    
    plt.tight_layout()
    plt.show()
    
    print("CEC implementation completed!")

if __name__ == "__main__":
    main()