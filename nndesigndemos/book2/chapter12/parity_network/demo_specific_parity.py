import numpy as np
from paritynetwork import SpecificParityNetwork

def run():
    network = SpecificParityNetwork()
    examples = [
        # [1],
        # [0],
        # [1, 0],
        # [1, 1],
        # [1, 0, 1],
        # [1, 1, 0, 1],
        # [1, 0, 1, 1, 0],
        # [1, 0, 1, 0, 1, 0],
        # [1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 1, 1],
    ]
    
    for sequence in examples:
        a1_outputs, a2_outputs = network.forward(sequence)
        print(f"Input: {sequence}")
        print(f"Layer 1 Output (a¹): {[[int(x) for x in row] for row in a1_outputs]}")
        print(f"Layer 2 Output (a²): {[int(x) for x in a2_outputs]}")
        
        all_correct = True
        for i in range(len(sequence)):
            cumulative_ones = sum(sequence[:i+1])
            expected_output = 1 if cumulative_ones % 2 == 1 else 0
            actual_output = int(a2_outputs[i])
            is_correct = actual_output == expected_output
            all_correct = all_correct and is_correct
            print(f"Time {i+1}: cumulative_ones={cumulative_ones}, expected={expected_output}, actual={actual_output}, correct={is_correct}")
        
        print(f"All time steps correct: {all_correct}")
        print()

if __name__ == "__main__":
    run()