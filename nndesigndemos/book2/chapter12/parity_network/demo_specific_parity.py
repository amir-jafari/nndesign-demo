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
        outputs = network.forward(sequence)
        print(f"Input: {sequence}")
        print(f"Output: {[int(x) for x in outputs]}")
        
        all_correct = True
        for i in range(len(sequence)):
            cumulative_ones = sum(sequence[:i+1])
            expected_output = 1 if cumulative_ones % 2 == 1 else 0
            actual_output = int(outputs[i])
            is_correct = actual_output == expected_output
            all_correct = all_correct and is_correct
            print(f"Time {i+1}: cumulative_ones={cumulative_ones}, expected={expected_output}, actual={actual_output}, correct={is_correct}")
        
        print(f"All time steps correct: {all_correct}")
        print()

if __name__ == "__main__":
    run()