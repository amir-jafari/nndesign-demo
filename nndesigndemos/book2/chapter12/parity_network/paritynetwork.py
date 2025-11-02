import numpy as np

class SpecificParityNetwork:
    def __init__(self):
        self.IW11 = np.array([1, 1])
        self.LW11 = np.array([[1, -1], [1, -1]])
        self.LW21 = np.array([1, -1])
        self.b1 = np.array([-0.5, -1.5])
        self.b2 = -0.5
        self.reset_state()
    
    def reset_state(self):
        self.delay_state = np.zeros(2)
    
    def hard_limit(self, x):
        if isinstance(x, np.ndarray):
            return (x >= 0).astype(float)
        else:
            return 1.0 if x >= 0 else 0.0
    
    def step(self, p):
        n1 = self.IW11 * p + self.b1
        feedback = np.dot(self.LW11, self.delay_state)
        n1 = n1 + feedback
        a1 = self.hard_limit(n1)
        n2 = np.dot(self.LW21, a1) + self.b2
        output = self.hard_limit(n2)
        self.delay_state = a1.copy()
        return output
    
    def forward(self, sequence):
        self.reset_state()
        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence)
        outputs = np.zeros(len(sequence))
        for i, p in enumerate(sequence):
            outputs[i] = self.step(p)
        return outputs