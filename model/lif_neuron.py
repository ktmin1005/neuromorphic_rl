import numpy as np

class LIFNeuron:
    def __init__(self, tau=20.0, v_thresh=1.0, v_reset=0.0):
        self.v = 0.0
        self.tau = tau
        self.v_thresh = v_thresh
        self.v_reset = v_reset

    def step(self, input_current, dt=1.0):
        dv = (-self.v + input_current) / self.tau
        self.v += dv * dt
        spike = int(self.v >= self.v_thresh)
        if spike:
            self.v = self.v_reset
        return spike
