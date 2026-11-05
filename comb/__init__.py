"""CoMB: the coherent memory Born machine.

A register of fixed size generates a sample one component per round: a work register is entangled with a memory
register and measured, and the memory, never measured, carries the correlations between rounds. Losses and gradients
are exact, through the characteristic function of the generated distribution.

    comb.showers   calorimeter showers (CaloChallenge Dataset 1)
    comb.digits    handwritten digits drawn as pen strokes
"""
__version__ = "1.0.0"
