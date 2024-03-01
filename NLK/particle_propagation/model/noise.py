from dataclasses import dataclass


@dataclass
class NoiseModel:
    x: float
    px: float
    nlk: float
    first_round: float