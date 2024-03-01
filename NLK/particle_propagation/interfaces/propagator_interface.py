from abc import ABCMeta, abstractmethod
from typing import Sequence
from ..model.noise import NoiseModel


class PropagatorInterface(metaclass=ABCMeta):
    @abstractmethod
    def propagation_single_round(*, x: Sequence[float], px: Sequence[float], kicker_strength : float,
                                 noise : NoiseModel):
        raise NotImplementedError

    @abstractmethod
    def propagation_thousand_rounds(*, x: Sequence[float], px: Sequence[float], kicker_strength : float,
                                    when_activate_NLK: int = 1,
                                    noise : NoiseModel):
        raise NotImplementedError
