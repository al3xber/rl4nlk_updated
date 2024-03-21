from abc import ABCMeta, abstractmethod
from typing import Sequence


class PropagatorInterface(metaclass=ABCMeta):
    @abstractmethod
    def propagation_single_round(*, x: Sequence[float], px: Sequence[float], kicker_strength : float,
                                 noise_x : float, noise_px : float, noise_NLK : float):
        raise NotImplementedError

    @abstractmethod
    def propagation_thousand_rounds(*, x: Sequence[float], px: Sequence[float], kicker_strength : float = 0.0,
                                    when_activate_NLK: int = 0, noise_x = 0.0, noise_px = 0.0, noise_NLK = 0.0, 
                                    noise_first_round = 0.0):
        raise NotImplementedError
        
        
        
__all__=["PropagatorInterface"]