import numpy as np
import os

from particle_propagation.thorscsi_propagation import Thor_SCSI_Propagator
from particle_propagation.approx_propagation import Approximated_Propagator
from interfaces.propagator_interface import PropagatorInterface

class Particle_Propagator():


    def __init__(self, propagator : PropagatorInterface):

        self.Propagator = propagator()
        
        self.given_noise_first_round = 65e-6
        self.given_noise_x = 6.5e-6
        self.given_noise_px = .8*6.5e-6
        self.given_noise_NLK = 0.0125
        
    def propagation_thousand_rounds(self, x, px, when_activate_NLK=0, kicker_strength=0.0,
                                    deterministic = False):
        """
        Input: x,px; NumpyArray x and px information
               y_rounds_to_save; List which px rounds so save
        """
        assert len(x) == len(px)
        assert (kicker_strength >= -1) and (kicker_strength <= 1)

        
        self.x = x
        self.px = px
        self.kicker_strength = kicker_strength
        self.when_activate_NLK = when_activate_NLK

        if deterministic:
            noise_first_round, noise_x, noise_px, noise_NLK = 0, 0, 0, 0
        else:
            noise_first_round = self.given_noise_first_round
            noise_x = self.given_noise_x
            noise_px = self.given_noise_px
            noise_NLK = self.given_noise_NLK


        
        result = self.Propagator.propagation_thousand_rounds(x, px, when_activate_NLK, kicker_strength,
                 noise_x = noise_x, noise_px = noise_px, noise_NLK = noise_NLK, 
                 noise_first_round = noise_first_round)
        #result has form (result,x,px) where result is the number of survived electrons,
        # x and px are the information of the electrons after the 1000 rounds
        #(in case mode="aproximated" the information is from the round before the NLK has been used)
        
        return result

        
        
    def propagation_single_round(self, x, px, kicker_strength = 0.0, deterministic = False, first_round = False):
        """
        Input: x,px; NumpyArray of x and px information
            
        """
        assert len(x) == len(px)
        assert (kicker_strength >= -1) and (kicker_strength <= 1)

        
        self.x = x
        self.px = px
        self.kicker_strength = kicker_strength
        
        #create noises dependent on inputs deterministic and first round
        
        if deterministic:
            noise_x, noise_px, noise_NLK = 0.0, 0.0, 0.0
        else: 
            noise_NLK = self.given_noise_NLK
            #in the first round extra noise is added, as the first round is noisier
            if first_round:
                noise_x, noise_px = self.given_noise_first_round, self.given_noise_first_round
            else:
                noise_x, noise_px = self.given_noise_x, self.given_noise_px
        
        
        result = self.Propagator.propagation_single_round(x, px, kicker_strength, 
                                                          noise_x, noise_px, noise_NLK)
        #result has form (result,x,px) where result is the number of survived electrons,
        # x and px are the information of the electrons after the 1000 rounds
        #(in case mode="aproximated" the information is from the round before the NLK has been used)
        
        return result
    



__all__=["Particle_Propagator"]