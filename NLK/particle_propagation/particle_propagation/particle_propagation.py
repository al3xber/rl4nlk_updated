import numpy as np
import os

from particle_propagation.thorscsi_propagation import Thor_SCSI_Propagator
from particle_propagation.approx_propagation import Approximated_Propagator

class Particle_Propagator():


    def __init__(self,mode = "approximated"):
        #for the given mode set the approximator 
        if mode == "approximated":
            self.Propagator = Approximated_Propagator()
        elif mode == "thorscsi": 
            self.Propagator = Thor_SCSI_Propagator()
        else:
            raise Exception("Only modes 'approximated' and 'thorscsi' are allowed!")
            
        self.mode = mode
       
    def propagate_1000rounds(self, x_list, px_list, when_activate_NLK=1, kicker_strength=1.0,
                             deterministic = False):
        """
        Input: x_list,px_list; List/NumpyArray or Float value of x and px information
               y_rounds_to_save; List which px rounds so save
        """
        if type(x_list) == float and type(px_list) == float:
            x_list = [x_list]
            px_list = [px_list]
        assert type(x_list) == type(px_list)
        assert len(x_list) == len(px_list)
        assert (kicker_strength >= -1) and (kicker_strength <= 1)

        
        self.x_list = x_list
        self.px_list = px_list
        self.kicker_strength = kicker_strength
        self.when_activate_NLK = when_activate_NLK

        if deterministic:
            self.noise_first_round, self.noise_x, self.noise_px, self.noise_NLK = 0, 0, 0, 0
        else:
            self.noise_first_round = 65e-6
            self.noise_x = 6.5e-6
            self.noise_px = .8*6.5e-6
            self.noise_NLK = 0.0125


        
        result = self.Propagator.propagation_1000_rounds(x_list, px_list, when_activate_NLK, kicker_strength,
                 noise_x = self.noise_x, noise_px = self.noise_px, noise_NLK = self.noise_px, 
                 noise_first_round = self.noise_first_round)
        #result has form (result,x,px) where result is the number of survived electrons,
        # x and px are the information of the electrons after the 1000 rounds
        #(in case mode="aproximated" the information is from the round before the NLK has been used)
        
        return result

        
        
    def propagate_single_round(self, x_list, px_list, kicker_strength=1.0, noise_x=0, noise_px=0, noise_NLK=0):
        """
        Input: x_list,px_list; List/NumpyArray or Float value of x and px information
               y_rounds_to_save; List which px rounds so save
        """
        if type(x_list) == float and type(px_list) == float:
            x_list = [x_list]
            px_list = [px_list]
        assert type(x_list) == type(px_list)
        assert len(x_list) == len(px_list)
        assert (kicker_strength >= -1) and (kicker_strength <= 1)

        
        self.x_list = x_list
        self.px_list = px_list
        self.kicker_strength = kicker_strength
        self.when_activate_NLK = when_activate_NLK
        
        result = self.Propagator.propagation_single_round(x_list, px_list, kicker_strength, 
                                 noise_x=noise_x, noise_px=noise_px, noise_NLK=noise_NLK)
        #result has form (result,x,px) where result is the number of survived electrons,
        # x and px are the information of the electrons after the 1000 rounds
        #(in case mode="aproximated" the information is from the round before the NLK has been used)
        
        return result
    



__all__=["Particle_Propagator"]