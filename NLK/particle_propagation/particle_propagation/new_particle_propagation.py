import numpy as np
import os



class Particle_Propagator():


    def __init__(self):
        pass
            
       
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


        
        
        result = True
        
        
        return result


    



__all__=["Particle_Propagator","calulate_sigma_px","emmitance_propagation","single_particle"]