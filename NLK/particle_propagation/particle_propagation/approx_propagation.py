import numpy as np
from scipy.stats import norm





class Approximated_Propagator():

    def __init__(self):
        
        #information for kicker usage
        self.round_survived_array = np.load("Zone_array5.npy")[0]
        self.optimal_NLK_strength_array = np.load("Ztwo_array5.npy")[0]
        
        self.x_list = np.linspace(-100.0e-3,90e-3,1280)
        self.px_list = np.linspace(-12e-3,7e-3,640)
        
        self.x_min,self.x_max, self.x_len     = self.x_list[0], self.x_list[-1], len(self.x_list)
        self.px_min, self.px_max, self.px_len = self.px_list[0], self.px_list[-1], len(self.px_list)
        
        
        
        #information for round to round behaviour
        self.roundX = np.load("ZroundX.npy").T
        self.roundPX = np.load("ZroundPX.npy").T
                
        self.x_list2 = np.linspace(-45e-3,24.5e-3,1280)
        self.px_list2 = np.linspace(-0.003,0.003,1280)
        
        self.x_min2,self.x_max2, self.x_len2     = self.x_list2[0], self.x_list2[-1], len(self.x_list2)
        self.px_min2, self.px_max2, self.px_len2 = self.px_list2[0], self.px_list2[-1], len(self.px_list2)
     
    
    def _single_round_without_kick(self,x,px,runde,noise_x,noise_px):

        #add noise 
        x += noise_x     
        px += noise_px


        #get indices within round to round behaviour files
        idx_x  = np.floor(self.x_len2*((x-self.x_min2)/(self.x_max2-self.x_min2))).astype("int")
        idx_px = np.floor(self.px_len2*((px-self.px_min2)/(self.px_max2-self.px_min2))).astype("int")

        #remove points that are out of the valid area
        valid_idx = (idx_x >= 0)*(idx_x < self.x_len2 - 1)*(idx_px >= 0)*(idx_px < self.px_len2 - 1)

        x = x[valid_idx]
        px = px[valid_idx]
        idx_x = idx_x[valid_idx]
        idx_px = idx_px[valid_idx]


        #remove points that contain nan
        where_nan = np.vstack([self.roundX[idx_px,  idx_x],
                                     self.roundX[idx_px,  idx_x+1],
                                     self.roundX[idx_px+1,idx_x],
                                     self.roundX[idx_px+1,idx_x+1],
                                     self.roundPX[idx_px,  idx_x],
                                     self.roundPX[idx_px,  idx_x+1],
                                     self.roundPX[idx_px+1,idx_x],
                                     self.roundPX[idx_px+1,idx_x+1]]).T
        where_nan = np.sum(where_nan, axis=1) 
        valid_idx = (1 - np.isnan(where_nan)).astype(bool)   

        x = x[valid_idx]
        px = px[valid_idx]
        idx_x = idx_x[valid_idx]
        idx_px = idx_px[valid_idx]
        points = np.vstack([x, px])


        #calculate for each point the distance to the individual corner points
        distance_matrix = np.vstack([np.sum(np.abs(points-
                                             np.vstack([self.x_list2[idx_x],self.px_list2[idx_px]])),axis=0),
                                     np.sum(np.abs(points-
                                             np.vstack([self.x_list2[idx_x],self.px_list2[idx_px+1]])),axis=0),
                                     np.sum(np.abs(points-
                                             np.vstack([self.x_list2[idx_x+1],self.px_list2[idx_px]])),axis=0),
                                     np.sum(np.abs(points-
                                             np.vstack([self.x_list2[idx_x+1],self.px_list2[idx_px+1]])),axis=0)]).T

        #normalize
        distance_matrix = distance_matrix/np.sum(distance_matrix,axis=1)[:,None]
        #points with smaller distance are more important
        distance_matrix = 1-distance_matrix
        distance_matrix = distance_matrix/np.sum(distance_matrix,axis=1)[:,None]

        assert np.isclose(np.sum(distance_matrix[0]),1.0)
        assert distance_matrix.shape == (len(x),4)

        predicted_x = np.vstack([self.roundX[idx_px,  idx_x],
                                     self.roundX[idx_px+1,  idx_x],
                                     self.roundX[idx_px,idx_x+1],
                                     self.roundX[idx_px+1,idx_x+1]]).T
        predicted_px = np.vstack([self.roundPX[idx_px,  idx_x],
                                     self.roundPX[idx_px+1,  idx_x],
                                     self.roundPX[idx_px,idx_x+1],
                                     self.roundPX[idx_px+1,idx_x+1]]).T

        assert distance_matrix.shape == predicted_x.shape

        predicted_x = np.sum(distance_matrix * predicted_x, axis = 1)
        predicted_px = np.sum(distance_matrix * predicted_px, axis = 1)

        #stack points
        points = np.vstack([predicted_x, predicted_px])

        #remove points that are outside of the septum
        points = points[:, points[0,:] < 0.015]

        x = points[0,:]
        px = points[1,:]


        return x,px


    
    def propagation_single_round(x_list, px_list, kicker_strength, 
                             noise_x=0, noise_px=0, noise_NLK=0):
        """
        Function that propagates electrons for a single round given NLK strength and noise.
        Note that in this mode we can only propagate electrons for a single round, if the NLK is not used!
        Input:   - x_list (list/np.ndarray) x-information of electrons
                 - px_list (list/np.ndarray) px-information of electrons
                 - kicker_strength (float)   strength of NLK, value in [-1, 1]
                 - noises (floats)
        Output:
                 - x_list, px_list information of the electrons after one round
        """
        if kicker_strength != 0.0:
            raise Exception("For the approximated mode, only the propagation of a single \
round can be done if the NLK is not activated.")
            
        return _single_round_without_kick(self,x,px,runde, noise_x=noise_x, noise_px=noise_x)
 

        
    
    def propagation_1000_rounds(self, x, px, when_activate_NLK, kicker_strength,
                 noise_x = 0.0, noise_px = 0.0, noise_NLK = 0.0, noise_first_round = 0.0):
        
        
        #until the NLK activation propagate the electrons without kick through the accelerator   
        for runde in range(when_activate_NLK):
            #generate noise
            if runde == 0:
                noise_x_sample = np.random.normal(0, noise_first_round)
                noise_px_sample = np.random.normal(0, noise_first_round)
            else:   
                noise_x_sample = np.random.normal(0, noise_x)
                noise_px_sample = np.random.normal(0, noise_px)

            x,px = self._single_round_without_kick(self,x,px,runde,noise_x= noise_x_sample, noise_px=noise_px_sample)
        
            
        #NLK is now activated!
        
        #generate noise
        if when_activate_NLK == 0:
            noise_x_sample = np.random.normal(0,noise_first_round)
            noise_px_sample = np.random.normal(0,noise_first_round)
        else:   
            noise_x_sample = np.random.normal(0,noise_x)
            noise_px_sample = np.random.normal(0,noise_px)

        #add noise 
        x += noise_x_sample     
        px += noise_px_sample


        
        
        #get indices within NLK usage files
        idx_x  = np.floor(self.x_len*((self.x-self.x_min)/(self.x_max-self.x_min))).astype("int")
        idx_px = np.floor(self.px_len*((self.px-self.px_min)/(self.px_max-self.px_min))).astype("int")

        #remove points that are out of the valid area
        valid_idx = (idx_x >= 0)*(idx_x < self.x_len - 1)*(idx_px >= 0)*(idx_px < self.px_len - 1)  

        self.x = self.x[valid_idx]
        self.px = self.px[valid_idx]
        idx_x = idx_x[valid_idx]
        idx_px = idx_px[valid_idx]

        #check if each electron has a neighbor that can be successfully injected
        rounds_survived = np.vstack([self.round_survived_array[idx_px,  idx_x],
                                     self.round_survived_array[idx_px,  idx_x+1],
                                     self.round_survived_array[idx_px+1,idx_x],
                                     self.round_survived_array[idx_px+1,idx_x+1]]).T
        rounds_survived = np.sum(rounds_survived == 1000,axis=1) 
        valid_idx = rounds_survived > 1  
        
        #remove points where no neighbor point has survived 1000 rounds
        x_mean = np.mean(self.x)
        px_mean = np.mean(self.px)

        self.x = self.x[valid_idx]
        self.px = self.px[valid_idx]
        idx_x = idx_x[valid_idx]
        idx_px = idx_px[valid_idx]

        self.points = np.vstack([self.x,self.px])

        #optimal kicker strength calculation
        #we use interpolation
        assert self.points.shape == np.vstack([self.x_list[idx_x],self.px_list[idx_px]]).shape
        #calulate distance to each neighbour in the grid
        distance_matrix = np.vstack([np.sum((self.points-
                                             np.vstack([self.x_list[idx_x],self.px_list[idx_px]]))**2,axis=0),
                                     np.sum((self.points-
                                             np.vstack([self.x_list[idx_x],self.px_list[idx_px+1]]))**2,axis=0),
                                     np.sum((self.points-
                                             np.vstack([self.x_list[idx_x+1],self.px_list[idx_px]]))**2,axis=0),
                                     np.sum((self.points-
                                             np.vstack([self.x_list[idx_x+1],self.px_list[idx_px+1]]))**2,axis=0)]).T

        #normalize
        distance_matrix = distance_matrix/np.sum(distance_matrix,axis=1)[:,None]
        #points with smaller distance are more important
        distance_matrix = 1-distance_matrix
        #normalize
        distance_matrix = distance_matrix/np.sum(distance_matrix,axis=1)[:,None]

        assert np.isclose(np.sum(distance_matrix[0]),1.0)
        assert distance_matrix.shape == (len(self.x),4)

        #get for each electron the optimal Kicker strength
        optimal_NLK_strength_matrix = np.vstack([self.optimal_NLK_strength_array[idx_px,  idx_x],
                                     self.optimal_NLK_strength_array[idx_px+1,  idx_x],
                                     self.optimal_NLK_strength_array[idx_px,idx_x+1],
                                     self.optimal_NLK_strength_array[idx_px+1,idx_x+1]]).T   

        assert distance_matrix.shape == optimal_NLK_strength_matrix.shape
        
        #calculate optimal kicker strength
        optimal_kicker_strength = np.sum(distance_matrix * optimal_NLK_strength_matrix,axis = 1)  

        #create noise
        noise_NLK_sample = np.random.normal(0,self.noise_NLK)  

        #calculate reward and add noise to kicker strength
        reward = ((self.points.shape[1])/1000)*(985/1000)*np.sum(np.exp(-(14.5*(kicker_strength+noise_NLK_sample-optimal_kicker_strength))**4)) / self.reward_normalization

        return reward, x, px, True 



        
        
        
    
    


__all__=["Approximated_Propagator"]