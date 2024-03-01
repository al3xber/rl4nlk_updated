import numpy as np
from scipy.stats import norm






class Thor_SCSI_Propagator():


    def __init__(self):
        #information for round to round behaviour
        self.roundX = np.load("ZroundX.npy").T
        self.roundPX = np.load("ZroundPX.npy").T
                
        self.x_list2 = np.linspace(-45e-3,24.5e-3,1280)
        self.px_list2 = np.linspace(-0.003,0.003,1280)
        
        self.x_min2,self.x_max2, self.x_len2     = self.x_list2[0], self.x_list2[-1], len(self.x_list2)
        self.px_min2, self.px_max2, self.px_len2 = self.px_list2[0], self.px_list2[-1], len(self.px_list2)


        pass
        
        
        
    def step(self, action):
        activation_round = (action[0]+1)*5 # -> (-1,-.8)->0, (-.8,-.6)->1, (-.6,-.4)->2, (-.4,-.2)->3, (-.2,0)->4
        probability_for_next = activation_round-np.floor(activation_round)
        activation_round = int(np.floor(activation_round))+np.random.binomial(1, probability_for_next)

        terminated = False

        for runde in range(activation_round):
            out_dict, reward, terminated, truncated, info = self._step(0.0)
            if terminated:
                return out_dict, reward, terminated, truncated, info
        
        out_dict, reward, terminated, truncated, info = self._step(action[1])
        terminated = True   #in case action strength is 0.0
        
        return out_dict, reward, terminated, truncated, info
           

    
    def propagation_1000_rounds(self, x_list, px_list, when_activate_NLK, kicker_strength,
                 noise_x = 0.0, noise_px = 0.0, noise_NLK = 0.0, noise_first_round = 0.0):
        
        x = x_list
        px = px_list


        #until the NLK activation look at the round to round behaviour
        for runde in range(when_activate_NLK):
            
            #generate noise
            if runde == 0:
                noise_x_sample = np.random.normal(0,noise_first_round)
                noise_px_sample = np.random.normal(0,noise_first_round)
            else:   
                noise_x_sample = np.random.normal(0,noise_x)
                noise_px_sample = np.random.normal(0,noise_px)

            #add noise 

            x += noise_x_sample     
            px += noise_px_sample


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
            x

            
        #NLK is now activated!
        #get indices within NLK usage files
        idx_x  = np.floor(self.x_len*((self.x-self.x_min)/(self.x_max-self.x_min))).astype("int")
        idx_px = np.floor(self.px_len*((self.px-self.px_min)/(self.px_max-self.px_min))).astype("int")

        #remove points that are out of the valid area
        valid_idx = (idx_x >= 0)*(idx_x < self.x_len - 1)*(idx_px >= 0)*(idx_px < self.px_len - 1)  

        self.x = self.x[valid_idx]
        self.px = self.px[valid_idx]
        idx_x = idx_x[valid_idx]
        idx_px = idx_px[valid_idx]

        #check if there are still points left. If no electrons are left, the points are out of the valid area
        check = self.check_if_empty()
        if check != False: 
            return check

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

        #check if there are still points left. If no electrons are left, NLK has been activated outside of Kicker area
        if self.check_if_empty() != False:                 
            reward = 0
            return self.check_if_empty(reward = reward, xpx_mean = np.array([x_mean*self.x_normalization, 
                                                                             px_mean*self.px_normalization]))

        #optimal kicker strength calculation
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
        distance_matrix = distance_matrix/np.sum(distance_matrix,axis=1)[:,None]

        assert np.isclose(np.sum(distance_matrix[0]),1.0)
        assert distance_matrix.shape == (len(self.x),4)

        #get for each electron the optimal Kicker strength
        optimal_NLK_strength_matrix = np.vstack([self.optimal_NLK_strength_array[idx_px,  idx_x],
                                     self.optimal_NLK_strength_array[idx_px+1,  idx_x],
                                     self.optimal_NLK_strength_array[idx_px,idx_x+1],
                                     self.optimal_NLK_strength_array[idx_px+1,idx_x+1]]).T   #transposed wichtig!

        assert distance_matrix.shape == optimal_NLK_strength_matrix.shape

        optimal_kicker_strength = np.sum(distance_matrix * optimal_NLK_strength_matrix,axis = 1)  

        #add noise
        noise_NLK_sample = np.random.normal(0,self.noise_NLK)  

        #calculate reward
        reward = ((self.points.shape[1])/1000)*(985/1000)*np.sum(np.exp(-(14.5*(kicker_strength+noise_NLK_sample-optimal_kicker_strength))**4)) / self.reward_normalization

        #normalize points
        self.points[0,:] *= self.x_normalization
        self.points[1,:] *= self.px_normalization

        #if this point is reached, some electrons can be successfully kicked
        #therefore we return the encoded points, the reward and terminated = True.
        xpx_mean, image = encoder(self.points, self.x_normalization, self.px_normalization)

        out_dict={"x,px mean": xpx_mean,
                  "image": image}
        truncated = False
        terminated = True
        info = {}
        return out_dict, reward, terminated, truncated, info



        
        
        
    
    def propagation_single_round(x_list, px_list, kicker_strength, 
                                 noise_x=0, noise_px=0, noise_NLK=0):
        """
        Function that propagates electrons for a single round given NLK strength and noise.
        Input:   - x_list (list/np.ndarray) x-information of electrons
                 - px_list (list/np.ndarray) px-information of electrons
                 - kicker_strength (float)   strength of NLK, value in [-1, 1]
                 - noises (floats)
        Output:
                 - x_list, px_list information of the electrons after one round
        """

        x_list, px_list, when_activate_NLK, kicker_strength, noise_x, noise_px, noise_NLK, noise_first_round = remaining_info 

        #creating simulation of the accelerator
        acc = accelerator_from_config(t_file)
        calc_config = tslib.ConfigType()

        #Description of NLK
        nlkfk = acc.find("KDNL1KR", 0)
        nlk_name = nlkfk.name
        _, nlkf_intp = create_nlk_interpolation(nlk_name)
        nlkfk.set_field_interpolator(nlkf_intp)
        assert(nlkfk.name == nlk_name)

        #set a random seed
        np.random.seed(round(float(str(time.time())[6:]))*(index+1))    

        #set kicker strength
        noise_NLK_sample = np.random.normal(0,noise_NLK)            #calculate npose noise
        nlkf_intp.set_scale(kicker_strength+noise_NLK_sample)       #set kicker strength    

        #go through each electron
        for i in range(len(x_list)):
            #create state space vector
            ps = create_state_space_vector(mu_x=x_list[i],mu_px=px_list[i]) 

            #generate noise
            noise_x_sample = np.random.normal(0,noise_x)
            noise_px_sample = np.random.normal(0,noise_px)

            #add noise
            ps.x += noise_x_sample     
            ps.px += noise_px_sample

            #propagate through the accelerator
            result = acc.propagate(calc_config, ps)
            assert result==len(acc) 

            #update x and px information
            x_list[i] = ps.x
            px_list[i] = ps.px


        return x_list, px_list





    
    
       
    def propagation_1000_rounds(self, x_list, px_list, when_activate_NLK, kicker_strength,
                 noise_x = 0.0, noise_px = 0.0, noise_NLK = 0.0, noise_first_round = 0.0):
        """
        Input: x_list,px_list (list/np.ndarray); List of x and px values
               when_activate_NLK (int) in which round to activate the NLK
               kicker_strengh (float in [-1,1]) strength of NLK in activation round
               noises (floats)
        """
        assert len(x_list)==len(px_list)

        #save all information in the class, s.t. the other functions can use them
        self.x_list = x_list
        self.px_list = px_list
        self.kicker_strength = kicker_strength
        self.when_activate_NLK = when_activate_NLK

        self.noise_x = noise_x
        self.noise_px = noise_px
        self.noise_NLK = noise_NLK
        self.noise_first_round = noise_first_round
        
        #the work is parallelized and split up
        a=time.time()
        result = self._run_jobs([('job', i) for i in range(0,len(x_list),25)], workers=20)   
        print("Needed time:", time.time()-a)
        
        return result


    def _run_jobs(self,jobs, workers=1):
        """
        Function that runs the different jobs (indices of x_list)
        """        
        
        q = Queue()  #Multiprocessing Queue to remember which jobs need to be done

        def worker(idx,q,queue_worker,queue_finish,remaining_info):
            """
            Each worker gets their individuall accelerator
            """
            
            acc = accelerator_from_config(t_file)
            calc_config = tslib.ConfigType()

            #Description of the NLK
            nlkfk = acc.find("KDNL1KR", 0)
            nlk_name = nlkfk.name
            _, nlkf_intp = create_nlk_interpolation(nlk_name)
            nlkfk.set_field_interpolator(nlkf_intp)
            assert(nlkfk.name == nlk_name)

            #compress all accelerator information
            worker_acc_info = (acc,calc_config,nlkfk,nlkf_intp)   


            try:
                while True:
                    
                    args = q.get(timeout=1)    #has form ("Job", job_index)

                    for running_idx in range(25): #let each worker run multiple runs at once, 
                                                  #to decrease time consumption

                        if args[1]+running_idx < len(remaining_info[0]):
                            result, x_process, px_process, start = run(idx,args[1]+running_idx, worker_acc_info,remaining_info)

                            queue_worker.put((idx,result, np.array(x_process),
                                              np.array(px_process),start),timeout = 15)#send information to output pipe
                            
            except Empty:    #if q is empty
                queue_finish.put((idx),timeout = 1)
                queue_worker.close()
                return 
            
            print(f"worker {idx:02d}, WHY END UP HERE??-------------------------------")
        
        def information_extractor_worker(queue_finish, worker_queue_list, x_result_array,
                                         px_result_array, result_list, start_list, len_x_list):
            finished_pipes_counter = 0            
            idx = 0         #running index

            while finished_pipes_counter < len(worker_queue_list) or idx<len_x_list: 
                #Check if workers are still working
                while finished_pipes_counter < len(worker_queue_list):
                    try:                                              # while not empty, get all finished workers
                        queue_finish.get(block=True, timeout=.01)     #if not empty -> worker finished
                        finished_pipes_counter+=1
                    except Empty:
                        break
                
                #empty worker queues
                for x in range(len(worker_queue_list)):
                    while True:
                        try:
                            _,result,x_process,px_process,start = worker_queue_list[x].get(block=True, timeout=.1)
                            x_process, px_process = [x_process], [px_process]
                            
                            
                            result_list[idx] = float(result)
                            start_list[idx*2:(idx+1)*2] = start      #start is 2 dimensional

                            x_result_array[idx : idx + 1] = x_process 
                            px_result_array[idx : idx + 1]=px_process
                            
                            idx+=1
                        except Empty:
                            break
            
            queue_finish.close()
            return 
        
        
        
        for job in jobs:
            q.put(job)

        worker_queue_list=[]    #list of input/output pipes
        processes = []          #list of all workers
        
        queue_finish = Queue()

        for i in range(0, workers):
            queue_worker = Queue()
            worker_queue_list.append(queue_worker)
            
            remaining_info = (self.x_list, self.px_list, self.when_activate_NLK, self.kicker_strength, 
                              self.noise_x, self.noise_px, self.noise_NLK, self.noise_first_round)
            p = Process(target=worker, args=[i,q,queue_worker,queue_finish,remaining_info])
            p.daemon = True
            p.start()
            processes.append(p)
        
        #Create variables
        unshared_arr = np.zeros(len(self.x_list))
        x_result_array = Array('d', unshared_arr)
        
        unshared_arr2 = np.zeros(len(self.x_list))
        result_list = Array('d', unshared_arr2)
        
        unshared_arr3 = np.zeros(len(self.x_list)*2)
        start_list = Array('d', unshared_arr3)
        
        unshared_arr4 = np.zeros(len(self.x_list))
        px_result_array = Array('d', unshared_arr4)
        
        p = Process(target=information_extractor_worker, args=[queue_finish, worker_queue_list, x_result_array,
                                                               px_result_array, result_list,
                                                               start_list, len(self.x_list)])
        p.daemon = True
        p.start()
        p.join()
        


        #Loading all variables
        x_result_arr = np.frombuffer(x_result_array.get_obj())
        x_result_arr = x_result_arr.reshape((len(self.x_list),1))
        
        results = np.frombuffer(result_list.get_obj())
        
        starts = np.frombuffer(start_list.get_obj())
        starts = starts.reshape((len(self.x_list),2))
        
        px_result_arr = np.frombuffer(px_result_array.get_obj())
        px_result_arr = px_result_arr.reshape((len(self.x_list),1))

        return results,x_result_array,px_result_array,starts
    
    


    
    
    


__all__=["Thor_SCSI_Propagator","calulate_sigma_px"]