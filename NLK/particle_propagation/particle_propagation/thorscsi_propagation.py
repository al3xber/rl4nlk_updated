from thor_scsi.factory import accelerator_from_config
from thor_scsi.pyflame import Config
import thor_scsi.lib as tslib

import numpy as np
import gtpsa
import os
import time

import importlib.resources

from multiprocessing import Process, Queue, Array
from queue import Empty

from interfaces.propagator_interface import PropagatorInterface


t_file = "BII_NLKmode_3d_start.lat"      





"""
------------------------------
   Thor SCSI Initialisation   
------------------------------
"""


#indices for Thor SCSI
x_, px_ = 0, 1
y_, py_ = 2, 3

emittance_start = 70e-9
nv = 6
mo = 1
default_desc = gtpsa.desc(nv, mo)



def create_nlk_interpolation(nlk_name):
    #this function is needed to create the NLK interpolation
    #function was given by Pierre
    
    def compute_mirror_position_plate(ref_pos, mirror_pos, *, y_plane=True):
        assert y_plane
        dy = ref_pos.imag - mirror_pos.imag
        return ref_pos - 2 * dy * 1j

    # fmt: off
    ref_pos1 =  8e-3 +  7e-3j
    ref_pos2 = 17e-3 + 15e-3j
    # fmt: on
    t_current = -7e2

    # fmt: off
    t_current *= 1 - 1 * 0.14 / 2
    ref_pos1  *= 1 - 0.14
    ref_pos2  *= 1 - 0.14

    plate_position1 = 5e-3j
    mirror_pos1 = compute_mirror_position_plate(ref_pos1, plate_position1)

    inner = tslib.aircoil_filament(ref_pos1.real, ref_pos1.imag,  t_current)
    outer = tslib.aircoil_filament(ref_pos2.real, ref_pos2.imag, -t_current)
    mirror = tslib.aircoil_filament(mirror_pos1.real, mirror_pos1.imag, -t_current * 0.14)
    nlkf_intp = tslib.NonLinearKickerInterpolation([inner, outer, mirror])

    c = Config()
    c.setAny("L", 0e0)
    c.setAny("name", nlk_name)
    c.setAny("N", 1)
    nlk = tslib.FieldKick(c)
    nlk.set_field_interpolator(nlkf_intp)
    return nlk, nlkf_intp



def calulate_sigma_px(sigma_x, *, emittance=emittance_start):
    #small function to calculate sigma px given sigma x and the emittance
    assert sigma_x <=emittance_start
    
    sigma_px = np.sqrt(emittance ** 2 - sigma_x ** 2)    #formula for sigma px
    
    return sigma_px


def create_state_space_vector(*, mu_x=0e0, mu_px=0e0, mu_y=0e0, mu_py=0e0, desc=default_desc):
    #function for the creation of a state space vector ps
    #ps holds the x,px information of an electron and is used to propagate electrons in Thor SCSI
    #function given by Pierre
    
    ps = gtpsa.ss_vect_double(0.0)
    ps.set_zero()
    
    ps.x+=mu_x
    ps.px+=mu_px
    ps.y+=mu_y
    ps.py+=mu_py
    ps.copy()
    
    return ps






def particle_propagation(index,acc_info,remaining_info):
    """
    This function has been created to allow the parallel calculation of 1000 electrons at a time.
    Each parallelisation-worker will have its own Thor SCSI simulation.
    The index input varies.
    
    Input:   - index (int) which index is currently worked on
             - acc_info (tuple) contains the accelerator information
             - remaining_info (tuple) contains additional information that is for each task and worker the same
    Output:
             - 4-tuple: (result, x, px, start)
             where: - result (bool) information if electron survived the injection
                    - x/px (float) information of electron after 1000 rounds
                    - start (tuple) start x/px information of the electron
    """
    #collect remaining information
    x_list, px_list, when_activate_NLK, kicker_strength, noise_x, noise_px, noise_NLK, noise_first_round = remaining_info 
    
    #get accelerator information
    acc, calc_config, nlkfk, nlkf_intp = acc_info    
    
    #get x/px value
    x = x_list[index]        
    px = px_list[index]
    
    #set a random seed
    np.random.seed(round(float(str(time.time())[6:]))*(index+1))    
    
    #go through 1000 rounds
    for runde in range(1000):
        #set NLK-strength to 0, if NLK not used
        if runde != when_activate_NLK:    
            nlkf_intp.set_scale(0.0)
        #if round is equal to NLK-activation-round activate the NLK with given strength
        elif runde == when_activate_NLK:   
            noise_NLK_sample = np.random.normal(0,noise_NLK)          #create noise
            nlkf_intp.set_scale(kicker_strength+noise_NLK_sample)     #add noise and set kicker strength
        else:
            raise ValueError("should not end up here")
        
        #create state space vector 
        ps = create_state_space_vector(mu_x=x,mu_px=px)   
        #create noise
        if runde == 0:
            noise_x_sample = np.random.normal(0,noise_first_round)
            noise_px_sample = np.random.normal(0,noise_first_round)
        else:   
            noise_x_sample = np.random.normal(0,noise_x)
            noise_px_sample = np.random.normal(0,noise_px)
        #add noise
        ps.x += noise_x_sample     
        ps.px += noise_px_sample
        #propagate through accelerator
        result = acc.propagate(calc_config, ps)
        assert result==len(acc) 
        #update the x and px values
        n_x = ps.x
        n_px = ps.px
        x = n_x
        px = n_px
        
        #check if it crashed into the septum
        if x>0.015:  
            return False, np.nan, np.nan, (x_list[index],px_list[index])
        #if not continue with the remaining rounds
        elif x<=0.015:
            continue
        #x might be NAN, if settings are extreme, checking for this case
        else:    
            return False, np.nan, np.nan, (x_list[index],px_list[index])  
    #if this point is reached, the 1000 rounds have been successfully survived!
    return True, x, px, (x_list[index],px_list[index])





def run(idx, args, worker_acc_info, remaining_info):
    """
    Function that is used to run the workers
    
    Inputs:  -idx: Worker index
             -args: Job index
    """
    return particle_propagation(args, worker_acc_info, remaining_info)






class Thor_SCSI_Propagator(PropagatorInterface):


    def __init__(self):
        #creating simulation of the accelerator
        with importlib.resources.path('particle_propagation', 'package_files') as data_path:
            default_config_path = data_path / t_file
            #contents = default_config_path.read_text()
            self.acc = accelerator_from_config(default_config_path)
        
        
        
        self.calc_config = tslib.ConfigType()

        #Description of NLK
        self.nlkfk = self.acc.find("KDNL1KR", 0)
        self.nlk_name = self.nlkfk.name
        _, self.nlkf_intp = create_nlk_interpolation(self.nlk_name)
        self.nlkfk.set_field_interpolator(self.nlkf_intp)
        assert(self.nlkfk.name == self.nlk_name)

        
    
    def propagation_single_round(self, x_list, px_list, kicker_strength, 
                                 noise_x, noise_px, noise_NLK):
        """
        Function that propagates electrons for a single round given NLK strength and noise.
        Input:   - x_list (list/np.ndarray) x-information of electrons
                 - px_list (list/np.ndarray) px-information of electrons
                 - kicker_strength (float)   strength of NLK, value in [-1, 1]
                 - noises
        Output:
                 - x_list, px_list information of the electrons after one round
        """
        
        
        
        
        

        #set a random seed
        np.random.seed(round(float(str(time.time())[6:])))    

        noise_NLK_sample = 0.0    
        if not (kicker_strength == 0.0):   #if NLK not activated generate noise sample, 
                                           #else the NLK isn't used so let strength 0  
            noise_NLK_sample = np.random.normal(0,noise_NLK)        #sample NLK noise
        #set kicker strength
        self.nlkf_intp.set_scale(kicker_strength+noise_NLK_sample)       

        #generate noise, we use the same noise for each electron.
        noise_x_sample = np.random.normal(0,noise_x)
        noise_px_sample = np.random.normal(0,noise_px)

        
        #go through each electron
        output_x_list, output_px_list = [], []
        for i in range(len(x_list)):
            #create state space vector
            ps = create_state_space_vector(mu_x=x_list[i],mu_px=px_list[i]) 


            #add noise
            ps.x += noise_x_sample     
            ps.px += noise_px_sample

            #propagate through the accelerator
            result = self.acc.propagate(self.calc_config, ps)
            assert result==len(self.acc) 

            #update x and px information
            output_x_list.append(ps.x)
            output_px_list.append(ps.px)


        return np.array(output_x_list), np.array(output_px_list)





    
    
       
    def propagation_thousand_rounds(self, x_list, px_list, when_activate_NLK, kicker_strength,
                 noise_x = 0.0, noise_px = 0.0, noise_NLK = 0.0, noise_first_round = 0.0):
        """
        Input: x_list,px_list (list/np.ndarray); List of x and px values
               when_activate_NLK (int) in which round to activate the NLK
               kicker_strengh (float in [-1,1]) strength of NLK in activation round
               noises (floats)
        Output:
              - x_list, px_list information of the electrons after one round
              Note!! the arrays will not be correctly ordered.

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
                            
                            idx += 1
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

        return np.sum(results), x_result_arr, px_result_arr
    
    


    
    
    


__all__=["Thor_SCSI_Propagator","calulate_sigma_px"]