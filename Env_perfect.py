# Import routines

import numpy as np
import math
import random
from numpy import array

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space =[(0,0)]+[(p,q) for p in range(0,m) for q in range(0,m) if p != q]
        
        self.state_space = [[i, j, k] for i in range(m) for j in range(t) for k in range(d)]
        with open('daemons123.txt', 'w') as fp:
             fp.write('\n'.join('{} {} {}'.format(x[0],x[1],x[2]) for x in self.state_space))
        self.state_init = self.state_space[random.randint(0,len(self.state_space)-1)]
        self.number_days=0
        self.count=0
        self.Terminalstate=False
       

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encoder(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint:    The vector is of size m + t + d."""
        #print(state)
        '''zeros=np.zeros((36),dtype=int)
        zeros[state[0]]=1
        zeros[m+state[1]]=1
        zeros[m+t+state[2]]=1
        print(state,zeros)
        zeros.resize((1,36),refcheck=False)

        #state_encod = array(state)
        #state_encod.resize((1,m + t + d),refcheck=False)'''
        #print(state)
        location = [0 for i in range(5)]
        location[state[0]]=1
        hours = [0 for i in range(24)]
        if state[1]>23:
            print("\n \n")
        hours[state[1]]=1
        day = [0 for i in range(7)]
        day[state[2]]=1
        encoded_state = location+hours+day
        encoded_state = np.array(encoded_state).reshape(1,36)
        #encoded_state = encoded_state.reshape(1,36)
        #print("shape",encoded_state.shape)
        return encoded_state


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)

        if location ==1:
            requests = np.random.poisson(12)
           
        if location ==2:
            requests = np.random.poisson(4)
            
        if location ==3:
            requests = np.random.poisson(7)
            
        if location ==5:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append([0,0])

        return possible_actions_index,actions   

    def preprocesstime_24hrs(self,time,day_week):      
        while(time>23):
            time = time-24
            day_week =day_week+1
        
        if day_week>=7:
            #self.number_days = self.number_days+(day_week-self.number_days)
            self.number_days = self.number_days+day_week
            day_week=0    
        if  self.number_days==28:
            self.count=self.count+1   
        if self.number_days+self.count==30:
            #print("\n",self.number_days+self.count,"\n")  
            self.number_days = 0  
            self.count=0 
            self.Terminalstate=True
        #self.day_count= self.day_count+1
        # if self.day_count>= 30:
        #     print("\n rei chusuko \n")   
        #if day_week==7:
         #     day_week=0
        return time,day_week     
           
    '''scenarios
    1.driver is at pickup location
    2.driver has to travel to pickup location
    3.driver rejects the request
    
    '''
    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        #𝑇𝑖𝑚𝑒 − 𝑚𝑎𝑡𝑟𝑖𝑥[𝑠𝑡𝑎𝑟𝑡 − 𝑙𝑜𝑐][𝑒𝑛𝑑 − 𝑙𝑜𝑐][ℎ𝑜𝑢𝑟 − 𝑜𝑓 − 𝑡ℎ𝑒 − 𝑑𝑎𝑦] [𝑑𝑎𝑦 − 𝑜𝑓 − 𝑡ℎ𝑒 − 𝑤𝑒𝑒𝑘]
        #time_takentocompleteride = Time_matrix[action[0]][action[1]][state[1]][state[2]]
        
        if action ==(0,0):
            
             reward = R*(0)-C*(1) 
                
        if (action[0]==state[0]):
            state[1],state[2] = self.preprocesstime_24hrs(state[1],state[2]) 
            time_takentocompleteride = Time_matrix[action[0]][action[1]][state[1]][state[2]]
            reward = R*(time_takentocompleteride)-C*(time_takentocompleteride+0)
            
        elif(state[0]!=action[0]):
            state[1],state[2] = self.preprocesstime_24hrs(state[1],state[2]) 
            transist_time= Time_matrix[state[0]][action[0]][state[1]][state[2]]#time taken from current state state[0] to pickup location action[0]
            state[0] = action[0]           
            state[1] = int(transist_time)+state[1]
            '''if (transist_time+state[1])<24:
                state[1]=transist_time+state[1]
            elif (transist_time+state[1])>24:
                
                time_extra_day = transist_time+state[1]
               
                while(time_extra_day>24):
                     time_extra_day = time_extra_day-24
                     #day =day+1
            state[1] = time_extra_day+state[1]
            state[2]=state[2]+day'''
            state[1],state[2] = self.preprocesstime_24hrs(state[1],state[2]) 
            time_takentocompleteride = Time_matrix[action[0]][action[1]][int(state[1])][state[2]]
            reward = R*(time_takentocompleteride)-C*(time_takentocompleteride+transist_time)
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        if action==(0,0):
            rejectingtime = 1
            state[1],state[2] = self.preprocesstime_24hrs(state[1],state[2]) 
            next_state = [state[0],state[1]+rejectingtime,state[2]]
            next_state[1],next_state[2] = self.preprocesstime_24hrs(next_state[1],next_state[2])
            #print("next_state",next_state[1])
            if next_state[1]>23:
                print("\n \n")

        if (action[0]==state[0]):
             state[1],state[2] = self.preprocesstime_24hrs(state[1],state[2])   
             ridetime = Time_matrix[action[0]][action[1]][state[1]][state[2]]
             #next_state = [action[1],state[1]+ridetime,state[2]+extraday]  #have to check if the transistion time can go beyond a day 
             state[1],state[2] = self.preprocesstime_24hrs(state[1],state[2]) 
             next_state = [action[1],state[1]+int(ridetime),state[2]]  
             next_state[1],next_state[2] = self.preprocesstime_24hrs(next_state[1],next_state[2]) 
             #print("next_state",next_state[1])
             if next_state[1]>23:
                print("\n \n")
        elif (state[0]!=action[0]):
             #𝑇𝑖𝑚𝑒 − 𝑚𝑎𝑡𝑟𝑖𝑥[𝑠𝑡𝑎𝑟𝑡 − 𝑙𝑜𝑐][𝑒𝑛𝑑 − 𝑙𝑜𝑐][ℎ𝑜𝑢𝑟 − 𝑜𝑓 − 𝑡ℎ𝑒 − 𝑑𝑎𝑦] [𝑑𝑎𝑦 − 𝑜𝑓 − 𝑡ℎ𝑒 − 𝑤𝑒𝑒𝑘]
            state[1],state[2] = self.preprocesstime_24hrs(state[1],state[2]) 
            transistiontime = Time_matrix[state[0]][action[0]][state[1]][state[2]]
            #newsstate_first = [action[0],state[1]+transistiontime,state[2]+extradays]#have to check if the transistion is going beyond a day then code for extra days
            
            newsstate_first = [action[0],state[1]+int(transistiontime),state[2]]
            newsstate_first[1],newsstate_first[2] = self.preprocesstime_24hrs(newsstate_first[1],newsstate_first[2]) 
            ridetime= Time_matrix[action[0]][action[1]][newsstate_first[1]][newsstate_first[2]]
            #newsstate_second = [action[1],newsstate_first[1]+ridetime,newsstate_first[2]+extradays]#have to check if the transistion is going beyond a day then code for extra days
            newsstate_second = [action[1],newsstate_first[1]+int(ridetime),newsstate_first[2]]
            newsstate_second[1],newsstate_second[2] = self.preprocesstime_24hrs(newsstate_second[1],newsstate_second[2]) 
            next_state = newsstate_second
            #print("next_state",next_state[1])
            if next_state[1]>23:
                print("\n \n")
        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
