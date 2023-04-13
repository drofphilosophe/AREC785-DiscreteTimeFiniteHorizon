import numpy as np
import pandas as pd

class finite_horizon :
    ################
    #Static class members
    ################
    #Payoff penalty for a constraint violation
    CONSTR_VIOLATION = -1000

    ######################
    #State space properties
    ######################
    #Minimum value of the state.
    #It's important to have constraint-violating values of the state
    #So we can include penalties for constraint violations in the
    #payoff function
    STATE_MIN = -1
    #Maximum value of the state
    STATE_MAX = 100
    #Grid point size
    STATE_GRID_SIZE = 0.1

    ################
    # Control Space Properties
    ################
    CONTROL_MIN = 0
    #Even though it's not defined in the problem, I exclude
    #values that I know wouldn't be selected from the control space
    #This reduces the space of options we need to consider when
    #computing the value function
    CONTROL_MAX = 25
    CONTROL_GRID_SIZE = 0.1
    
    #Number of time periods
    T = 10

    #Discount factor
    BETA = 0.9

    def __init__(self) :
        #An array of discretized state values        
        self.s_grid = np.linspace(self.STATE_MIN,
                                  self.STATE_MAX,
                                  round( (self.STATE_MAX-self.STATE_MIN)/self.STATE_GRID_SIZE ) + 1
                                  )
        
        #Count the number of entries in the state grid
        self.NS = len(self.s_grid)
    
        #An array of discretized control values
        self.c_grid = np.linspace(self.CONTROL_MIN,
                                  self.CONTROL_MAX,
                                  round( (self.CONTROL_MAX-self.CONTROL_MIN)/self.CONTROL_GRID_SIZE ) + 1
                                  )
        
        #Count the number of entries in the control grid
        self.NC = len(self.c_grid)

        #Create a NC x NS array of state values
        self.sval_mat = np.repeat(
            np.reshape(self.s_grid,(self.NS,1)),
            self.NC,axis=1)
        
        #Create a NC x NS array of control values
        self.cval_mat = np.repeat(
            np.reshape(self.c_grid,(1,self.NC)),
            self.NS,axis=0)
        
        #Define the value function
        #It is a NS x T array and will contain the
        #continuation value of ending period t with state s
        self.VF = np.empty((self.NS,self.T+1))

        #Define the policy function array. It is an NS x T
        #Array that contains the index of the optimal control
        #In period t given starting the period with control index
        # CIDX
        self.PF = np.empty((self.NS,self.T),dtype=np.int)

    #Convert a discretized state index to a value
    def sidx_2_sval(self,sidx) :
        return self.s_grid[sidx]

    #Convert a state value to its closest discretized state index
    def sval_2_sidx(self,sval) :
        #Find the index of the first grid element larger than sval
        lidx = np.searchsorted(self.s_grid,sval,side='left')
        #Find the index of the last grid element smaller than sval
        ridx = np.clip(lidx-1,0,self.NS-1)
        #Return the index that nets the closer value
        return np.where( np.abs(self.s_grid[lidx] - sval) < np.abs(self.s_grid[ridx]-sval),lidx,ridx)

    
    #Convert a discretized control index to a value
    def cidx_2_cval(self,cidx) :
        return self.c_grid[cidx]

    #Convert a state value to its closest discretized state index
    def cval_2_cidx(self,cval) :
        #Find the index of the first grid element larger than cval
        lidx = np.searchsorted(self.c_grid,cval,side='left')
        #Find the index of the last grid element smaller than cval
        ridx = np.clip(lidx-1,0,self.NC-1)
        #Return the index that nets the closer value
        return np.where( np.abs(self.c_grid[lidx] - cval) < np.abs(self.c_grid[ridx]-cval),lidx,ridx)


    #Compute the value function in period t
    def compute_vf(self,t) :

        #In period T, use the scrap value
        if t == self.T :
            #Set the value function for each control to the scrap value
            self.VF[:,t] = self.scrap_value()
            #No choice is made in thie period, so you do not need to set
            #anything in the policy function
        else :
            #Compute the payoff for each possible state and control
            #It is NC x NS
            p = self.payoff(self.sval_mat,self.cval_mat,t)

            #Compute the new value of the control
            sval_new = self.newstate(self.sval_mat,self.cval_mat,t)

            #Compute the period t+1 state indexes
            sidx_new = self.sval_2_sidx(sval_new)
            
            #Add in the value function for each new value of the control
            p = p + self.BETA*self.VF[sidx_new,t+1]

            #For each possible value of the state, find the control
            #That maximizes p
            self.PF[:,t] = np.argmax(p,axis=1)

            #Update the value function
            self.VF[:,t] = np.max(p,axis=1)

    #Compute the value of the costate for period starting state sval in period t
    def compute_costate(self,sval,t) :
        #Determine the state index
        sidx = self.sval_2_sidx(sval)

        #If we are at the extrema of the allowed values, we can't compute the costate using centered differences
        if sval <= 0 :
            costate = (self.VF[sidx+1,t]-self.VF[sidx,t])/(self.STATE_GRID_SIZE)
        elif sidx == self.NS - 1 :
            costate = (self.VF[sidx,t]-self.VF[sidx-1,t])/(self.STATE_GRID_SIZE)
        else :
            #Compute the value of the costate as the centered difference of present value payoffs
            costate = (self.VF[sidx+1,t]-self.VF[sidx-1,t])/(self.STATE_GRID_SIZE*2)
        return costate
    
        
    #Solve for the optimal controls and value
    #function using backwards induction
    def backwards_induct(self) :
        
        #Work backward from period T to 0
        for t in range(self.T,-1,-1) :
            print("Solving period",t)
            self.compute_vf(t)


    #Simulate forward from time zero and T0 state sval
    def forward_simulate(self,sval,csv_out=None) :

        #Init storage for the results.
        result = []

        sidx = self.sval_2_sidx(sval)
        running_total_payoff = 0

        for t in range(0,self.T) :
            sval = self.sidx_2_sval(sidx)
            cidx = self.PF[sidx,t]
            cval = self.cidx_2_cval(cidx)
            payoff = self.payoff(sval,cval,t)
            running_total_payoff += self.BETA**t*payoff
            result += [{ 'Period' : t,
                          'InitialState' : sval,
                          'OptimalControl' : cval,
                          'CurrentPeriodPayoff' : payoff,
                          'T0CumulativePayoff' : running_total_payoff,
                          'CurrentPeriodValueFunction' : self.VF[sidx,t],
                          'CuurentPeriodCostate' : self.compute_costate(sval,t)
                          }]
            
            sval_new = self.newstate(sval,cval,t)
            sidx = self.sval_2_sidx(sval_new)

        result_df = pd.DataFrame(result).set_index('Period')

        if csv_out is not None :
            print("Writing CSV file at:",csv_out)
            result_df.to_csv(csv_out)
            
        print(result_df.to_string())



    ###########################
    ## Model parameters
    ###########################
    #Each of these functions may take a N x 1 numpy array of state/control values
    #And potentially a scalar time index. Their return values are also numpy
    #N x 1 numpy arrays
            
    #Compute the payoff array in period t for some sval, cval
    def payoff(self,sval,cval,t) :
        p = (50 - 0.5*cval*cval-cval)*cval - 4*cval
        
        #Check for reserve constraint violations and replace the
        #Payoff with the constraint violation value in those cases
        p = np.where(sval - cval < 0, self.CONSTR_VIOLATION, p)

        return p

    #Return vector of scrap values for each control
    def scrap_value(self) :
        return np.zeros_like(self.s_grid)
        
    #Define state transitions from period t to t+1
    def newstate(self,sval,cval,t) :
        #The new state is just the old state minus the control
        return sval - cval
            
        
#####################
## Sample code to run the solver
#####################
x=finite_horizon()
x.backwards_induct()
x.forward_simulate(20)
for sval in range(0,101) :
    print("State:",sval,"Costate:",x.compute_costate(sval,0))
    
