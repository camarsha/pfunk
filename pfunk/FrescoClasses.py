import numpy as np
import re
import os
from collections import OrderedDict
import pandas as pd

############################################
###############Functions####################
############################################



def filerun(filename):
    command = 'fresco' + '<' + filename + '> /dev/null' 
    os.system(command)

        
# Reads fort.200 files returns LineObject
def read_cross(filename):
    cross = pd.read_csv(filename, sep='\s+', skiprows=10, header=None)
    theta = cross[cross.columns[0]].tolist()[:-1]
    sigma = cross[cross.columns[1]].tolist()[:-1]
    E = 0.0
    graphline = LineObject(theta,sigma,E,'0','+')
    return graphline

# Reads two col. data files returns DataObject
def read_data(filename,delim=None):
    if delim:
        data = pd.read_csv(filename, sep=delim)

    else:
        data = pd.read_csv(filename, sep='\s+')
    theta = data[data.columns[0]].tolist()
    sigma = data[data.columns[1]].tolist()
    try:
        erry =  data[data.columns[2]]
        erry[erry == 0.0] = 1.e-6
        erry = erry.tolist()
    except IndexError:
        erry = None
    graphline = DataObject(theta,sigma,erry)
    return graphline

#Reads fort 17 file and returns a wavefunction class object to plot
#def read_wavefunction(filename):


#do a sum of two cross sections that have the same angles
def cs_sum(cs1,cs2):
    if not cs1.theta == cs2.theta:
        print("These cross sections don't have the same angles!")
    else:
        return LineObject(cs1.theta,cs1.sigma+cs2.sigma,cs1.E,cs1.J,cs1.par)





############################################
#########Classes For Plotting###############
############################################

#new generic class for angular distrubutions
class Angles():
    def __init__(self,theta,sigma):
        self.theta = np.asarray(theta, dtype='float64')
        self.sigma = np.asarray(sigma, dtype='float64')


#This is the tenenative class for graphs. It includes scaling for elastic fits.
class LineObject(Angles):
    def __init__ (self,theta,sigma,E,J,par):
        self.E = E
        self.J = J
        self.par = par
        Angles.__init__(self,theta,sigma)

    def scale_it(self,value,angle,constant=None):
        if constant:
            self.sigma[:] = value*self.sigma
        else:
            index = self.find_angle(angle)
            scale = value/self.sigma[index]
            print('Factor is: ', scale) 
            #Added slice overwrite to be a bit more careful with list
            self.sigma[:] = scale*self.sigma

            
    #Picks out list index for a given angle.
    def find_angle(self,angle):
        angle = float(angle)
        if angle in self.theta:
            return np.argwhere(self.theta == angle) 
        
        else:
            angle = float(input('Angle not found try again! \n'))
            print(self.theta)
            self.find_angle(angle)

    #resize angles
    def angle_range(self,angle):
        index = self.find_angle(angle)+1 #slicing is exclusive, so we need to add one
        return (self.theta[0:index],self.sigma[0:index])
        
    #function for angle ref Ian Thompsons's book written in a form for root finding since it is transendental
    def com_fun(self,x,a,b):
        return (np.tan(a) - (np.sin(x)/(b+np.cos(x))))

    #function for cross section
    def cross_factor(self,com_angle,rho):
        return ((1+rho**2+2*rho*np.cos(com_angle))**(3.0/2.0))/(np.abs(1+rho*np.cos(com_angle)))

    #lab angle transformation
    def lab_fun(self,a,b):
        return (np.arctan((np.sin(a)/(b+np.cos(a)))))

    
    #function to make angles positive
    def make_positive(self,angle):
        np.putmask(angle,angle<0.0,angle+180.0) #putmask is destructive
        print(angle)
        return angle
    
    #Transfers lab frame data to center of mass.
    #Freaks out at 90 degrees as one might expect
    def to_com(self,massa,massb,massc,massd,Elab,Q):
        angle = self.theta*(np.pi/180.0) #to radians 
        rho = np.sqrt((massa*massc)/(massd*massb)*Elab/(Elab+Q))
        sol = optimize.fsolve(self.com_fun,angle,args=(angle,rho))
        #now alter sigma
        cs_scale = self.cross_factor(sol,rho) #solve transformation for cross section
        com_angle = sol*(180.0/np.pi) #to degrees
        com_angle = self.make_positive(com_angle)
        self.theta = com_angle
        self.sigma = cs_scale*self.sigma

    #from com to lab
    def to_lab(self,massa,massb,massc,massd,Elab,Q):
        angle = self.theta*(np.pi/180.0)
        rho = np.sqrt((massa*massc)/(massd*massb)*Elab/(Elab+Q))
        lab_scale = (self.cross_factor(angle,rho))**(-1.0) #uses com angles for scale
        lab_angle = self.lab_fun(angle,rho)*(180.0/np.pi)
        lab_angle = self.make_positive(lab_angle)
        self.theta = lab_angle
        self.sigma = lab_scale*self.sigma
        
#new subclass for experimental data.
class DataObject(LineObject):
    def __init__(self,theta,sigma,erry):
        Angles.__init__(self,theta,sigma)
        #We do not expect all data files to have errors
        self.erry = np.asarray(erry)

            
class NamelistInput():

    """
    This assumes variables have been laid out line by line. We can simply define
    which lines to alter, and hopefully save time.
    """

    def __init__(self, filename):

        self.initial_file = []
        
        with open(filename, 'r') as f:
            for line in f:
                self.initial_file.append(line)

        self.name_positions = []
        self.new_file = np.asarray(self.initial_file)
        self.names = []
         
    def find_var_name(self, name):
        """
        This is the initial positions of the variables to be
        changed. Once found, the goal is to never match strings again.
        """
        
        positions = []

        # Pick out all occurrences of the variable
        for i, ele in enumerate(self.initial_file):
            if name in ele:
                positions.append(i) 

        if len(positions) == 0:
            print('Variable '+name+' not found!')

        elif len(positions) > 1:
            print('Multiple occurrences of '+name+' found.')
            print('Adding variable on lines '+str(positions))
            self.names.append()
            for ele in positions:
                self.name_positions.append(ele)

        else:
            print('Adding variable on lines '+str(positions[0]))
            self.name_positions.append(positions[0])

    def swap_values(self, values):
        """
        Replaces the variables in the array and writes to a new input file.
        """
        
        new_values = []
        # Construct the new strings to replace
        for i, j in zip(self.names,values):
            new_values.append('    '+i+' = '+str(j)+'\n') # Format the string properly
        self.new_file[self.name_positions] = new_values # Swap them
        # Now write to file
        np.savetxt('new_input', self.new_file,
                   fmt=('%s'), newline='')

    def initial_values(self):
        """
        Find the initial values of the parameters that have been assigned 
        line numbers and names.
        """
        self.x0  = [] # Initial values 
        for ele in self.name_positions:
            value = self.initial_file[ele].split('=')[1]
            self.x0.append(float(value))
