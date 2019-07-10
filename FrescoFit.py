import os
import numpy as np
from scipy import interpolate
from scipy import optimize as opt
import FrescoClasses as fc
import pymc as pm
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
from scipy.stats import norm
import sys
from scipy.stats import uniform
#This file will provide all the algorithms for doing a  
#chi-squared minimization on the the fresco generated cross sections

#First we must create a function from the fresco output
#This uses a cubic spline with documentation found at
#http://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
#Give it an instance of lineobject from frescoclasses and desired angle.
def cross_interpolate(cross):
    return interpolate.splrep(cross.theta,cross.sigma)
    


#Just function to return chi squared for chi-squared minimization.
def chi_sq(theory,exper,err):
    return ((theory-exper)/(err))**2.0

def sf_chi_sq(theory,exper,err,sf):
    return ((theory*sf-exper)/(err))**2.0


#Pretty much the heart of this code given all the inputs changes potential values,
#runs fresco, creates new spline, returns chisquare for minimization.
def elastic_chi(val,fresco,data,pot,term,var):
    for i,j in zip(var,val):
        fresco.change_pot(pot,term,i,str(j)) #Change potentials
    fresco.update_all()
    fresco.write('new_input')
    fc.filerun('new_input')
    cross = fc.read_cross('fort.201')
    spline = cross_interpolate(cross)
    theory = interpolate.splev(data.theta,spline)
    exper = data.sigma
    err = data.erry #Might need to consider aysmmetric data in future. Keep eye out.
    chi_list = list(map(chi_sq,theory,exper,err))
    return np.sum(chi_list)

def spec_chi(sf,cross,data):
    spline = cross_interpolate(cross)
    theory = interpolate.splev(data.theta,spline)
    exper = data.sigma
    err = data.erry
    sf = sf*np.ones(len(exper))#create list of spectroscopic factor so I can use map
    chi_list = list(map(sf_chi_sq,theory,exper,err,sf))
    return np.sum(chi_list)

def spec_factors(cross,data):
    result = opt.minimize(spec_chi,1.0,method='Nelder-Mead',
                          args=(cross,data))
    return result.x

#trying to enforce bounds on the basin hopping with this condition. Ripped straight from scipy page

class BasinBounds(object):
    def __init__(self, xmax, xmin): #should pass upper and lower bounds that we want to the object
         self.xmax = np.array(xmax)
         self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
         x = kwargs["x_new"] #proposed values passed by minimizer
         tmax = bool(np.all(x <= self.xmax)) #see if they fall within range
         tmin = bool(np.all(x >= self.xmin))
         return tmax and tmin

class ElasticStep(object):
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize
    def __call__(self, x):
        s = self.stepsize
        #x[0] += np.random.uniform(-500.*s, 500.*s)
        x[:] += np.random.uniform(-s, s, x[:].shape)
        #x[3] += np.random.uniform(-10.*s, 10.*s)
        return x


#Because of the constrains by the scipy otimize package this needed to be a class, or I would have had global variables out the ass.
class ElasticFit():

    def __init__(self, fresco, data, names, line_numbers,
                 percent_range=.2): #user selects what range they want to vary potentials within defualts to 20%
        self.fresco = fc.NamelistInput(fresco)
        self.fresco.names = names
        self.fresco.name_positions = line_numbers
        self.fresco.initial_values()
        self.data = fc.read_data(data)
        self.x0 = np.asarray(self.fresco.x0)
        self.init_chi = new_elastic_chi(self.x0, self.fresco, self.data) #get inital chi square value
        print("The initial elastic chi squared value is ",self.init_chi) 
        fc.filerun('new_input')
        self.init_cs = fc.read_cross('fort.201') #the inital cross section data
        os.remove('fort.201')
        self.iterations = 0 #number of basin hops completed
        self.percent_range = percent_range
        self.bounds = self.set_bounds(percent_range) #range we are allowed to jump around in. We will still sample, but auto rejected outside this range 

        #These are default parameters that can be set with set_basin
        self.T = 1.0
        self.steps = 50
        self.step_size = .01
        self.take_step = ElasticStep(stepsize=self.step_size)
        #results
        self.results = None
        self.accepted_values = []
        

    #callback function that is passed data after each basin iteration    
    def print_fun(self, x, f, accepted):
        self.iterations = self.iterations + 1
        sys.stdout.write("\r At minimum %.4f accepted %d iteration %d  Total Accepted : %d" % (f, int(accepted),self.iterations, len(self.accepted_values)))
        
        if int(accepted):
            self.accepted_values.append({'x':x, 'chi2':f})
        
    #method that creates a BasinBounds object and returns. Upper and lower created from allowed percent variation
    def set_bounds(self,percent,bnds=True):
        upper = []
        lower = []
        for ele in self.x0: 
                lower.append(ele - (percent*ele))
                upper.append(ele + (percent*ele))
        if bnds:
            bounds = BasinBounds(upper,lower)
            return bounds
        else:
            return lower,upper

    #asks user to give metrolpolis parameters
    def set_basin(self):
        T = float(input('What temperature do you want?'))
        self.T = T
        steps = int(input('How many steps?'))
        self.steps = steps
        step_size = float(input('How large should the trial steps be?'))
        self.step_size = step_size
        
    #start a basin run
    def run(self):
        result = opt.basinhopping(new_elastic_chi,self.x0,minimizer_kwargs={'method':'Nelder-Mead','args':(self.fresco, self.data)},
                                  niter=self.steps,stepsize=self.step_size,T=self.T,callback=self.print_fun, accept_test=self.bounds, take_step=self.take_step)
        self.results = result
        #now use the best result to generate the fit

    def run_norm(self):
        """
        This method assumes there is some systematic shift in the data.
        Data is scaled by N, which is given a starting value of 1.0.
        """
        N = np.array([1.0])
        new_x0 = np.concatenate((N, self.x0))
        result = opt.basinhopping(new_elastic_chi_norm,new_x0,minimizer_kwargs={'method':'Nelder-Mead','args':(self.fresco, self.data)},
                                  niter=self.steps,stepsize=self.step_size,T=self.T,callback=self.print_fun) 
        self.results = result
        #now use the best result to generate the fit
        
    #differential evolution approach
    def run_differential(self):
        #get bounds
        up,low = self.set_bounds(self.percent_range,bnds=False)
        bnds = [(i,j) for i,j in zip(up,low)]
        result = opt.differential_evolution(new_elastic_chi,bnds,args=(self.fresco, self.data))
        self.results = result
        #now use the best result to generate the fit
        new_elastic_chi(result.x,self.fresco,self.data)

    def run_differential_norm(self):
        #get bounds
        up,low = self.set_bounds(self.percent_range,bnds=False)
        norm_bnd = [(-100.0, 100.0)]
        bnds = [(i,j) for i,j in zip(up,low)]
        all_bnds = norm_bnd + bnds
        result = opt.differential_evolution(new_elastic_chi_norm,all_bnds,args=(self.fresco, self.data))
        self.results = result
      
    def single_run(self):
        result = opt.minimize(new_elastic_chi, self.x0, method='Nelder-Mead', args=(self.fresco, self.data))
        self.results = result

class TransferFit(ElasticFit):

    def __init__(self, fresco, elastic_data, transfer_data,
                 names, line_numbers, percent_range=.2):
        ElasticFit.__init__(self, fresco, elastic_data, names, line_numbers, percent_range=percent_range)
        self.transfer_data = fc.read_data(transfer_data)
        self.init_chi = transfer_chi(self.x0, self.fresco, self.data, self.transfer_data)

    def run_transfer(self, elastic_first=True):
        if elastic_first:
            print("Fitting elastic first!")
            self.run()
            self.x0 = self.results.x
            self.accepted_values = []
            self.iterations = self.iterations - self.steps
        result = opt.basinhopping(transfer_chi, self.x0, minimizer_kwargs={'method':'Nelder-Mead','args':(self.fresco, self.data, self.transfer_data)},
                                  niter=self.steps, stepsize=self.step_size, T=self.T, callback=self.print_fun)
        self.results = result

# Yet another go at getting a reasonable run time for fitting.

def transfer_chi(vals, file_object, data_elastic, data_transfer):
    file_object.swap_values(vals)
    fc.filerun('new_input')
    try: 
        cross_elastic = fc.read_cross('fort.201')
        cross_transfer = fc.read_cross('fort.202')
    except IOError:
        return np.inf
    os.remove('fort.201')
    os.remove('fort.202')
    spline_elastic = cross_interpolate(cross_elastic)
    spline_transfer = cross_interpolate(cross_transfer)
    theory_elastic = interpolate.splev(data_elastic.theta, spline_elastic)
    theory_transfer = interpolate.splev(data_transfer.theta, spline_transfer)
    chi_e = list(map(chi_sq, theory_elastic, data_elastic.sigma, data_elastic.erry))
    chi_t = list(map(chi_sq, theory_transfer, data_transfer.sigma, data_transfer.erry))
    return  np.sum(chi_e) + np.sum(chi_t)
    
def new_elastic_chi(vals, file_object, data):
    file_object.swap_values(vals)
    fc.filerun('new_input')
    try:
        cross = fc.read_cross('fort.201')
    except IOError:
        return np.inf
    os.remove('fort.201')
    spline = cross_interpolate(cross)
    theory = interpolate.splev(data.theta, spline)
    exper = data.sigma
    err = data.erry #Might need to consider aysmmetric data in future. Keep eye out.
    chi_list = list(map(chi_sq,theory, exper, err))
    return np.sum(chi_list)

def new_elastic_chi_norm(vals, file_object, data):
    file_object.swap_values(vals[1:])
    fc.filerun('new_input')
    cross = fc.read_cross('fort.201')
    spline = cross_interpolate(cross)
    theory = interpolate.splev(data.theta, spline)
    exper = data.sigma*vals[0]
    err = data.erry*vals[0]
    chi_list = list(map(chi_sq,theory, exper, err))
    return np.sum(chi_list)



def set_bounds(values, percent=.5):
    upper = []
    lower = []
    for ele in values: 
            lower.append(ele - (percent*ele))
            upper.append(ele + (percent*ele))
    return lower,upper


def monte_carlo_unc(initial_file, parameters, line_numbers, percent=.2, samples=10000,
                    flat=False):
    
    # Set up the input and data files
    input_file = fc.NamelistInput(initial_file)
    input_file.names = parameters
    input_file.name_positions = line_numbers
    input_file.initial_values()
    # Array of random samples
    all_values = np.zeros([samples, len(parameters)])
    # Draw the samples
    for i,val in enumerate(input_file.x0):
        if flat:
            temp = uniform.rvs(loc=(val-percent*val), scale=(val+percent*val), size=samples)
        else:
            temp = norm.rvs(loc=val, scale=(percent*val), size=samples)
        all_values[:,i] = temp[:]
    elastic = []
    transfer = []
    #Now loop and run
    for i in range(samples):
        if not i%10:
            sys.stdout.write("\r Iteration %d" % i)
        input_file.swap_values(all_values[i,:])
        fc.filerun('new_input')
        data_e = fc.read_cross('fort.201')
        data_t = fc.read_cross('fort.202')
        elastic.append(data_e)
        transfer.append(data_t)
    return elastic, transfer
        
