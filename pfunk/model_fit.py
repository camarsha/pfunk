import os
import numpy as np
from scipy import interpolate
from scipy import optimize as opt
from . import fresco_classes as fc
import scipy.interpolate  
from scipy.stats import norm
from scipy.stats import norm
import sys
from scipy.stats import uniform



# General utilities for quick spectroscopic factor checks.

def cross_interpolate(cross):
    return interpolate.UnivariateSpline(cross.theta,
                                        cross.sigma,
                                        s=0)

def sf_chi_sq(theory, exper, err, sf):
    return (((theory*sf)-exper)/(err))**2.0


def spec_chi(sf, cross, data):
    spline = cross_interpolate(cross)
    theory = spline(data.theta)
    exper = data.sigma
    err = data.erry
    #create list of spectroscopic factor so I can use map
    sf = sf*np.ones(len(exper))
    chi_list = list(map(sf_chi_sq, theory, exper, err, sf))
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

    def __init__(self, model, percent_range=.2): #user selects what range they want to vary potentials within defualts to 20%
        self.model = model
        self.fresco = model.fresco
        self.x0 = np.asarray(self.model.x0[:])
        self.init_chi = self.lnprob(self.x0) #get inital chi square value
        print("The initial logprob value is ", self.init_chi) 
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

    def lnprob(self, x):
        val = self.model.lnprob(x)
        return -1.0 * val

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
        result = opt.basinhopping(self.lnprob, self.x0,minimizer_kwargs={'method':'Nelder-Mead'},
                                  niter=self.steps, stepsize=self.step_size,
                                  T=self.T,callback=self.print_fun,
                                  accept_test=self.bounds, take_step=self.take_step)
        self.results = result
        #now use the best result to generate the fit

    #differential evolution approach
    def run_differential(self):
        #get bounds
        up, low = self.set_bounds(self.percent_range, bnds=False)
        bnds = [(i,j) for i,j in zip(up,low)]
        result = opt.differential_evolution(self.lnprob, bnds)
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




        
