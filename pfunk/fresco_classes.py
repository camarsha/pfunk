import numpy as np
import os
import pandas as pd


def filerun(filename):
    """filerun(filename)
    Runs a fresco input file. Output is piped into /dev/null
    in order to avoid writing output file.

    :param filename: name of the fresco input file to run

    """
    command = 'fresco' + '<' + filename + '> /dev/null'
    os.system(command)


def read_cross(filename):
    """ Reads fort.200 files from FRESCO returns
    instance of LineObject.

    :param filename: Name of fort.20* file to read.
    :returns: instance of LineObject

    """

    cross = pd.read_csv(filename, sep='\s+', skiprows=10, header=None)
    theta = cross[cross.columns[0]].tolist()[:-1]
    sigma = cross[cross.columns[1]].tolist()[:-1]
    graphline = LineObject(theta, sigma)
    return graphline


def read_data(filename, delim=None):
    """ Reads col. data files returns DataObject.
    Expected format is:


    Angles   Cross Sections   Uncertainties
    5.0      10.0             0.2
    etc.

    :param filename: Name of the data file
    :param delim: The character that separates the columns
    :returns: instance of DataObject

    """
    if delim:
        data = pd.read_csv(filename, sep=delim)

    else:
        data = pd.read_csv(filename, sep='\s+')

    theta = data[data.columns[0]].tolist()
    sigma = data[data.columns[1]].tolist()

    try:
        erry = data[data.columns[2]]
        erry[erry == 0.0] = 1.e-6
        erry = erry.tolist()
    except IndexError:
        erry = None

    graphline = DataObject(theta, sigma, erry)
    return graphline


class Angles():
    def __init__(self, theta, sigma):
        """Basic class that defines differential cross
        sections.

        :param theta: :code:`np.array` of angles
        :param sigma: :code:`np.array` of differential cross sections

        """
        self.theta = np.asarray(theta, dtype='float64')
        self.sigma = np.asarray(sigma, dtype='float64')


class LineObject(Angles):
    def __init__(self, theta, sigma):
        """Class intended for theoretically calculated
        cross sections. At this time, it has no
        additional features when compared to :class:`Angles`.

        :param theta: :code:`np.array` of angles
        :param sigma: :code:`np.array` of differential cross sections

        """
        Angles.__init__(self, theta, sigma)


class DataObject(Angles):
    def __init__(self, theta, sigma, erry):
        """

        :param theta: :code:`np.array` of angles
        :param sigma: :code:`np.array` of differential cross sections
        :param erry: :code:`np.array` of cross section uncertainties

        """
        Angles.__init__(self, theta, sigma)
        self.erry = np.asarray(erry)

    # def rutherford(self, E_lab, theta):
    
    #     T = self.E_lab * (self.A.m)/(self.a.m + self.A.m)
    #     # First all the constant parts.
    #     const = 1.296*((self.a.Z*self.A.Z)/T)**2.0
    #     theta = (np.pi/180.0)*(theta/2.0)  # Go ahead and convert
    #     temp = np.sin(theta)**(4.0)
    #     sigma_R = const*(1.0/temp)
    #     return sigma_R

    # def ratio_to_rutherford(self, a, A, b, B, E_lab, theta) 

class NamelistInput():

    """
    Class that keeps track of a FRESCO input file. File is assumed to
    have one variable per line, which can then be specified and changed,
    """

    def __init__(self, filename):

        """
        Read the file into an array. Each array element is
        one variable.

        :param filename: fresco input file
        """
        self.initial_file = []

        with open(filename, 'r') as f:
            for line in f:
                self.initial_file.append(line)

        self.name_positions = []
        # Bug fix, array could be cut off due to dtype being too short.
        self.new_file = np.asarray(self.initial_file[:], dtype='<U60')
        self.names = []

    def unpacklist(self, alist):
        """ Create a list that expands all elements
        that are tuples.

        :param alist: input list
        :returns: new list

        """
        new_list = []
        for ele in alist:
            if isinstance(ele, tuple):
                for subele in ele:
                    new_list.append(subele)
            else:
                new_list.append(ele)
        return new_list

    def find_tuples(self, alist):
        """ Create a list with the indices of the tuples, i.e duplicate values.
        This list is stored as a member variable :param:`self.dup_size`

        :param alist: list that contains tuples

        """

        self.dup_size = [len(i) if isinstance(i, tuple) else 1 for i in alist]

    def create_names_positions(self, names, positions):
        """This method initializes all of the information
        associated with the parameter names and locations.
        Importantly it handles all of the tuples.

        :param names: list of all FRESCO variable names
        :param positions: list of array indices to find variables

        """
        self.find_tuples(positions)
        self.names = self.unpacklist(names)
        self.name_positions = self.unpacklist(positions)
        self.unique_name_positions = [i[0] if isinstance(i, tuple)
                                      else i for i in positions]

    def transform_values(self, values):
        """ Transforms an array of n values to
        one of m values, where m is the length of
        the vector including the duplicated/shared values defined before.

        :param values: np.array of n values
        :returns: np.array of m values

        """

        m_values = [np.repeat(i, j) for i, j in zip(values, self.dup_size)]
        return np.concatenate(m_values)  # Flatten everything

    def swap_values(self, values):
        """Method that takes array of values from minimization or sampler
        and generates a new FRESCO file to run.

        :param values: np.array of new values
        :returns: generates a new input file named "new_input"

        """
        values = self.transform_values(values)
        new_values = []
        # Construct the new strings to replace
        for i, j in zip(self.names, values):
            new_values.append('    '+i+' = '+str(j)+'\n')  # Format the string
        self.new_file[self.name_positions] = new_values  # Swap them
        # Now write to file
        np.savetxt('new_input', self.new_file,
                   fmt=('%s'), newline='')

    def initial_values(self):
        """ Find the initial values of the parameters that have been
        assigned line numbers and names.
        """
        self.x0 = []  # Initial values
        for ele in self.unique_name_positions:
            value = self.initial_file[ele].split('=')[1]
            self.x0.append(float(value))
        self.x0 = np.asarray(self.x0)

    def create_original(self):
        """ Generate "new_input" file that is
        identical to the original input file.

        """
        np.savetxt('new_input', self.initial_file,
                   fmt=('%s'), newline='')
