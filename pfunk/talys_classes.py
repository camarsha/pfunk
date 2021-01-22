import pandas as pd
#from . import fresco_classes as fc
import os
import numpy as np


def filerun(filename):
    """Run talys on the input file given
    by 'filename'

    :param filename: name of the talys input file
    :returns: None

    """

    command = 'talys' + '<' + filename + '> /dev/null'
    os.system(command)


def make_energies(low, high, spacing, filename='range'):
    energies = np.linspace(low, high, spacing)
    energies = np.around(energies, decimals=3)
    np.savetxt(filename, energies, fmt='%.3f')

# class TalysInput(fc.NamelistInput):

#     """
#     The ultimate goal will require us to read elastic and reaction
#     cross sections at multiple energies, so this time we will wrap
#     the read functions into the input class
#     """

#     def __init__(self, filename):

#         """
#         Read the file into an array. Each array element is
#         one variable.

#         :param filename: talys input file
#         """
#         fc.NamelistInput.__init__(self, filename)
#         self.energies = []
        
#     def unpacklist(self, alist):
#         """ Create a list that expands all elements
#         that are tuples.

#         :param alist: input list
#         :returns: new list

#         """
#         new_list = []
#         for ele in alist:
#             if isinstance(ele, tuple):
#                 for subele in ele:
#                     new_list.append(subele)
#             else:
#                 new_list.append(ele)
#         return new_list

#     def find_tuples(self, alist):
#         """ Create a list with the indices of the tuples, i.e duplicate values.
#         This list is stored as a member variable :param:`self.dup_size`

#         :param alist: list that contains tuples

#         """

#         self.dup_size = [len(i) if isinstance(i, tuple) else 1 for i in alist]

#     def create_names_positions(self, names, positions):
#         """This method initializes all of the information
#         associated with the parameter names and locations.
#         Importantly it handles all of the tuples.

#         :param names: list of all FRESCO variable names
#         :param positions: list of array indices to find variables

#         """
#         self.find_tuples(positions)
#         self.names = self.unpacklist(names)
#         self.name_positions = self.unpacklist(positions)
#         self.unique_name_positions = [i[0] if isinstance(i, tuple)
#                                       else i for i in positions]

#     def transform_values(self, values):
#         """ Transforms an array of n values to
#         one of m values, where m is the length of
#         the vector including the duplicated/shared values defined before.

#         :param values: np.array of n values
#         :returns: np.array of m values

#         """

#         m_values = [np.repeat(i, j) for i, j in zip(values, self.dup_size)]
#         return np.concatenate(m_values)  # Flatten everything

#     def swap_values(self, values):
#         """Method that takes array of values from minimization or sampler
#         and generates a new FRESCO file to run.

#         :param values: np.array of new values
#         :returns: generates a new input file named "new_input"

#         """
#         values = self.transform_values(values)
#         new_values = []
#         # Construct the new strings to replace
#         for i, j in zip(self.names, values):
#             new_values.append('    '+i+' = '+str(j)+'\n')  # Format the string
#         self.new_file[self.name_positions] = new_values  # Swap them
#         # Now write to file
#         np.savetxt('new_input', self.new_file,
#                    fmt=('%s'), newline='')

#     def initial_values(self):
#         """ Find the initial values of the parameters that have been
#         assigned line numbers and names.
#         """
#         self.x0 = []  # Initial values
#         for ele in self.unique_name_positions:
#             value = self.initial_file[ele].split('=')[1]
#             self.x0.append(float(value))
#         self.x0 = np.asarray(self.x0)

#     def create_original(self):
#         """ Generate "new_input" file that is
#         identical to the original input file.

#         """
#         np.savetxt('new_input', self.initial_file,
#                    fmt=('%s'), newline='')
