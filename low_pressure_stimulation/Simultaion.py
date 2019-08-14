
# coding: utf-8

# # Low pressure stimulation of subsurface reservoir
# Description: This notebook presents a weakly coupled flow, mechanics and fracture deformation problem reported in the paper *PorePy: An Open-Source Simulation Tool for Flow and Transport in Deformable Fractured Rocks*, by
# Eirik Keilegavlen, Alessio Fumagalli, Runar Berge, Ivar Stefansson, Inga Berre. See arXiv:1712:00460 for details. The code in the notebook was that used to produce figure 7 in the paper, and if ran on a separate system, (almost) the same results should result - factors such as grid generation, linear solvers etc. may produce minor differences.
# 
# To navigate quickly through the procedure, the main steps are: i) Create a mesh, ii) assign parameters for both flow and transport problems, iii) set up appropriate solvers, iv) discretize and solve. 
# 
# The equations we solve are:
# Flow:
# $$ \phi c_f \frac{\partial p}{\partial t} - \nabla \cdot \mathcal K\nabla p  = q$$
# 
# Elasticity:
# $$ \nabla \cdot \sigma = 0$$
# 
# Friction:
# $$ T_s \le \mu (T_n - p)$$
# 
# ## Preliminaries
# To run this, you need to have PorePy installed and set up with Gmsh. The simulations will be time consuming. 
# 
# ## Imports
# The first step is to import all dependencies

# In[1]:


import numpy as np
import scipy.sparse as sps
# For plotting 
from IPython.display import HTML, display
# Porepy
import porepy as pp


# ## Grid generation
# The below function creates a 3D fracture network from a a set of elliptic fractures given in the data file fractures.csv.

# In[2]:


def create_grid():
    file_name = 'fractures.csv'  
    data = np.genfromtxt(file_name, delimiter=',')
    data = np.atleast_2d(data)
    centers = data[:, 0:3]
    maj_ax = data[:, 3]
    min_ax = data[:, 4]
    maj_ax_ang = data[:, 5]
    strike_ang = data[:, 6]
    dip_ang = data[:, 7]
    if data.shape[1] == 9:
        num_points = data[:, 8]
    else:
        num_points = 16 * np.ones(data.shape[0])

    frac_list = []

    for i in range(maj_ax.shape[0]):
        frac_list.append(pp.EllipticFracture(centers[i, :],
                                             maj_ax[i],
                                             min_ax[i],
                                             maj_ax_ang[i],
                                             strike_ang[i],
                                             dip_ang[i],
                                             num_points[i]))
    frac_network = pp.FractureNetwork(frac_list)
    box = {'xmin': -5000, 'ymin': -5000, 'zmin': -5000,
           'xmax': 10000, 'ymax':  10000, 'zmax': 10000}
    gb = pp.meshing.simplex_grid(frac_network, box, mesh_size_bound=10000,
                                 mesh_size_frac=500, mesh_size_min = 200)
    return gb


# ## Problem setup: Parameter specification and solvers
# To set up the flow model, we will use a SlightlyCompressibleModel. The model automatically sets up simple discretization scheme, and tries to design decent linear solvers for the resulting systems of equations.
# 
# ### Pressure data
# The first step is to provide simulation data. Simulation parameters are stored as part of the GridBucket, but accessing this can be somewhat cumbersome. To assist the assignment, and also to provide a reasonable way of setting default parameters, each of the models (pre-defined solvers), are accompanied by a DataAssigner. This can be used directly to define a simulaiton with default parameters, or modified as desired. Below, we give an example for the pressure equation.

# In[3]:


# The units module contains various physical constants
# Set a relative high matrix permeability, this would correspond to
# a quite high density of upscaled fractures.
class MatrixDomain(pp.SlightlyCompressibleDataAssigner):
    """ Set data for the 3D domain (matrix) in the pressure equation.
    
    Fields that are not assigned here, will have the default values 
    prescribed in SlighlyCompressibleDataAssigner (which again may point further to defaults
    in the Parameter class).
    """        
    def initial_condition(self):
        p = 40 * pp.MEGA * pp.PASCAL
        return p * np.ones(self.grid().num_cells)

    def compressibility(self):
        return 4.6e-10 / pp.PASCAL

    def permeability(self):
        kxx = np.ones(self.grid().num_cells) * pp.NANO * pp.DARCY
        return pp.SecondOrderTensor(3, kxx / self.viscosity())

    def viscosity(self):
        return .45 * pp.MILLI * pp.PASCAL * pp.SECOND

    def porosity(self):
        return 0.01 * np.ones(self.grid().num_cells)

    def density(self):
        return 1014 * pp.KILOGRAM / pp.METER**3

    
class FractureDomain(MatrixDomain):
    def __init__(self, g, data):
        self.E0 = .1 * pp.MILLI * pp.METER * np.ones(g.num_cells)
        self.Ed = 0. * np.ones(g.num_cells)        
        MatrixDomain.__init__(self, g, data)

    def aperture(self):
        return (self.E0 + self.Ed)**(3 - self.grid().dim)

    def permeability(self):
        kxx = (self.E0 + self.Ed)**2 / 12
        return pp.SecondOrderTensor(3, kxx / self.viscosity())
#        return tensor.SecondOrder(self.g.dim, np.ones(self.g.num_cells))

    def porosity(self):
        return 1 * np.ones(self.grid().num_cells)


class InjectionDomain(FractureDomain):
    def source(self, t):
        tol = 1e-4
        value = np.zeros(self.grid().num_cells)

        cell_coord = np.atleast_2d(np.array([1200, 2200, 2000])).T
        distance = np.sqrt(np.sum(np.abs(self.grid().cell_centers - cell_coord)**2, axis=0))
        cell = np.argmin(distance)

        if t < 6000 * pp.SECOND + 1e-6:
            value[cell] = 10.0 * pp.KILOGRAM / pp.SECOND / self.density()
        return value


# ### Mechanics data
# Next, we define the simulation data for the linear elasticity problem: $\nabla\cdot  \sigma = 0$

# In[4]:


class MechDomain(pp.StaticDataAssigner):
    """ Set data for the 3D domain (matrix) for the linear elasticity.
    
    Fields that are not assigned here, will have the default values 
    prescribed in StaticDataAssigner (which again may point further to defaults
    in the Parameter class).
    """   
    def bc(self):
        """
        The default boundary condition is Neuman, so we overload this function 
        to define zero Dirichlet condition on the boundary. 
        """
        bc_cond = pp.BoundaryCondition(
            self.grid(), self.grid().get_all_boundary_faces(), 'dir')
        return bc_cond

    def stress_tensor(self):
        """
        We set the stress tensor based on the parameters assigned to the Rock class
        """
        mu = self.data()['rock'].MU * np.ones(self.grid().num_cells)
        lam = self.data()['rock'].LAMBDA * np.ones(self.grid().num_cells)
        return pp.FourthOrderTensor(self.grid().dim, mu, lam)

    def background_stress(self):
        """
        The background stress defines stress tensor, and we assume the same stress
        throughout our domain
        """
        T_x = .120 * pp.GIGA * pp.PASCAL
        T_y = .080 * pp.GIGA * pp.PASCAL
        T_z = .100 * pp.GIGA * pp.PASCAL
        sigma = -np.array([[T_x, 0, 0], [0, T_y, 0], [0, 0, T_z]])
        return sigma


# ### Assign data
# Having defined parameter classes for all geometric objects, assigning the data is easy: Simply loop over the GridBucket, and choose DataAssigner according to the grid dimension.

# In[5]:


# Define method to assign parameters to all nodes in the GridBucket
def assign_data(gb):
    # First we define the rock
    matrix_rock = pp.Granite()
    matrix_rock.MU = 20 * pp.GIGA * pp.PASCAL
    matrix_rock.LAMBDA = 20 * pp.GIGA * pp.PASCAL
    
    # We define the variable aperture_change which will be used to update the aperture
    # at each time step
    gb.add_node_props(['aperture_change'])
    for g, d in gb:
        d['aperture_change'] = np.zeros(g.num_cells)
        if g.dim == 3:
            d['rock'] = matrix_rock
            d['flow_data'] = MatrixDomain(g, d)
            d['mech_data'] = MechDomain(g, d)
            d['slip_data'] = pp.FrictionSlipDataAssigner(g, d)
        else:
            # We define an injection in the first fracture
            if d['node_number'] == 1:
                d['flow_data'] = InjectionDomain(g, d)
            else:
                d['flow_data'] = FractureDomain(g, d)


# ### Transfer data
# The linear elasticity and fracture deformation models are defined on the 3D grid. The traction calculations and shear and normal deformation will take place on the faces of the 3D grid that are connected to the 2D fracture cells. For the flow problem, however, the aperture and pressure are defined in the cells of the 2D fractures. We therefore need two simple functions that map data from the cells of 2D grids to faces of 3D grids and vice versa
# 

# In[6]:


def cell_2_face(gb, variable):
    g3 = gb.grids_of_dimension(3)[0]
    data3 = gb.node_props(g3)
    face_variable = np.zeros(g3.num_faces)
    for g, d in gb:
        if g.dim != 2:
            continue
        f_c = gb.edge_props((g3, g), 'face_cells')
        ci, fi, _ = sps.find(f_c)
        face_variable[fi] = d[variable][ci]

    data3['face_' + variable] = face_variable

def face_2_cell(gb, variable):
    g3 = gb.grids_of_dimension(3)[0]
    data3 = gb.node_props(g3)
    for g, d in gb:
        if g.dim != 2:
            continue
        f_c = gb.edge_props((g3, g), 'face_cells')
        ci, fi, _ = sps.find(f_c)

        cell_variable = np.zeros(g.num_cells)
        num_hit = np.zeros(g.num_cells)
        for i, face in enumerate(fi):
            cell_variable[ci[i]] += data3[variable][face]
            num_hit[ci[i]] += 1
        d[variable] = cell_variable / num_hit


# ### Aperture update
# 
# At each time step the fracture may possible slip. If a fracture do slip we will get an equivalent increse in aperture. For convenience, we define a function that updates the aperture based on this aperture increase
# 

# In[7]:


def update_aperture(gb, name='aperture_change'):
    for g, d in gb:
        if g.dim != 2:
            continue
        E0 = d['flow_data'].E0
        d['param'].set_aperture(E0 + d[name])


# # Set up solvers
# We are finally ready to define our solver objects and solve for flow and temperature. With all parameters defined, this is a relatively simple code:

# In[8]:


gb = create_grid()
g3 = gb.grids_of_dimension(3)[0]
data3 = gb.node_props(g3)

# Create an exporter object, and dump the grid
exporter = pp.Exporter(gb, 'low_pressure_stimulation', folder='results')
exporter.write_vtk()


# The resulting grid looks like this, after some manipulation in Paraview

# In[9]:


display(HTML("<img src='fig/mesh.png'>"))


# ### Define solvers
# The flow problem is dependent on time, and needs the time step as an argument. The mechanics and fracture deformation are both quasi-static, i.e., slip happens instantaneous when the Mohr-Colomb criterion is violated
# 

# In[10]:


# Define the time stepping
dt = 10 * pp.MINUTE
T = 18 * dt
t = 0
# Assign data to grid bucket
assign_data(gb)

# Define pressure solver for the given grid.
# This will assign parameters, using the above classes
flow_solver = pp.SlightlyCompressibleModel(gb, time_step=dt)
mech_solver = pp.StaticModel(g3, data3)
friction_solver = pp.FrictionSlipModel(g3, data3)


# ## Define Time loop

# In[11]:


# save initial condition
flow_solver.pressure('pressure')
friction_solver.aperture_change('aperture_change')
face_2_cell(gb, 'aperture_change')
exporter.write_vtk(['pressure', 'aperture_change'], 0)

# Discretize linear elasticity
mech_solver.reassemble()

# List for storing discretization times
time_steps = []
time_steps.append(t)
k = 0

while t < T:
    t += dt
    k += 1
    time_steps.append(t)
    print('Solving time step: ', k)
    
    # Solve flow
    flow_solver._solver.update(t)    # Update injection
    flow_solver.reassemble()         # Reasemble rhs
    flow_solver.step()               # solve for next time step
    flow_solver.pressure('pressure') # save solution to data
    cell_2_face(gb, 'pressure')      # map cell pressure to 3D faces

    # solve mechanics
    do_slip = True
    # At the start of each time step we assume no fractures are slipping
    friction_solver.is_slipping = np.zeros(g3.num_faces, dtype=np.bool)
    while np.any(do_slip):
        mech_solver.solve(discretize=False)
        mech_solver.traction('traction')
        do_slip = friction_solver.step()
        data3['param'].set_slip_distance(friction_solver.x.ravel('F'))

    friction_solver.aperture_change('aperture_change')  # Save aperture change to data
    face_2_cell(gb, 'aperture_change')                  # Map aperture change to 2D cells
    update_aperture(gb)                                 # Update the aperture
    exporter.write_vtk(['pressure', 'aperture_change'], time_step=k)
friction_solver
exporter.write_pvd(np.array(time_steps))


# Here is what the evolution in aperture looks like

# In[12]:


HTML('<img src="fig/aperture_change.gif">')

