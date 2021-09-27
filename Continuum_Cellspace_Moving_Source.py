####Environment for agent (particle) dynamics
from Grid_Space import *
import numpy as np
import os

######Initialize systerm parameters
f_wall_lim = 100.0                      #set the magnitude of the wall repulsion force
f_collision_lim = 100.0                 #set the magnitude of the particle collision force
door_size = 0.5                         #size of door
agent_size = 0.5                        #size of agent (particle)
reward = -0.1
end_reward = 0.

offset = np.array([0.5, 0.5])           #offset after scale to [0, 1]
dis_lim = (agent_size + door_size)/2    #set the distance from the exit which the agent is regarded as left
action_force = 1.0                      #unit action force
desire_velocity = 2.0                   #desire velocity
relaxation_time = 0.5                   #relaxation_time
delta_t = 0.1                           #time difference of simulation
cfg_save_step = 5                       #time steps interval for saving Cfg file

######Initialize Exit positions range [0, 1]
Exit = []  ###No exit

######Initialize obstacles
Ob = []   ###No obstable
Ob_size = []

#####Neighbor cell list for cell looping interactions
cell_list =np.array( [ [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0],
             [0, 0, -1], [1, 0, -1], [1, 1, -1], [0, 1, -1],
             [-1, 1, -1], [-1, 0, -1], [-1, -1, -1], [0, -1, -1], [1, -1, -1] ], dtype = 'int')

#####list of neighbor cells
neighbor_list =np.array( [ [-1, 1, 0], [0, 1, 0], [1, 1, 0],
                          [-1, 0, 0], [0, 0, 0], [1, 0, 0], 
                          [-1, -1, 0], [0, -1, 0],[1, -1, 0] ], dtype = 'int')


#####Define Agent (Particle) class 
class Particle:
    
    def __init__(self, ID,  x, y, z, vx, vy, vz, mass = 80.0, type = 1):
        self.position = np.array((x, y, z))
        self.velocity = np.array((vx, vy, vz))
        self.acc = np.array((0., 0., 0.))
        self.mass = mass
        self.type = type
        self.ID = ID
    
    def leapfrog(self, dt, stage):
        
        if stage ==0:
            self.velocity += dt/2 * self.acc
            self.position += dt*self.velocity          
        else:
            self.velocity = self.velocity + dt/2 * self.acc

    def sacle_velocity(self, value = 1.0):

        self.velocity /= np.sqrt(np.sum(self.velocity**2))
        self.velocity *= value        
  

#####Define Cell class      
class Cell:
    
    def __init__(self, ID, idx, idy, idz, d_cells, L, n_cells):
        
        self.Particles = []  ###Particle list to store agents in this cell
        self.Neighbors = []  ###Identify and store neighbor cells
        self.ID_number = ID  ###ID number of the cell
        self.ID_index = np.array([idx, idy, idz])   #####ID index of the cell
        self.L = np.zeros_like(L)                   ###Lower and upper boundary of the cell
        self.L[:,0] = L[:,0] + self.ID_index * d_cells
        self.L[:,1] = self.L[:,0] + d_cells
        self.n_cells = n_cells
        
        self.find_neighbors()   
        
    def add(self, particle):
        self.Particles.append(particle)
    
    ###find and set the neighbor cells
    def find_neighbors(self):
        
        idx = self.ID_index + cell_list        
        valid = (idx < self.n_cells) & (idx >=0)
        idx = idx[np.all(valid, axis = 1)]
        
        for n in range(len(idx)):
            
            i = idx[n, 0]
            j = idx[n, 1]
            k = idx[n, 2]
            
            N = k * (self.n_cells[0] * self.n_cells[1]) + j * self.n_cells[0] + i
            self.Neighbors.append(N)        
    
#####Continuum Cell Spac, simulation environment
class Cell_Space:
    
    def __init__(self, xmin = 0., xmax = 1., ymin = 0., ymax = 1.,
                 zmin = 0., zmax = 1., rcut = 0.5, dt = 0.1, Number = 1,
                 source_total_steps = 100,
                 n_frames = 10, dl = 3.5, dt_c = 0.001, dr_c = 0.1,
                 compute_C = True):
        
        self.dt = dt
        self.Number = Number    ###Current number of agents
        self.Total = Number     ###Total number of agents
        self.T = 0.             ###Temperature of the system
        
        self.source_total_steps = source_total_steps        
        self.n_frames = n_frames
        self.dl = dl
        self.dr_c = dr_c
        self.dt_c = dt_c
        self.C_Frames = []
        
        self.step_interval = self.source_total_steps/self.n_frames 
        self.vl = self.dl/self.source_total_steps
        self.t = self.dt/self.dt_c
        
        ####Size of the system
        self.L = np.array([ [xmin, xmax], [ymin, ymax], [zmin, zmax] ], dtype = np.float32)
        self.rcut = rcut
        
        ####Number of cells in each dimension
        self.n_cells = np.array(((xmax- xmin), (ymax- ymin) ,(zmax- zmin) ))/rcut 
        self.n_cells = self.n_cells.astype('int')
        
        ####Cell size in each dimension
        self.d_cells = (self.L[:,1] - self.L[:,0])/self.n_cells

        ####Set exit information
        self.Exit = []
        for e in Exit:            
            self.Exit.append(self.L[:,0] + e * (self.L[:,1] - self.L[:,0]) )
            
        ####Set Obstacles information
        self.Ob = []
        self.Ob_size = []
        for idx, ob in enumerate(Ob):  
            tmp = []
            for i in ob:
                tmp.append(self.L[:,0] + i * (self.L[:,1] - self.L[:,0])) 
                
            self.Ob.append(tmp)    
            self.Ob_size.append(Ob_size[idx])

        ####Initialize cells and particles
        self.Cells = []
        self.initialize_cells()
        self.initialize_particles()
   
        ####Set action space
        diag = np.sqrt(2)/2
        self.action = np.array([[0,1,0],[-diag,diag,0],[-1,0,0],[-diag,-diag,0],
                                [0,-1,0],[diag,-diag,0],[1,0,0],[diag,diag,0]] , 
                                dtype = np.float32) ## 8 actions
        self.action *= action_force       
        
        ####initial 2D Grid Space   
        self.Grid = GridSpace_2D(self.L[0,0],
                                 self.L[0,1],
                                 self.L[1,0],
                                 self.L[1,1],
                                 dr = self.dr_c, dt = self.dt_c, D = 1)
        if compute_C:
            self.Compute_Grid_Space_Concentration()
        
        ####set reward
        self.reward = reward     ###reward for taking each time step
        self.end_reward = end_reward  ###reward for exit

    def Compute_Grid_Space_Concentration(self, dynamic = False):
        
        #######initial equlibrium
        step_max = 1000
        s = 1
        
        print('Compute the initial concentration distribution.')
        while s <= step_max:
            
            self.Grid.step_compute()
            s+=1
        
        C = []
        for g in self.Grid.Grids:
            C.append(g.C_prev)
        self.C_Frames.append(C)            
        print('Initial Concentration compute completed!')
        
        if dynamic:
            #########moving source dynamic
            print('Compute Concetration frames.')
            
            for i in range(1,self.n_frames + 1):
                
                x = 5 + i*self.vl*self.step_interval 
                y = 5 + i*self.vl*self.step_interval
                
                self.Grid.reset_source([(x,y)], [2])
                
                s = 1
                while s <= self.t * self.step_interval:
                
                    self.Grid.step_compute()
                    s+=1        
                
                C = []
                for g in self.Grid.Grids:
                    C.append(g.C_prev)
                self.C_Frames.append(C)  
                
            print('Compute Concetration frames completed!')        
        else:
            ############moving source fixed equlibrium
            print('Compute Concetration equlibrium frames.')
            
            for i in range(1,self.n_frames + 1):
                
                x = 5 + i*self.vl*self.step_interval 
                y = 5 + i*self.vl*self.step_interval
                
                self.Grid.reset_source([(x,y)], [2])
                
                s = 1
                while s <= 200:
                
                    self.Grid.step_compute()
                    s+=1        
                
                C = []
                for g in self.Grid.Grids:
                    C.append(g.C_prev)
                self.C_Frames.append(C)  
                

                
            print('Compute Concetration equlibrium frames completed!')
        
    def initialize_cells(self):
                
        nx = self.n_cells[0]
        ny = self.n_cells[1]
        nz = self.n_cells[2]
        
        np = nx * ny
        n_total = np * nz
        
        for n in range(n_total):
            
            ####Convert to the cell ID index
            i = n % np % nx
            j = n % np // nx
            k = n // np
          
            self.Cells.append(Cell(n, i, j, k, self.d_cells, self.L, self.n_cells) )
    
    def Zero_acc(self):
        
        for c in self.Cells:
            for p in c.Particles:
                p.acc[:] = 0.
                
    ####Normalization to [0,1]
    def Normalization(self, position):

        return (position - self.L[:,0])/(self.L[:,1] - self.L[:,0])
    
    ####Normalization to [0,1] at xy plane and take offset
    def Normalization_XY(self, position, offset = offset):

        return (position - self.L[:2,0])/(self.L[:2,1] - self.L[:2,0])- offset
    
    def Integration(self, stage):
        
        self.T = 0.
        
        for c in self.Cells:
            for p in c.Particles:
                p.leapfrog(dt = self.dt, stage = stage)
                self.T += 0.5 * p.mass * np.sum(p.velocity**2)
                
        self.T /= self.Number
     
    ####Berendsen thermostats           
    def Berendsen(self, tem):
        
        NDIM = 2  ###2D
        
        factor = np.sqrt(0.5*NDIM * tem / self.T)
        
        self.T = 0.
        for c in self.Cells:
            for p in c.Particles:
                p.velocity *= factor
                self.T += 0.5 * p.mass * np.sum(p.velocity**2)
        self.T /= self.Number
        
    ####Temperatue adjustment for particles
    def Adjust_temp(self, tem):
        
        NDIM = 2  ###2D

        for c in self.Cells:
            for p in c.Particles:
                vv = p.mass * np.sum(p.velocity**2)
                factor = np.sqrt(NDIM * tem / vv)
                p.velocity *= factor
                
    #####Save to Cfg file          
    def save_output(self, file):  
        
        N_obs = 0
        for ob in self.Ob:
            N_obs += len(ob)
        
        with open(file, 'w+') as f:
            HX = self.L[0,1]- self.L[0,0]
            HY = self.L[1,1]- self.L[1,0]
            HZ = self.L[2,1]- self.L[2,0]
            f.write('''Number of particles = {}
A = 1.0 Angstrom (basic length-scale)
H0(1,1) = {} A
H0(1,2) = 0 A
H0(1,3) = 0 A
H0(2,1) = 0 A
H0(2,2) = {} A
H0(2,3) = 0 A
H0(3,1) = 0 A
H0(3,2) = 0 A
H0(3,3) = {} A
entry_count = 7
auxiliary[0] = ID [reduced unit]
'''.format(self.Number + len(self.Exit) + N_obs, HX, HY, HZ))
            
            f.write('10.000000\nAt\n')
            
            for e in self.Exit:  
                x,y,z = self.Normalization(e)
                f.write('{} {} {} {} {} {} {}\n'.format(x, y, z, 0, 0, 0, -1))   
            
            for idx, b in enumerate(self.Ob): 
                
                if idx == 0:
                    f.write('1.000000\nC\n')
                elif idx == 1:
                    f.write('1.000000\nSi\n')
                
                for e in b:
                    x,y,z = self.Normalization(e)
                    f.write('{} {} {} {} {} {} {}\n'.format(x, y, z, 0, 0, 0, -1))                
            
            if(self.Number !=0):
                f.write('1.000000\nBr\n')
                for c in self.Cells:
                    for p in c.Particles:
                        x,y,z = self.Normalization(p.position)
                        
                        f.write('{} {} {} {} {} {} {}\n'.format(x, y, z, 
                                p.velocity[0], p.velocity[1], p.velocity[2], p.ID))
            

    
    def insert_particle(self, particle):

        index = (particle.position - self.L[:, 0])/ self.d_cells
        
        ####Out of boundary check
        if (index<0).any() or (index >= self.n_cells).any():
            print("Out of boundary, check the velocity and other systerm configuration!")
            
        index = index.astype('int')
        
        N = index[2] * (self.n_cells[0] * self.n_cells[1]) + index[1] * self.n_cells[0] + index[0]
        self.Cells[N].add(particle)
    
    def initialize_particles(self, file = None):
        
        if file == None:    ######initialize particles manually
            
            P_list = []
            for i in range(self.Number):
                
                reselect = True

                while reselect:
                    ####initial positions of 2D random distribution at xy plane
                    pos = self.L[0:2,0] + 0.05 * (self.L[0:2,1] - self.L[0:2,0])  \
                    + np.random.rand(len(self.L) -1) * (self.L[0:2,1] - self.L[0:2,0])*0.9
                    
                    pos = np.append(pos, self.L[2,0] + 0.5* (self.L[2,1] - self.L[2,0]) )
                    
                    ###avoid overlap
                    reselect = False
                    for p in P_list:
                        dis = np.sqrt(np.sum((pos - p)**2))
                        if dis < agent_size:
                            reselect = True
                            break 
 
                    for idx, ob in enumerate(self.Ob):
                        for p in ob:
                            dis = np.sqrt(np.sum((pos - p)**2))
                            if dis < (agent_size + self.Ob_size[idx])/2:
                                reselect = True
                            break 
                        if reselect:
                            break
                       
                    if not reselect:
                        break
                
                P_list.append(pos)                
                pos = pos.tolist()
                
                ####set initial velocity
                v = np.random.randn(len(self.L)) *0.01
                
                v[2] = 0.
                v = v.tolist()
                particle = Particle(i, *pos, *v)
                
                if i ==0:
                    self.agent = particle
                
                self.insert_particle(particle)
                
        else:               #####read from Cfg file
            pass

    
    ###Check positions of agent and move it to the right cell
    def move_particles(self):
        
        for cell in self.Cells:
            
            i = 0
            while i < len(cell.Particles):
                
                position = cell.Particles[i].position
                
                inside = (position >= cell.L[:,0]) & (position < cell.L[:, 1])
                inside = inside.all()
                
                if inside:
                    i+=1
                else:                
                    self.insert_particle(cell.Particles.pop(i))                                        
                    
    ####Region confine force. Wall, obstacles and friction force
    def region_confine(self):
        
        for c in self.Cells:
            for p in c.Particles:
                
                ##wall
                dis = p.position[:, np.newaxis] - self.L    
                dis = np.abs(dis)
                f = np.where(dis <agent_size, f_wall_lim * np.exp((agent_size - dis)/0.08) * dis, 0.) 
                f[:, 1] = -f[:,1]
                f = f.sum(axis = 1)
                
                p.acc += f/p.mass         
                
                ###########Obstacles
                for idx, ob in enumerate(self.Ob):
                    for i in ob:                    
                        dr = p.position - i
                        dis = np.sqrt(np.sum(dr**2))
                        dis_eq = (agent_size + self.Ob_size[idx])/2
                        
                        if dis < dis_eq:
                            f = f_collision_lim * np.exp((dis_eq - dis)/0.08)                        
                        else:
                            f = 0.
                            
                        f = f* dr/dis
                        p.acc += f/p.mass
                
                ######friction force
                
                f = -p.mass/relaxation_time * p.velocity
                p.acc += f/p.mass            
                
               
    ####Loop particles in the same cell
    def loop_cells(self):
        
        for c in self.Cells:            
            l = len(c.Particles)
            
            for i in range(l):
                for j in range(i+1, l):
                    
                    p1 = c.Particles[i]
                    p2 = c.Particles[j]
                    
                    dr = p1.position - p2.position
                    dis = np.sqrt(np.sum(dr**2))
                    
                    if dis <agent_size:
                        f = f_collision_lim * np.exp((agent_size - dis)/0.08)                        
                    else:
                        f = 0.
                        
                    f = f* dr/dis
                    p1.acc += f/p1.mass
                    p2.acc -= f/p2.mass
    
    ####Loop particles in the neighbor cells
    def loop_neighbors(self):
        
        for c in self.Cells:               
            for n in c.Neighbors:
                
                for p1 in c.Particles:
                    for p2 in self.Cells[n].Particles:
                        
                        dr = p1.position - p2.position
                        dis = np.sqrt(np.sum(dr**2))
                        
                        if dis <agent_size:
                            f = f_collision_lim * np.exp((agent_size - dis)/0.08)
                        else:
                            f = 0.
                            
                        f = f* dr/dis
                        p1.acc += f/p1.mass
                        p2.acc -= f/p2.mass                        
    
    ####Reset initial configuration the Continuum cell space
    def reset(self):
        
        self.Exit.clear()
        for e in Exit:            
            self.Exit.append(self.L[:,0] + e * (self.L[:,1] - self.L[:,0]) )
            
        for cell in self.Cells:
            cell.Particles.clear()
            
        self.Number = self.Total
        self.initialize_particles()
        
        true_angle = self.Grid.get_gradient_from_C(self.agent.position[0], 
                                                   self.agent.position[1], 
                                                   self.C_Frames[0])
        true_angle = np.array(true_angle)
        true_angle = true_angle / np.sqrt(np.sum(true_angle ** 2))
      
        
        return (self.agent.position[0], self.agent.position[1],
                true_angle[0], true_angle[1])

    ####Choose random action from action list
    def choose_random_action(self):
        
        action = np.random.choice(len(self.action))
        return action
    
    ####Get the number of particles from neighbor cells
    def Get_Neighbior_Cells(self, position):
        
        position = np.array(position)
        position = np.append(position, self.L[2,0] + 0.5*(self.L[2,1] - self.L[2,0]) )
        index = (position - self.L[:, 0])/ self.d_cells
        index = index.astype('int')
        
        idx = index + neighbor_list        
        valid = (idx < self.n_cells) & (idx >=0)
        mask = np.all(valid, axis = 1)   
        idx = idx[mask]
        
        neighbors = []
       
        for d in range(len(idx)):
            i = idx[d, 0]
            j = idx[d, 1]
            k = idx[d, 2]
            
            N = k * (self.n_cells[0] * self.n_cells[1]) + j * self.n_cells[0] + i            
            neighbors.append(len(self.Cells[N].Particles))
        
        neighbor_cells = np.zeros(9)
        neighbor_cells[mask] = neighbors
        
        return neighbor_cells

    ####Step funtion for agent 0 taking certain action and others taking random actions
    def step(self, action):
        
        reward = self.reward
        done = False
        ext = None
        
        self.Zero_acc()
 
        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()
        
        for c in self.Cells:
            for p in c.Particles:
                
                if p.ID !=0:
                    action = np.random.choice(len(self.action))
                    
                p.acc += 1/relaxation_time * desire_velocity *self.action[action]
        
        self.Integration(1)               
        self.Integration(0)
        self.move_particles()
        
        next_state = (self.agent.position[0], self.agent.position[1],
                      self.agent.velocity[0], self.agent.velocity[1])
        
        for idx, e in enumerate(self.Exit):
            dis = self.agent.position - e
            dis = np.sqrt(np.sum(dis**2))
            if dis < dis_lim:
                done = True
                reward = self.end_reward
                ext = idx
                break
        
        return next_state, reward, done, ext

    def step_continuous(self, action):
        
        reward = self.reward
        done = False
        ext = None
        
        self.Zero_acc()
 
        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()
        
        for c in self.Cells:
            for p in c.Particles:
                
                if p.ID !=0:
                    action = np.random.rand()*2 -1
                    
                action = action * np.pi + np.pi ###-1,1 to 0, 2pi
                angle = np.array([np.cos(action), np.sin(action), 0.], dtype = 'float')
                
                p.acc += 1/relaxation_time * desire_velocity *angle
        
        self.Integration(1)               
        self.Integration(0)
        self.move_particles()
        
        next_state = (self.agent.position[0], self.agent.position[1],
                      self.agent.velocity[0], self.agent.velocity[1])
        
        for idx, e in enumerate(self.Exit):
            dis = self.agent.position - e
            dis = np.sqrt(np.sum(dis**2))
            if dis < dis_lim:
                done = True
                reward = self.end_reward
                ext = idx
                break
        
        return next_state, reward, done, ext

    def step_continuous_local_cell(self, action):
        
        done = False
        ext = None
        
        self.Zero_acc()
 
        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()
        
        for c in self.Cells:
            for p in c.Particles:
                
                if p.ID !=0:
                    action = np.random.rand()*2 -1
                    
                action = action * np.pi + np.pi ###-1,1 to 0, 2pi
                angle = np.array([np.cos(action), np.sin(action), 0.], dtype = 'float')
                
                p.acc += 1/relaxation_time * desire_velocity *angle
                
                if p.ID == 0:
                            
                    true_angle = self.Grid.get_gradient(p.position[0], p.position[1])
                    true_angle = np.array(true_angle)
                    true_angle = true_angle / np.sqrt(np.sum(true_angle ** 2))
                            
                    local = np.matmul(angle[:2], true_angle)       
                    theta = np.arccos(local)
                    
#                    reward = -10*theta / np.pi
                    reward = 10*np.exp(-theta**2)
        
        self.Integration(1)               
        self.Integration(0)
        self.move_particles()
        
        next_state = (self.agent.position[0], self.agent.position[1],
                      self.agent.velocity[0], self.agent.velocity[1])
        
        for idx, e in enumerate(self.Exit):
            dis = self.agent.position - e
            dis = np.sqrt(np.sum(dis**2))
            if dis < dis_lim:
                done = True
                reward = self.end_reward
                ext = idx
                break
        
        return next_state, reward, done, ext

    def step_continuous_moving_source(self, action, t):
        
        done = False
        ext = None
        
        self.Zero_acc()
 
        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()
        
        for c in self.Cells:
            for p in c.Particles:
                
                if p.ID !=0:
                    action = np.random.rand()*2 -1
                    
                action = action * np.pi + np.pi ###-1,1 to 0, 2pi
                angle = np.array([np.cos(action), np.sin(action), 0.], dtype = 'float')
                
                p.acc += 1/relaxation_time * desire_velocity *angle *0.1
                
                if p.ID == 0:
                            
                    i = int(t/self.step_interval)
                    true_angle = self.Grid.get_gradient_from_C(p.position[0], p.position[1], 
                                                               self.C_Frames[i])
                    true_angle = np.array(true_angle)
                    true_length = np.sqrt(np.sum(true_angle ** 2))
                    if true_length != 0:                        
                        true_angle = true_angle / true_length
                            
                    local = np.matmul(angle[:2], true_angle)       
                    theta = np.arccos(local)
                    
#                    reward = -10*theta / np.pi
                    reward = 10*np.exp(-theta**2)
        
        self.Integration(1)               
        self.Integration(0)
        self.move_particles()
        
        
        i = int((t+1)/self.step_interval)
        true_angle = self.Grid.get_gradient_from_C(self.agent.position[0], 
                                                   self.agent.position[1], 
                                                   self.C_Frames[i])
        true_angle = np.array(true_angle)
        true_length = np.sqrt(np.sum(true_angle ** 2))
        if true_length != 0:                        
            true_angle = true_angle / true_length
        
        next_state = (self.agent.position[0], self.agent.position[1],
                      true_angle[0], true_angle[1])
        
        for idx, e in enumerate(self.Exit):
            dis = self.agent.position - e
            dis = np.sqrt(np.sum(dis**2))
            if dis < dis_lim:
                done = True
                reward = self.end_reward
                ext = idx
                break
        
        return next_state, reward, done, ext
    
    def step_continuous_moving_source_all(self, sess, net, t):
        
        done = False
        ext = None
        
        self.Zero_acc()
 
        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()
        
        for c in self.Cells:
            for p in c.Particles:

                i = int(t/self.step_interval)
                true_angle = self.Grid.get_gradient_from_C(p.position[0], p.position[1], 
                                                               self.C_Frames[i])
                true_angle = np.array(true_angle)
                true_length = np.sqrt(np.sum(true_angle ** 2))
                if true_length != 0:                        
                    true_angle = true_angle / true_length

                feed_state = np.array([p.position[0], p.position[1], 
                                       true_angle[0], true_angle[1] ])
                feed_state[:2] =  self.Normalization_XY(feed_state[:2])  
                    
                ###### deterministic policy    
                feed = {net.actor_inputs: feed_state[np.newaxis, :]}                
                action = sess.run(net.action, feed_dict=feed)[0] 
                action = action * np.pi + np.pi ###-1,1 to 0, 2pi
                angle = np.array([np.cos(action), np.sin(action), 0.], dtype = 'float')                
                p.acc += 1/relaxation_time * desire_velocity *angle *0.1
                
                #########Sources
                for idx, e in enumerate(self.Exit):                    
                    dr = p.position - e
                    dis = np.sqrt(np.sum(dr**2))
                    dis_eq = dis_lim 
                        
                    if dis < dis_eq:
                        f = f_collision_lim * np.exp((dis_eq - dis)/0.08)                        
                    else:
                        f = 0.
                            
                    f = f* dr/dis
                    p.acc += f/p.mass
                
                
                local = np.matmul(angle[:2], true_angle)       
                theta = np.arccos(local)
                    
                reward = 10*np.exp(-theta**2)
        
        self.Integration(1)               
        self.Integration(0)
        self.move_particles()
        
        return reward, done, ext
        

if __name__ == '__main__':
    
    ####Test partcle motions
    
    Exit.append( np.array([0.5, 0.5, 0.5]) )     ##Source
    
    a = Cell_Space(0, 10, 0, 10, 0, 2, rcut= 2.0, dt=delta_t, Number=1, source_total_steps = 1000)
    
    Cfg_path = './Cfg'
    if not os.path.isdir(Cfg_path):
        os.mkdir(Cfg_path)    
    
    
    cases = 1
    
    for i in range(cases):
        state = a.reset()
        t = 1
        pathdir = Cfg_path + '/case_' + str(i)
        
        if not os.path.isdir(pathdir):
            os.mkdir(pathdir)
        
        a.save_output(pathdir + '/s.' + str(t))
        
        while t <= a.source_total_steps:   
            
            a.Exit[0][0] += a.vl
            a.Exit[0][1] += a.vl
            
            print("step: {}".format(t))
            action = np.random.rand()*2 -1  
            next_state, reward, done, ext = a.step_continuous_moving_source(action, t)
            
            t +=1
            
            if done:
                a.save_output(pathdir + '/s.' + str(t))                 
                break
            
            state = next_state
            
            
            if t%cfg_save_step ==0:
                a.save_output(pathdir + '/s.' + str(t))
    
       
    