import time
import gtsam
import math
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
import numpy as np
import prm
from map import Map
import matplotlib.colors as colors
import matplotlib.lines as mlines
from tqdm import tqdm
from config import *


from gaussian_belief import GaussianBelief




class MeasurementSimplification(object):
    """
    Class representing a belief space planning system.

    Parameters:
    - scenario (str): The scenario for generating landmarks. Default is 'uniform'.
    - num_landmarks (int): The number of landmarks to generate. Default is 1000.
    - map_size (int): The size of the map. Default is 40.
    - prior_mapping (str): The prior mapping scenario. Default is 'boxes'.
    - goal (tuple): The goal location. Default is (32, 0).
    - num_paths (int): The number of paths to generate. Default is 100.
    Attributes:
    - prior (np.array): The prior belief state.
    - actions_random (list): A list of random actions.
    - actions (list): A list of predefined actions.
    - scenario (str): The scenario for generating landmarks.
    - num_landmarks (int): The number of landmarks to generate.
    - map_size (int): The size of the map.
    - landmarks (dict): A dictionary containing information about the landmarks.
    - belief (GaussianBelief): The belief object.
    - fig_num (int): The figure number for plotting.
    """

    def __init__(self):
        """
        Initialize the MeasurementSimplification object acording to config parameters.
        """
        self.prior = PRIOR
        self.actions_random = ACTIONS_RANDOM
        self.actions = ACTIONS
        self.scenario = SCENARIO
        self.num_landmarks = NUM_LANDMARKS
        self.map_size = MAP_SIZE
        self.goal = GOAL
        self.num_paths = NUM_PATHS
        self.landmarks = {}
        self.prior_mapping = PRIOR_MAPPING
        self.belief = GaussianBelief(MOTION_MODEL_NOISE, OBSERVATION_MODEL_NOISE)
        self.belief.add_prior_factor(self.prior, PRIOR_NOISE)
        self.generate_landmarks()
        self.generate_prior()
        self.fig_num = NUM_FIGURES

    def generate_prior(self):
        if self.prior_mapping == 'boxes':
            self.prior_mapping_boxes()
        elif self.prior_mapping == 'line':
            self.prior_mapping_line()

    def generate_random_paths(self):
        """
        Generates random paths.
        """
        paths = []
        plt.rcParams['font.size'] = 14
        map = Map(40, 40)
        map.show()
        curr_mean = self.belief.get_curr_mean()
        map.start = (curr_mean[0], curr_mean[1])
        map.end = self.goal

        if self.prior_mapping == 'boxes':
            x = np.linspace(10, 18, 100)
            y = np.linspace(25, 32, 100)  
            X, Y = np.meshgrid(x, y)
            obstacle_coords = np.vstack((X.flatten(), Y.flatten())).T
            map.obstacle = obstacle_coords
            X, Y = np.meshgrid(x+12, y)
            obstacle_coords = np.vstack((X.flatten(), Y.flatten())).T
            map.obstacle = obstacle_coords
            X, Y = np.meshgrid(x, y-18)
            obstacle_coords = np.vstack((X.flatten(), Y.flatten())).T
            map.obstacle = obstacle_coords
            X, Y = np.meshgrid(x+12, y-18)
            obstacle_coords = np.vstack((X.flatten(), Y.flatten())).T
            map.obstacle = obstacle_coords

        elif self.prior_mapping == 'line':
            map.obstacle = [[-5,-5],[-5,-5]] 

        for _ in range(self.num_paths):
            map.path, roadmap = prm.prm_planning(map, display=False, return_roadmap=True)
            paths.append(prm.get_actions(map.path))
        return paths,map

    def generate_landmarks(self):
        if self.scenario == 'uniform':
            self.generate_uniform_landmarks()
        elif self.scenario == 'clustered':
            self.generate_clustered_landmarks()
        elif self.scenario == 'random':
            self.generate_random_landmarks()
        elif self.scenario == 'linear':
            self.generate_linear_landmarks()
        elif self.scenario == 'grid':
            self.generate_grid_landmarks()
        elif self.scenario == 'corridor':
            self.generate_corridor_landmarks()
        elif self.scenario == 'centralized':
            self.generate_centralized_landmarks()
        elif self.scenario == 'sparse':
            self.generate_sparse_landmarks()
        elif self.scenario == 'radial':
            self.generate_radial_landmarks()

    def generate_uniform_landmarks(self):
        for i in range(1, self.num_landmarks + 1):
            x = np.random.random_sample() * self.map_size
            y = np.random.random_sample() * self.map_size
            self.landmarks.update({
                i: {'type': 'triangle', 'marker': '^', 'size': 50, 'color': 'blue', 'pose': np.array([x, y])}
            })

    def generate_clustered_landmarks(self):
        num_clusters = 8
        landmarks_per_cluster = self.num_landmarks // num_clusters

        cluster_centers = np.random.uniform(0, self.map_size, size=(num_clusters, 2))

        for cluster_id, center in enumerate(cluster_centers):
            for i in range(1, landmarks_per_cluster + 1):
                x = center[0] + np.random.normal(0, self.map_size / 20)
                y = center[1] + np.random.normal(0, self.map_size / 20)
                self.landmarks.update({
                    cluster_id * landmarks_per_cluster + i: {'type': 'triangle', 'marker': '^', 'size': 50,
                                                            'color': 'red', 'pose': np.array([x, y])}
                })

    def generate_random_landmarks(self):
        for i in range(1, self.num_landmarks + 1):
            x = np.random.rand() * self.map_size
            y = np.random.rand() * self.map_size
            self.landmarks.update({
                i: {'type': 'triangle', 'marker': '^', 'size': 50, 'color': 'green', 'pose': np.array([x, y])}
            })

    def generate_linear_landmarks(self):
        slope = np.random.rand() * 2 - 1  # Random slope between -1 and 1
        intercept = np.random.rand() * self.map_size
        x_values = np.random.rand(self.num_landmarks) * self.map_size
        y_values = slope * x_values + intercept
        self.landmarks = {i: {'type': 'triangle', 'marker': '^', 'size': 50, 'color': 'orange',
                            'pose': np.array([x_values[i], y_values[i]])} for i in range(0, self.num_landmarks)}

    def generate_grid_landmarks(self):
        grid_size = int(np.sqrt(self.num_landmarks))
        x_values, y_values = np.meshgrid(np.linspace(0, self.map_size, grid_size),
                                        np.linspace(0, self.map_size, grid_size))
        x_values = x_values.flatten()[:self.num_landmarks]
        y_values = y_values.flatten()[:self.num_landmarks]
        self.landmarks = {i: {'type': 'triangle', 'marker': '^', 'size': 50, 'color': 'purple',
                            'pose': np.array([x_values[i], y_values[i]])} for i in range(0, len(x_values) )}

    def generate_corridor_landmarks(self):
        corridor_width = self.map_size / 8
        num_corridors = 5
        num_landmarks_per_corridor = self.num_landmarks // num_corridors

        for corridor_id in range(1, num_corridors + 1):
            for i in range(1, num_landmarks_per_corridor + 1):
                x = np.random.uniform(low=(corridor_id - 1) * self.map_size / num_corridors,
                                    high=corridor_id * self.map_size / num_corridors)
                y = np.random.uniform(low=corridor_width, high=self.map_size - corridor_width)

                # Introduce T-intersections and turns
                if np.random.rand() < 0.3:  # 30% chance for a T-intersection or turn
                    intersection_type = np.random.choice(['T', 'turn'])
                    if intersection_type == 'T':
                        y = np.random.choice([corridor_width, self.map_size - corridor_width])
                    elif intersection_type == 'turn':
                        x = np.random.choice([(corridor_id - 1) * self.map_size / num_corridors,
                                            corridor_id * self.map_size / num_corridors])
                        y = np.random.choice([corridor_width, self.map_size - corridor_width])

                self.landmarks.update({
                    (corridor_id - 1) * num_landmarks_per_corridor + i: {'type': 'triangle', 'marker': '^', 'size': 50,
                                                                        'color': 'cyan', 'pose': np.array([x, y])}
                })

    def generate_centralized_landmarks(self):
        center = np.array([self.map_size / 2, self.map_size / 2])
        for i in range(1, self.num_landmarks + 1):
            angle = np.random.rand() * 2 * np.pi
            radius = np.random.rand() * self.map_size / 4
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            self.landmarks.update({
                i: {'type': 'triangle', 'marker': '^', 'size': 50, 'color': 'magenta', 'pose': np.array([x, y])}
            })

    def generate_sparse_landmarks(self):
        density = 0.1  
        num_sparse_landmarks = int(self.num_landmarks * density)
        x_values = np.random.rand(num_sparse_landmarks) * self.map_size
        y_values = np.random.rand(num_sparse_landmarks) * self.map_size
        self.landmarks = {i: {'type': 'triangle', 'marker': '^', 'size': 50, 'color': 'yellow',
                            'pose': np.array([x_values[i], y_values[i]])} for i in range(1, num_sparse_landmarks)}

    def generate_radial_landmarks(self):
        center = np.array([self.map_size / 2, self.map_size / 2])
        angles = np.linspace(0, 2 * np.pi, self.num_landmarks)
        radii = np.random.rand(self.num_landmarks) * self.map_size / 2
        x_values = center[0] + radii * np.cos(angles)
        y_values = center[1] + radii * np.sin(angles)
        self.landmarks = {i: {'type': 'triangle', 'marker': '^', 'size': 50, 'color': 'brown',
                            'pose': np.array([x_values[i], y_values[i]])} for i in range(1, self.num_landmarks)}

    def prior_mapping(self):
        self.belief.add_odometry(self.actions[4])    
        for i in range(1,10):
            # take action
            self.belief.add_odometry(self.actions[0])
            self.get_observation_to_closest_landmark(self.belief)
        self.belief.add_odometry(self.actions[5])
        self.get_observation_to_closest_landmark(self.belief)
        for i in range(1,40):
            # take action
            self.belief.add_odometry(self.actions[0])
            self.get_observation_to_closest_landmark(self.belief)
        self.belief.add_odometry(self.actions[5])
        self.get_observation_to_closest_landmark(self.belief)
        for i in range(1,7):   
            # take action
            self.belief.add_odometry(self.actions[0])
            self.get_observation_to_closest_landmark(self.belief)
        self.belief.add_odometry(self.actions[4])

    def prior_mapping_line(self):  
        for i in range(1,60):
            # take action
            self.belief.add_odometry(self.actions[0]*0.6)
            self.get_observation_to_closest_landmark(self.belief)
        self.belief.add_odometry(self.actions[5])
        self.belief.add_odometry(self.actions[5])

    def prior_mapping_boxes(self): 
        # straight line 
        for i in range(1,35):
            self.belief.add_odometry(self.actions[0]*0.6)
            self.get_observation_to_closest_landmark(self.belief)
        # turn left
        self.belief.add_odometry(self.actions[5])
        for i in range(1,20):
            self.belief.add_odometry(self.actions[0]*0.6)
            self.get_observation_to_closest_landmark(self.belief)
        # turn right
        self.belief.add_odometry(self.actions[4])
        for i in range(1,25):
            self.belief.add_odometry(self.actions[0]*0.6)
            self.get_observation_to_closest_landmark(self.belief)
        # turn right
        self.belief.add_odometry(self.actions[4])
        for i in range(1,20):
            self.belief.add_odometry(self.actions[0]*0.6)
            self.get_observation_to_closest_landmark(self.belief)
        # turn right
        self.belief.add_odometry(self.actions[4])

    def isam_entropy(self,isam,factors,initials, new_initials):
        
        if not factors:
            factors = self.belief.f_graph
        if not initials:
            initials = self.belief.initials
        start_time = time.time()
        result = isam.update(factors, new_initials)
        update_time = time.time() - start_time 
        gfg = isam.getFactorsUnsafe().linearize(initials)
        bn = gfg.eliminateSequential()
        start_time = time.time()
        newLogDetR = bn.logDeterminant()
        logdet_time = time.time() - start_time
        total_time = update_time + logdet_time
        return total_time

    def evaluate_path(self, index, path, prior, marginals, belief,isam):
        posterior_belief = self.copy_belief(belief)
        propogated_belief = self.copy_belief(belief)
        new_key = []
        for action in path:
            posterior_belief.add_odometry(action)
            propogated_belief.add_odometry(action)
            new_key.append(gtsam.symbol('x', posterior_belief.fg_pose_idx-1))
            self.get_observation_to_closest_landmark(posterior_belief, future=True)
        # get bounds
        horizon = len(path)
        N = 3*horizon
        logdet_prior = np.linalg.slogdet(prior)[0]*np.linalg.slogdet(prior)[1]
        propogated_entropy = self.entropy(propogated_belief)
        number_of_factors_old = self.belief.f_graph.size()
        number_of_factors_new = posterior_belief.f_graph.size() 
        total_numer_of_factors = number_of_factors_new - number_of_factors_old
        if total_numer_of_factors == horizon:
            print(f'Path #{index} has no new factors for simplification, skipping')
            return np.inf, np.inf, 0, 0, 0, 0, 0
            
        lb, ub, bounds_time, isam_time, keys_cov =  self.bounds_via_ramdl(marginals, prior,propogated_entropy, posterior_belief,total_numer_of_factors, \
                                                    N, new_key,isam, lemma = 1)
        keys = self.collect_keys(posterior_belief.f_graph, range(self.belief.f_graph.size(), posterior_belief.f_graph.size()))
        for key in new_key:
            keys.remove(key)
        jacobian, jac_overhead = self.collect_jacobian(posterior_belief, range(self.belief.f_graph.size(), posterior_belief.f_graph.size()), gtsam.NonlinearFactorGraph()) 
        det, reward_time = self.det_via_ramdl(marginals, logdet_prior, jacobian, keys, N, lemma = 1)
        actual_reward = 0.5*((prior.shape[0]+N)*np.log(2*np.pi*np.e)-det)
        #candidates.append({'actual': actual_reward, 'lb': lb, 'ub': ub, 'bounds_time': bounds_time, 'reward_time':reward_time, 'path':path})
        return lb,ub, bounds_time, actual_reward, reward_time+jac_overhead, isam_time, keys_cov
    
    def cum_reward(self, path, prior_belief):
        cum_reward = 0
        for action in path:
            prior_belief.add_odometry(action)
            self.get_observation_to_closest_landmark(prior_belief, future=False)
            cum_reward -= self.entropy(prior_belief)
        return cum_reward
    
    def bound_via_factors(self, belief):    
        belief_p = self.copy_belief()
        belief_s = self.copy_belief()
        belief_nots = self.copy_belief()
        index_list = [n for n in range(belief_p.f_graph.size(),belief.f_graph.size())]
        if len(index_list)==0:
            return 0,0
        motion_factor = index_list.pop(0)
        half = len(index_list)//2
        index_s = [motion_factor] + index_list[0:half] 
        index_nots = [motion_factor] + index_list[half:] 
        self.copy_factors(belief, belief_s, index_s)
        self.copy_factors(belief, belief_nots, index_nots)
        self.copy_factors(belief, belief_p, [motion_factor])
        h_s = self.entropy(belief_s)
        h_nots = self.entropy(belief_nots)
        h_p = self.entropy(belief_p)
        h_post = self.entropy(belief)
        lb_approx = 2*h_s-h_p
        lb = h_s+h_nots-h_p
        ub = h_s
        return lb, lb_approx, ub

    def bounds_via_ramdl(self,marginals, prior, propogated_entropy, belief,total_numer_of_factors, N, new_keys,isam, lemma = None):
        
        fg_s = gtsam.gtsam.NonlinearFactorGraph()
        fg_nots = gtsam.gtsam.NonlinearFactorGraph()
        # make a list of factors to copy
        index_list = sorted([belief.f_graph.size()-n for n in range(1, total_numer_of_factors)])

        dif = self.subtract_graphs(self.belief.f_graph, belief.f_graph)
        new_initials = gtsam.gtsam.Values()
        for key in new_keys:
            new_initials.insert(key, belief.initials.atPose2(key))
        isam_time = self.isam_entropy(isam,dif,belief.initials,new_initials)
        factors = self.split_factors_by_type(dif)
        motion_factors = factors[gtsam.gtsam.BetweenFactorPose2]
        observation_factors = factors[gtsam.gtsam.BearingRangeFactor2D]

        index_list = list(range(observation_factors.size()))
        # devide factor list to s and nots
        half = len(index_list)//2
        index_s = index_list[0:half] 
        index_nots = index_list[half:] 

        fg_s.push_back(motion_factors)
        fg_nots.push_back(motion_factors)
        for index in index_s:
            fg_s.push_back(observation_factors.at(index))
        for index in index_nots:
            fg_nots.push_back(observation_factors.at(index))

        # collect jacobians
        jac_start = time.time()
        jacobian_s = fg_s.linearize(belief.initials).jacobian()[0]
        jacobian_nots = fg_nots.linearize(belief.initials).jacobian()[0]
        jac_overhead = time.time() - jac_start
        # collect keys
        keys_s = self.collect_keys(fg_s, range(fg_s.size()))
        keys_nots = self.collect_keys(fg_nots, range(fg_nots.size()))
        keys_s = [k for k in  fg_s.keyVector()]
        keys_nots = [k for k in  fg_nots.keyVector()]

        for key in new_keys:    
            keys_s.remove(key)
            keys_nots.remove(key)
        logdet_prior = np.linalg.slogdet(prior)[0]*np.linalg.slogdet(prior)[1]
        keys_cov = set(keys_s+keys_nots)
        det_s, time_s = self.det_via_ramdl(marginals, logdet_prior, jacobian_s, keys_s, N, lemma)

        det_nots, time_nots = self.det_via_ramdl(marginals, logdet_prior, jacobian_nots, keys_nots, N, lemma)
        total_time = time_s+time_nots+jac_overhead
        dim = prior.shape[0]+N
        h_s = 0.5*(dim*np.log(2*np.pi*np.e)-(det_s))
        h_nots = 0.5*(dim*np.log(2*np.pi*np.e)-(det_nots))
        lb = h_s+h_nots-propogated_entropy
        ub = h_s
        return lb, ub, total_time, isam_time, keys_cov
    
    
    def hierarchy_bounds(self,marginals, prior, propogated_entropy, belief, N, new_keys, lemma = None, depth=1):
        contin = False
        dif = self.subtract_graphs(self.belief.f_graph, belief.f_graph)
        factors = self.split_factors_by_type(dif)
        motion_factors = factors[gtsam.gtsam.BetweenFactorPose2]
        observation_factors = factors[gtsam.gtsam.BearingRangeFactor2D]
        main_list = np.array(range(observation_factors.size()))
        index_list = self.split_list(main_list , depth)
        graph_list = []
        for i in range(len(index_list)):
            graph_list.append(gtsam.NonlinearFactorGraph())
        for i, i_time  in enumerate(index_list):
            if len(i_time)>1:
                contin = True
            graph_list[i].push_back(motion_factors)
            for index in i_time:
                graph_list[i].push_back(observation_factors.at(index))

        # collect jacobians
        jacobian_list = []
        for i in range(len(graph_list)):
            jacobian_list.append(graph_list[i].linearize(belief.initials).jacobian()[0])
        
        # collect keys
        keys_list = []
        for i in range(len(graph_list)):
            keys_list.append([k for k in  graph_list[i].keyVector()])
            for key in new_keys:    
                keys_list[i].remove(key)
        
        det_list = []
        time_list = []
        logdet_prior = np.linalg.slogdet(prior)[0]*np.linalg.slogdet(prior)[1]
        for i in range(len(graph_list)):
            det, time = self.det_via_ramdl(marginals, logdet_prior, jacobian_list[i], keys_list[i], N, lemma)
            det_list.append(det)
            time_list.append(time)
        total_time = sum(time_list)
        dim = prior.shape[0]+N
        h_list = []
        for i in range(len(graph_list)):
            h_list.append(0.5*(dim*np.log(2*np.pi*np.e)-(det_list[i])))
        lb = sum(h_list)-propogated_entropy*(len(graph_list)-1)
        ub = min(h_list)
        return lb, ub, total_time, contin 


    def split_list(self,lst, depth):
        if depth == 0:
            return [lst]
        if len(lst) == 1:
            return [lst]
        else:
            mid = len(lst) // 2
            left = lst[:mid]
            right = lst[mid:]
            return self.split_list(left, depth-1) + self.split_list(right, depth-1)

    @staticmethod
    def subtract_graphs(graph_prev, graph_curr):
        delta_factors = {}
        for idx in range(graph_curr.size()):
            factor = graph_curr.at(idx)
            delta_factors[tuple(factor.keys())] = factor
        for idx in range(graph_prev.size()):
            factor = graph_prev.at(idx)
            if tuple(factor.keys()) in delta_factors:
                del delta_factors[tuple(factor.keys())]
        delta_graph = gtsam.NonlinearFactorGraph()
        for delta_factor in delta_factors.values():
            delta_graph.push_back(delta_factor)
        return delta_graph
    
    @staticmethod
    def split_factors_by_type(graph):
        result = {}
        size = graph.size()
        factor_indexes = range(size)
        for idx in factor_indexes:
            factor = graph.at(idx)
            factor_type = type(factor)
            if factor_type in result:
                result[factor_type].add(factor)
            else:
                result[factor_type] = gtsam.NonlinearFactorGraph()
                result[factor_type].add(factor)
        return result
    
    def bounds_convergence_s(self,marginals, prior, propogated_entropy, belief,total_numer_of_factors, N, new_keys, lemma = None):


        # make a list of factors to copy
        index_list = sorted([belief.f_graph.size()-n for n in range(1, total_numer_of_factors)])

        dif = self.subtract_graphs(self.belief.f_graph, belief.f_graph)
        factors = self.split_factors_by_type(dif)
        motion_factors = factors[gtsam.gtsam.BetweenFactorPose2]
        observation_factors = factors[gtsam.gtsam.BearingRangeFactor2D]


        index_list = list(range(observation_factors.size()))
        # devide factor list to s and nots
        half = 1
        index_s = index_list[0:half] 
        index_nots = index_list[half:] 
        
        ub_list = []
        lb_list = []
        logdet_prior = np.linalg.slogdet(prior)[1]*np.linalg.slogdet(prior)[0]
        while len(index_s)>0 and len(index_nots)>0:
            fg_s = gtsam.gtsam.NonlinearFactorGraph()
            fg_nots = gtsam.gtsam.NonlinearFactorGraph()
            fg_s.push_back(motion_factors)
            fg_nots.push_back(motion_factors)
            for index in index_s:
                fg_s.push_back(observation_factors.at(index))
            for index in index_nots:
                fg_nots.push_back(observation_factors.at(index))
            jacobian_s = fg_s.linearize(belief.initials).jacobian()[0]
            jacobian_nots = fg_nots.linearize(belief.initials).jacobian()[0]
            keys_s = [k for k in  fg_s.keyVector()]
            keys_nots = [k for k in  fg_nots.keyVector()]

            for key in new_keys:    
                keys_s.remove(key)
                keys_nots.remove(key)
            det_s, time_s = self.det_via_ramdl(marginals, logdet_prior, jacobian_s, keys_s, N, lemma)
            det_nots, time_nots = self.det_via_ramdl(marginals, logdet_prior, jacobian_nots, keys_nots, N, lemma)
            dim = prior.shape[0]+N
            h_s = 0.5*(dim*np.log(2*np.pi*np.e)-(det_s))
            h_nots = 0.5*(dim*np.log(2*np.pi*np.e)-(det_nots))
            lb = h_s+h_nots-propogated_entropy
            ub = h_s
            ub_list.append(ub)
            lb_list.append(lb) 
            index_s.append(index_nots.pop())
        return lb_list, ub_list

    def collect_keys(self, f_graph, indeces):
        keys = set()
        for index in indeces:
            key = (f_graph.at(index).keys())
            for k in key:
                keys.add(k)
        return list(keys)

    def save_results(self, results, file_names, res_path, serialization_required = True):
        for res, fname in zip(results, file_names):
            with open(res_path + fname + ".txt", 'w') as f:
                if serialization_required:
                    f.write(res.serialize())
                else:
                    f.write(res)

    def collect_jacobian(self, belief, indeces, f_graph):
        for index in indeces:
            f_graph.add(belief.f_graph.at(index))
        start = time.time()
        jacobian = f_graph.linearize(belief.initials).jacobian()[0]
        jac_time = time.time()-start
        return jacobian, jac_time

    def det_via_ramdl(self,marginals, logdet_prior, jacobian, keys, N, lemma = None):
        """
        parameter: marginals: marginals object
        parameter: prior: prior information matrix
        parameter: jacobian: jacobian of new factors
        parameter: key: keys vector of old states
        parameter: N: number of new states
        return: determinant of the posterior belief
        """
        joint_marginal = marginals.jointMarginalCovariance(keys)
        
        marg_cov_mat = joint_marginal.fullMatrix()
        # devide jacobian to A_old, A_new
        A_new = jacobian[:,-N:]
        A_old = jacobian[:,:-N]
        if lemma == 1 or lemma == None:
            # calculating delta (according to lemma 1)
            delta = np.eye(A_old.shape[0])+A_old@marg_cov_mat@A_old.T

            # regular det is exploding to inf, so we use slogdet
            # slogdet is very slow so det is used for timing
            start = time.time()
            det_d = np.linalg.det(delta)
            time_d = time.time()-start
            start = time.time()
            inv_delta = np.linalg.inv(delta)
            time_inv = time.time()-start
            start = time.time()
            det_a = np.linalg.det(A_new.T@inv_delta@A_new)
            time_a = time.time()-start
            total_time = time_d+time_inv+time_a
            # calculating determinant
            det_d = np.linalg.slogdet(delta)
            det_a = np.linalg.slogdet(A_new.T@np.linalg.inv(delta)@A_new)
            det = (logdet_prior+det_d[0]*det_d[1]+det_a[0]*det_a[1]) 
            if lemma == 1:
                return det, total_time

        # calculating delta (according to lemma 2)
        B_old = A_old[:3,:3]
        B_new = A_new[:3,:]
        D_new = A_new[3:,:]
        delta1 = np.eye(B_old.shape[0])+B_old@marg_cov_mat@B_old.T
        
        # calculating determinant
        det1 = np.linalg.det(prior)*np.linalg.det(delta1)*np.linalg.det(B_new.T@np.linalg.inv(delta1)@B_new + D_new.T@D_new)
        
        if lemma == 2:
            return det1, total_time
        
        assert np.isclose(det, det1) ,"det1: {} det2: {}".format(det1, det)

        return det, total_time
  

    def get_bounds_R(self, jacobian, R):
        '''
        parameter: jacobian: jacobian matrix of the posterior belief
        parameter: R: square root of  prior jacobian matrix
        '''
        N = R.shape[1] #number of prior states
        N_new = jacobian.shape[1]-N #number of new states
        new_rows = jacobian.shape[0]-R.shape[0] #number of new rows
        s = int((new_rows-3)/2)
        motion_factor = jacobian[N:N+3]
        measurement_factors = jacobian[N+3:]
        R_pad = np.pad(R, ((0,0),(0,N_new)), 'constant')
        R_aug_s = np.concatenate((R_pad,jacobian[N:N+s+3]), axis=0) # augmenting with new measurements
        R_s = np.linalg.qr(R_aug_s, mode='r') #factorizing new measurements
        det_R_s = np.linalg.slogdet(((R_s).transpose()@R_s))[1] #det of new info matrix

        R_aug_not_s = np.concatenate((R_pad,jacobian[N+s+3:]), axis=0)# augmenting with new measurements
        R_not_s = np.linalg.qr(R_aug_not_s, mode='r') #factorizing new measurements
        det_R_not_s = np.linalg.slogdet(((R_not_s).transpose()@R_not_s))[1] #det of new info matrix

        prior = np.matmul(np.matrix(R).transpose(),np.matrix(R)) #prior information matrix
        entropy_prior = 0.5*(N*np.log(2*np.pi*np.e)-(np.linalg.slogdet(prior)[1]))
        entropy_s = 0.5*(N_new*np.log(2*np.pi*np.e)-det_R_s)
        entropy_not_s = 0.5*(N_new*np.log(2*np.pi*np.e)-det_R_not_s)
        upperb = entropy_s
        lb = entropy_s+entropy_not_s-entropy_prior
        ub_s = [N+s]
        ub_values = [upperb]
        for i in range(N+s+2,jacobian.shape[0],2):
            R_aug_s = np.concatenate((R_s,jacobian[i:i+2]), axis=0)# augmenting with new measurements
            R_s = np.linalg.qr(R_aug_s, mode='r') #factorizing new measurements
            det_R_s = np.linalg.slogdet(((R_s).transpose()@R_s))[1] #det of new info matrix
            ub_values.append(0.5*(N*np.log(2*np.pi*np.e)-det_R_s))
            ub_s.append(i)
        ub = []
        ub.append(ub_s)
        ub.append(ub_values)
        return lb, ub #lower bound, upper bound

    def entropy(self, belief):
        '''
        calculates the entropy of a belief
        '''
        ord = belief.pose_landmark_ordering()
        jacobian = belief.get_jacobians(ord)[0]
        logdet = np.linalg.slogdet(np.matmul(np.matrix(jacobian).transpose(),np.matrix(jacobian)))[1]
        N = jacobian.shape[1]
        return 0.5*(N*np.log(2*np.pi*np.e)-logdet)

    def copy_factors(self,belief1, belief2, index_list):
        for index in index_list:
            factor = belief1.f_graph.at(index) #get factor according to index
            keys = factor.keys() #get keys of factor
            belief2.f_graph.add(factor) #add factor to belief2
            for key in keys:
                if key not in belief2.initials.keys(): #check if prior exists
                    initial = belief1.initials.atPose2(key) #get prior value of key
                    belief2.initials.insert(key, initial) #add prior value to belief2

    def givens(self, A):
    # Initialization of the orthogonal matrix Q and the upper triangular matrix R
        n, m = A.shape
        Q = np.eye(n)
        R = np.copy(A)

        rows, cols = np.tril_indices(n, -1, m)
        for (row, col) in zip(rows, cols):
        # If the subdiagonal element is nonzero, then compute the nonzero 
        # components of the rotation matrix
            if R[row, col] != 0:
                r = np.sqrt(R[col, col]**2 + R[row, col]**2)
                c, s = R[col, col]/r, -R[row, col]/r

            # The rotation matrix is highly discharged, so it makes no sense 
            # to calculate the total matrix product
                R[col], R[row] = R[col]*c + R[row]*(-s), R[col]*s + R[row]*c
                Q[:, col], Q[:, row] = Q[:, col]*c + Q[:, row]*(-s), Q[:, col]*s + Q[:, row]*c

        return Q[:, :m], R[:m]

    
    def get_optimal_action(self):
        optimal_action = None
        j = np.inf
        
        for action in self.actions:
            # copy belief 
            belief_copy = self.copy_belief()
            
            # take action
            belief_copy.add_odometry(action)
            
            # get relative pose measurement to closest landmark
            z, bearing, closest_landmark_id = self.get_observation_to_closest_landmark(belief_copy)
            belief_copy.add_landmark(z, bearing, closest_landmark_id)
            
            # do inference
            mean, cov = belief_copy.inference()
            
            # calc objective/cost function - in this simple case minimize determinant of posterior covariance
            if np.linalg.det(cov) < j:
                optimal_action = action
                j = np.linalg.det(cov)
                
        return optimal_action


    def copy_belief(self, belief):
        belief_copy = GaussianBelief(MOTION_MODEL_NOISE, OBSERVATION_MODEL_NOISE)
        belief_copy.f_graph = belief.f_graph.clone()
        belief_copy.initials.insert(belief.initials)
        belief_copy.fg_pose_idx = belief.fg_pose_idx
        return belief_copy

    def pad(array, reference_shape, offsets):
        # Create an array of zeros with the reference shape
        result = np.zeros(reference_shape)
        # Create a list of slices from offset to offset + shape in each dimension
        insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
        # Insert the array in the result at the specified offsets
        result[insertHere] = array
        return result


    def get_observation_to_closest_landmark(self, belief, future=False):
        # get curr mean and find closest landmark
        # if future is true, observation are taken in relation to the prior (planning)
        curr_mean = belief.get_curr_mean()
        noise = np.random.multivariate_normal([0,0],OBSERVATION_MODEL_NOISE.covariance(),1) #measurement noise    
        min_dist = 3
        for landmark in self.landmarks.keys():
            dist = np.sqrt(np.sum(np.abs(self.landmarks[landmark]['pose'] - curr_mean[0:2])**2,axis=-1))
            if dist < min_dist:
                landmark_id = landmark
                rel_pos = self.landmarks[landmark_id]['pose'] - curr_mean[0:2]
                z = rel_pos + noise 
                bearing = np.math.atan2(z[0][1],z[0][0])
                belief.add_landmark(z, bearing, landmark_id, future)  
    
    def plot(self, plot_lm=False):
        plt.figure(self.fig_num)
        fig = plt.gcf()
        ax = fig.gca()
        
        # plot landmarks
        if plot_lm:
            for landmark in self.landmarks:
                ax.scatter(self.landmarks[landmark]['pose'][0], 
                        self.landmarks[landmark]['pose'][1], 
                        marker=self.landmarks[landmark]['marker'], s=self.landmarks[landmark]['size'], c=self.landmarks[landmark]['color'], alpha=0.5)
        
        # plot belief
        marginals = gtsam.Marginals(self.belief.f_graph, self.belief.initials)
        for i in range(0,self.belief.fg_pose_idx,5):
            pose_symbol = gtsam.symbol('x', i)
            east_north = gtsam.Pose2(self.belief.initials.atPose2(pose_symbol).x(), 
                                     self.belief.initials.atPose2(pose_symbol).y(), 
                                     self.belief.initials.atPose2(pose_symbol).theta())
            sigmas = np.diag(marginals.marginalCovariance(pose_symbol))
            east_north_cov = 0.05 * np.diag(np.array([sigmas[0],sigmas[1],sigmas[2]]))
            gtsam_plot.plot_pose2(fig.number, east_north, 1.5, east_north_cov)
            
        ax.set_xlim(-10, 50)
        ax.set_ylim(-10, 50)
        plt.xlabel('east')
        plt.ylabel('north')
        plt.show()


    def plot_observations(self, values=None):
        """
        Plots the observations on a 2D graph.

        Args:
            values (Optional[gtsam.Values]): The values representing the poses and landmarks. If not provided, the initial values from the belief will be used.

        Returns:
            None
        """
        plt.figure(self.fig_num)
        fig = plt.gcf()
        ax = fig.gca()
        if values is None:
            values = self.belief.initials
        # plot landmarks
        cmap_g = plt.get_cmap('plasma')
        truncated_reds = self.truncate_colormap(cmap_g, 0.0, 0.7)
        observations = self.split_factors_by_type(self.belief.f_graph)[gtsam.BearingRangeFactor2D]
        
        landmark_x = []
        landmark_y = []
        pose_x = []
        pose_y = []
        for index in range(observations.size()):
            pose_key, landmark_key = observations.at(index).keys()
            pose = values.atPose2(pose_key)
            landmark_pos = values.atPoint2(landmark_key)
            landmark_x.append(landmark_pos[0])
            landmark_y.append(landmark_pos[1])
            pose_x.append(pose.x())
            pose_y.append(pose.y())
        weights_l = np.arange(len(landmark_x))
        ax.scatter(landmark_x,landmark_y,c=weights_l, cmap=truncated_reds,marker='x',s=10,alpha=0.25)
        ax.scatter(pose_x[0],pose_y[0],c='g',marker='>',s=150)
        ax.plot(pose_x,pose_y,color='g',linewidth=3,alpha=0.5)
        plt.xlabel('east')
        plt.ylabel('north')
        legend_elements = [mlines.Line2D([], [], color='g',linewidth=3,alpha=0.5, label='Trajectory'),
                    mlines.Line2D([], [],color='purple',alpha=0.4, label='Possible Paths'),
                    mlines.Line2D([], [], color=[0.33, 0.1, 0.64], marker='x', linestyle='None',
                markersize=4, label='Observed Landmarks')]
        plt.legend(handles=legend_elements, fontsize=13,loc='upper right')

    def plot_bearing_range_factor(self,factor, values,color='r--'):
        # Get the keys for the factor and extract the pose and landmark values
        pose_key, landmark_key = factor.keys()
        pose = values.atPose2(pose_key)

        # Get the range and bearing from the factor
        range_val = factor.measured().range()
        bearing_val = factor.measured().bearing().theta()

        # Rotate the bearing vector by the pose orientation
        bearing_vec = gtsam.Point2(math.cos(bearing_val), math.sin(bearing_val))
        bearing_vec_rotated = pose.rotation().rotate(bearing_vec)

        # Calculate the position of the measurement endpoint
        x_meas = pose.x() + range_val * bearing_vec_rotated[0]
        y_meas = pose.y() + range_val * bearing_vec_rotated[1]

        # Plot the range-bearing measurement as a line
        plt.plot([pose.x(), x_meas], [pose.y(), y_meas], color)

        # Label the plot
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Bearing-Range Measurement')
        plt.axis('equal')
        
    def plot_landmark_location(self,factor, values):
        # Get the keys for the factor and extract the pose and landmark values
        pose_key, landmark_key = factor.keys()
        pose = values.atPose2(pose_key)
        landmark_pos = values.atPoint2(landmark_key)

        plt.scatter(landmark_pos[0],landmark_pos[1],c='r',marker='x',s=10)
        plt.scatter(pose.x(),pose.y(),c='b',marker='o',s=10)

        # Label the plot
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Bearing-Range Measurement')
        plt.axis('equal')

    def time_covariance_recovery(self):
        keys_cov = [k for k in  self.belief.f_graph.keyVector()]
        marginals = gtsam.Marginals(self.belief.f_graph, self.belief.initials)
        cov_time = time.time()
        joint_marginal = marginals.jointMarginalCovariance(keys_cov).fullMatrix()
        cov_time = time.time() - cov_time
        return cov_time

    @staticmethod
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    
    def plot_bounds(self, sorted_candidates):
        plt.figure()
        plt.rcParams['font.size'] = 16
        for i, path in enumerate(sorted_candidates):
            plt.plot(i, path['ub'], 'rv', markersize=6)
            plt.plot(i, path['reward'], 'b*', markersize=6)
            plt.plot(i, path['lb'], 'g^', markersize=6)
            
        plt.legend(['ub', 'Entropy', 'lb' ])
        plt.ylabel('Entropy')
        plt.xlabel('Path')
        plt.xticks(np.arange(0,len(sorted_candidates),15))
        plt.xlim([0,len(sorted_candidates)])
        fig = plt.gcf()
        fig.set_size_inches((9, 7), forward=False) 
        fig.savefig(f'bounds_{i}_paths.pdf', dpi=500)
        plt.close()

    def run(self):
        """
        Evaluate a set of randomly generated paths and select the best one based on the entropy of the posterior belief.
        The posterior entorpy is calculated via 3 different methods: iSAM2, rAMDL and Measurement Selection (MS).
        Finally, each method is evaluated in terms of run-time.
        """


        # follow path with re-planning]
        prior = self.belief.get_prior_info_mat()
        marginals = gtsam.Marginals(self.belief.f_graph, self.belief.initials)
        keys_cov_set = set()
        
        paths, map = self.generate_random_paths()     
        self.plot_observations()
        candidates = []
        posterior_belief = self.copy_belief(self.belief)
        curr_mean = posterior_belief.get_curr_mean()
        map.start = (curr_mean[0], curr_mean[1])
        b_time =[]
        r_time = []
        i_time = []
        best_lb = np.inf
        best_path = []
        print(f'***  Evaluating {len(paths)} paths ***')
        for index, path in enumerate(tqdm(paths)):
            isam = gtsam.ISAM2()
            isam.update(posterior_belief.f_graph, posterior_belief.initials)
            lb,ub, bounds_time, actual_reward, reward_time, isam_time, keys_cov = self.evaluate_path(index,path, prior, marginals,posterior_belief,isam)
            if lb==np.inf:
                continue
            candidates.append({ 'path': path, 'reward': actual_reward, 'lb': lb, 'ub': ub})
            b_time.append(bounds_time) 
            r_time.append(reward_time)  
            i_time.append(isam_time)
            keys_cov_set.update(keys_cov) 
            if lb < best_lb:
                best_lb = lb
                best_path = path

        print('***  Evaluating paths is done    ***')

        # print stats
        print ('stats for bounds: ', np.sum(b_time), np.std(b_time))
        print ('stats for reward: ', np.sum(r_time), np.std(r_time))
        print ('stats for isam: ', np.sum(i_time), np.std(i_time))
        cov_time = time.time()
        _ = marginals.jointMarginalCovariance(list(keys_cov_set)).fullMatrix()
        cov_time = time.time() - cov_time
        print("involved covariance recovery time: {}".format(cov_time))

        sorted_candidates = sorted(candidates, key=lambda k: k['reward'], reverse=True)
        self.plot_bounds(sorted_candidates)

        
if __name__ == '__main__':
   bsp = MeasurementSimplification()
   bsp.run()
