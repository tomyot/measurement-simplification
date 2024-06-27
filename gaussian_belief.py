from gtsam import gtsam
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
import numpy as np
import time


class GaussianBelief:
    """
    Class implementing a Guassian belief using a gtsam factor graph representation.

    Parameters:
    - motion_model_noise (gtsam.noiseModel): The noise model for the motion model.
    - observation_model_noise (gtsam.noiseModel): The noise model for the observation model.
    Attributes:
    - f_graph (gtsam.NonlinearFactorGraph): The factor graph representing the belief.
    - initials (gtsam.Values): The prior values for the belief states.
    - fg_pose_idx (int): The index of the current pose in the factor graph.
    """
        
    def __init__(self, motion_model_noise, observation_model_noise):
        
        self.f_graph = gtsam.NonlinearFactorGraph()
        self.motion_model_noise = motion_model_noise
        self.observation_model_noise = observation_model_noise
        
        self.initials = gtsam.Values()
        self.fg_pose_idx = 0
            
    def add_prior_factor(self, mean, diagonal_noise):
        pose_symbol = gtsam.symbol('x', self.fg_pose_idx)
        prior_mean = gtsam.Pose2(mean[0], mean[1], mean[2])
        
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(diagonal_noise)
        
        prior = gtsam.PriorFactorPose2(pose_symbol, prior_mean, prior_noise)
        self.f_graph.add(prior)
        self.initials.insert(pose_symbol, prior_mean)

        self.fg_pose_idx += 1
    
    def copy_last_pose_prior_factor(self, belief1, diagonal_noise):
        prior_pose_symbol = gtsam.symbol('x', belief1.fg_pose_idx-1)
        prior_mean = belief1.initials.atPose2(prior_pose_symbol)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(diagonal_noise)
        pose_symbol = gtsam.symbol('x', self.fg_pose_idx)
        prior_factor = gtsam.PriorFactorPose2(pose_symbol, prior_mean, prior_noise)
        self.f_graph.add(prior_factor)
        self.initials.insert(pose_symbol, prior_mean)


    def pose_landmark_ordering(self,keys=None):
        ord = gtsam.Ordering()
        if not keys:
            keys = self.initials.keys()
        for index, key in enumerate(keys):
            if str(gtsam.Symbol(key))[0] == 'x':
                poses = keys[index:]
                landmarks = keys[:index]
                break

        for key in poses:
            ord.push_back(key)

        for key in landmarks:
            ord.push_back(key)

        return ord

    def get_graph_logdet(self,ord=None):
        lfg = self.f_graph.linearize(self.initials)
        if ord:
            bn = lfg.eliminateMultifrontal(ordering=ord)
        else:
            bn = lfg.eliminateMultifrontal()
        return bn.logDeterminant()
    
    def get_R_matrix(self):
        info_mat = self.get_prior_info_mat()
        R = np.linalg.qr(info_mat, mode='r')
        return R


    def get_jacobians(self, ord=None):
        lfg = self.f_graph.linearize(self.initials)
        if ord:
            return lfg.jacobian(ordering=ord)   
        else:
            return lfg.jacobian()

    def get_hessian(self):
        lfg = self.f_graph.linearize(self.initials)
        return lfg.hessian()  

    def get_prior_info_mat(self):
        jacobian = self.get_jacobians()[0]
        prior = np.matmul(np.matrix(jacobian).transpose(),np.matrix(jacobian))
        return prior  

    def get_qr_update(self, jacobian):
        N = jacobian.shape[1]*3
        prior = self.get_prior()
        R = np.linalg.qr(prior, mode='r') #factorizing information matrix
        R_pad = np.pad(R, [(0, N),(0,N)], mode='constant') #padding for 1 new state
        R_aug = np.concatenate((R_pad,jacobian), axis=0)# augmenting with new measurements
        start = time.time()
        R_post = np.linalg.qr(R_aug, mode='r') #factorizing new measurements
        end = time.time()-start
        return R_post, end

    def add_odometry(self, action):
        #notice that action is in local frame
        prev_pose_symbol = gtsam.symbol('x', self.fg_pose_idx - 1)
        next_pose_symbol = gtsam.symbol('x', self.fg_pose_idx)
        
        prev_pose = self.initials.atPose2(prev_pose_symbol)
        gtsam_measurement = gtsam.Pose2(action[0],action[1],action[2])
        next_pose = prev_pose.compose(gtsam_measurement)
        
        factor = gtsam.BetweenFactorPose2(prev_pose_symbol, next_pose_symbol, gtsam_measurement, self.motion_model_noise)
        self.f_graph.add(factor)
        self.initials.insert(next_pose_symbol, next_pose)
        
        self.fg_pose_idx += 1
   
    def add_landmark(self, z, bearing, landmark_id, future):
        curr_pose_symbol = gtsam.symbol('x', self.fg_pose_idx - 1)
        l_symbol = gtsam.symbol('l', landmark_id)
        
        #if landmark symbol does not exist add prior
        if future and l_symbol not in self.initials.keys():
            return

        if l_symbol not in self.initials.keys():
            curr_pose = self.initials.atPose2(curr_pose_symbol)
            l_mean = np.array([curr_pose.x() + z[0][0], curr_pose.y() + z[0][1]])
            self.initials.insert(l_symbol, l_mean)
            
        factor = gtsam.BearingRangeFactor2D(curr_pose_symbol, 
                                            l_symbol, 
                                            gtsam.Rot2(bearing), 
                                            np.linalg.norm(z), self.observation_model_noise)
        self.f_graph.add(factor)
        
    def inference(self):
        curr_pose_symbol = gtsam.symbol('x', self.fg_pose_idx - 1)
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.f_graph, self.initials)
        result = optimizer.optimizeSafely()
        marginals = gtsam.Marginals(self.f_graph, result)
        self.initials = result
        
        return self.initials.atPose2(curr_pose_symbol) , marginals.marginalCovariance(curr_pose_symbol)

    def get_curr_mean(self):
        curr_pose_symbol = gtsam.symbol('x', self.fg_pose_idx - 1)
        gtsam_pose = self.initials.atPose2(curr_pose_symbol)
        return np.array([gtsam_pose.x(), gtsam_pose.y(), gtsam_pose.theta()], dtype=float)
    
    def plot(self):
        marginals = gtsam.Marginals(self.f_graph, self.initials)
        for i in range(self.fg_pose_idx):
            pose_symbol = gtsam.symbol('x', i)
            east_north = gtsam.Pose2(self.initials.atPose2(pose_symbol).x(), 
                                     self.initials.atPose2(pose_symbol).y(), 
                                     self.initials.atPose2(pose_symbol).theta())
            sigmas = np.diag(marginals.marginalCovariance(pose_symbol))
            east_north_cov = np.diag(np.array([sigmas[0],sigmas[1],sigmas[2]]))
            gtsam_plot.plot_pose2(plt.gcf().number, east_north, 1.5, east_north_cov)
    