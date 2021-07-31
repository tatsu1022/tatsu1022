from .two_link_model import TwoLinkModel
import numpy as np


class UR5TwoLinkModel:
    def __init__(self):
        self.L1 = 0.425
        self.L2 = 0.3922
        self.ORIGIN = np.sum(([0, 0.0213], [0, 0.0679]), axis=0)    # ignoring x_shift for simplicity
        self.ARM_CONFIGURATION = 'positive' # positive / negative
        self.model = TwoLinkModel(self.L1, self.L2)

    def forward(self, q):   # expecting [q1, q2]
        # adapt to ur5 coordinate system
        theta = q.copy()
        theta[0] = np.pi/2 - theta[0]
        theta[1] -= np.pi/2

        pos_local = self.model.forward(*theta)
        # print('local coordinate position: {}'.format(pos_local))

        pos = pos_local + self.ORIGIN
        # print('base coordinate position: {}'.format(pos))
        return pos
    
    def backward(self, p):  # expecting a point on yz plane
        local_pos = np.array(p, copy=True) - self.ORIGIN
        theta_local = self.model.backward(*local_pos)
        theta = theta_local[self.ARM_CONFIGURATION]
        theta[0] = np.pi/2 - theta[0]
        theta[1] += np.pi/2
        # print(self.ARM_CONFIGURATION + ' target configuration: {}'.format(theta))
        return theta

    def get_workspace_radius_range(self, theta2_min, theta2_max):
        assert theta2_min < theta2_max
        if theta2_min <= np.pi <= theta2_max:
            low = max(0, self.L1 - self.L2)
            high = self.L1 + self.L2
        elif theta2_max < np.pi:
            low = max(0, self.L1 + self.L2 * np.sin(theta2_min))
            high = self.L1 + self.L2 * np.sin(theta2_max)
        else:
            raise ValueError('something went wrong during calculating workspace radius.')
        
        return np.array([low, high])
        


if __name__ == '__main__':
    ur5 = UR5TwoLinkModel()
    q = [-0.55, 0.45]
    pos = ur5.forward(q)
    print('applied configuration: {}'.format(q))
    q_new = ur5.backward(pos)


        
