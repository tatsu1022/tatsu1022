import numpy as np
from .two_link_linear_model import TwoLinkLinearModel
from .ur5_two_link_model import UR5TwoLinkModel


class UR5TwoLinkLinearModel(UR5TwoLinkModel):
    def __init__(self, const_z):
        super().__init__()
        self.const_z = const_z
        self.model = TwoLinkLinearModel(self.L1, self.L2, self.const_z)
   
    def backward(self, y):  # y in Frame1 (attached to the joint 0)
        theta_local = self.model.backward(y)
        theta = theta_local[self.ARM_CONFIGURATION]
        theta[0] = np.pi/2 - theta[0]
        theta[1] += np.pi/2
        # print(self.ARM_CONFIGURATION + ' target configuration: {}'.format(theta))
        return theta

    def get_workspace_radius_range(self, theta2_min, theta2_max):
        radius_low, radius_high = super().get_workspace_radius_range(theta2_min, theta2_max)

        horisontal_radius_high = np.sqrt(radius_high**2 - self.const_z**2)
        horisontal_radius_low = radius_low * horisontal_radius_high / radius_high

        return np.array([horisontal_radius_low, horisontal_radius_high])