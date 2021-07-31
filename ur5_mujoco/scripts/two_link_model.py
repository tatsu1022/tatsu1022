import numpy as np


class TwoLinkModel:
    def __init__(self, L1, L2):
        self.L1 = L1
        self.L2 = L2

    def forward(self, theta1, theta2):
        x = self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2)
        y = self.L1 * np.sin(theta1) + self.L2 * np.sin(theta1 + theta2)
        return np.array([x, y])
    
    def backward(self, x, y):
        # theta1
        # print("local_pos: {}".format([x,y]))
        val_acos = ( x**2 + y**2 + self.L1**2 - self.L2**2 ) / ( 2*self.L1 * np.sqrt(x**2 + y**2) )

        assert -1 < val_acos < 1, 'val_acos = {}'.format(val_acos)

        theta1_pos = np.arccos(val_acos) + np.arctan2(y, x)
        theta1_neg = -np.arccos(val_acos) + np.arctan2(y, x)

        # theta2_pos
        theta2_pos = np.arctan2(y - self.L1*np.sin(theta1_pos), x - self.L1*np.cos(theta1_pos)) - theta1_pos
        # theta2_neg
        theta2_neg = np.arctan2(y - self.L1*np.sin(theta1_neg), x - self.L1*np.cos(theta1_neg)) - theta1_neg

        return {'positive': np.array([theta1_pos, theta2_pos]), 'negative': np.array([theta1_neg, theta2_neg])}

    '''
    def backward2(self, x, y):
        cos_theta2 = ( x**2 + y**2 - self.L1**2 - self.L2**2 ) / ( 2 * self.L1 * self.L2 )

        assert -1 < cos_theta2 < 1, 'cos_theta2 = {}'.format(cos_theta2)
            
        theta2_pos = np.arccos(cos_theta2) 
        theta2_neg = -np.arccos(cos_theta2)

        theta1_pos = np.arctan2(y,x) - np.arctan2(self.L2*np.sin(theta2_pos), (self.L1 + self.L2*np.cos(theta2_pos)))
        theta1_neg = np.arctan2(y,x) - np.arctan2(self.L2*np.sin(theta2_neg), (self.L1 + self.L2*np.cos(theta2_neg)))
     
        return {'positive': np.array([theta1_pos, theta2_pos]), 'negative': np.array([theta1_neg, theta2_neg])}
    '''


def test(theta1, theta2):
    L1 = 0.425
    L2 = 0.3922
    coordinate_shift = 0.0213 + 0.0679

    print('applied degree: {}'.format([theta1, theta2]))

    model = TwoLinkModel(L1, L2)

    theta = [np.pi/2 - theta1, -np.pi/2 + theta2]   # adapt to ur5 coordinate system
    calc_pos = model.forward(*theta)
    # print('calculated position: {}'.format(calc_pos))

    pos = np.concatenate(([0], calc_pos))   # yz plane
    pos[2] += coordinate_shift
    # print('target position: {}'.format(pos))

    deg = model.backward(*calc_pos)
    deg_pos = deg['positive']
    deg_pos[0] = np.pi/2 - deg_pos[0]
    deg_pos[1] += np.pi/2
    print('positive degree: {}'.format(deg_pos))

    deg_neg = deg['negative']
    deg_neg[0] = np.pi/2 - deg_neg[0]
    deg_neg[1] += np.pi/2
    print('negative degree: {}'.format(deg_neg))


def test_backward():
    L1 = 0.425
    L2 = 0.3922

    model = TwoLinkModel(L1, L2)

    from sphere_space import CircleSpace
    space = CircleSpace(L1+L2)

    b = []
    for _ in range(1000):
        pos = space.sample()
        deg = model.backward(*pos)

        deg_pos = deg['positive']
        deg_pos[0] = np.pi/2 - deg_pos[0]
        deg_pos[1] += np.pi/2
        print('positive degree: {}'.format(deg_pos))

        deg_neg = deg['negative']
        deg_neg[0] = np.pi/2 - deg_neg[0]
        deg_neg[1] += np.pi/2
        print('negative degree: {}'.format(deg_neg))




if __name__ == '__main__':
    test_backward()






