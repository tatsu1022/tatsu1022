import numpy as np
from .two_link_model import TwoLinkModel


class TwoLinkLinearModel(TwoLinkModel):
    def __init__(self, L1, L2, const_y=0):
        super().__init__(L1, L2)
        self.const_y = const_y

    def backward(self, x):
        # theta1
        # print("local_pos: {}".format([x, self.const_y]))
        val_acos = ( x**2 + self.const_y**2 + self.L1**2 - self.L2**2 ) / ( 2*self.L1 * np.sqrt(x**2 + self.const_y**2) + 1e-6 )
        assert -1 < val_acos < 1, 'x = {}\nval_acos = {}'.format(x, val_acos)

        theta1_pos = np.arccos(val_acos) + np.arctan2(self.const_y, x)
        theta1_neg = -np.arccos(val_acos) + np.arctan2(self.const_y, x)

        # theta2_pos
        theta2_pos = np.arctan2(self.const_y - self.L1*np.sin(theta1_pos), x - self.L1*np.cos(theta1_pos)) - theta1_pos
        # theta2_neg
        theta2_neg = np.arctan2(self.const_y - self.L1*np.sin(theta1_neg), x - self.L1*np.cos(theta1_neg)) - theta1_neg

        return {'positive': np.array([theta1_pos, theta2_pos]), 'negative': np.array([theta1_neg, theta2_neg])}


def test_backward():
    L1 = 0.425
    L2 = 0.3922

    model = TwoLinkLinearModel(L1, L2)

    u_low = max(0, L1-L2) **2
    u_high = (L1+L2) **2

    b = []
    for _ in range(1000):
        pos = np.sqrt(np.random.uniform(u_low, u_high))
        deg = model.backward(pos)

        deg_pos = deg['positive']
        deg_pos[0] = np.pi/2 - deg_pos[0]
        deg_pos[1] += np.pi/2
        print('positive degree: {}'.format(deg_pos))

        deg_neg = deg['negative']
        deg_neg[0] = np.pi/2 - deg_neg[0]
        deg_neg[1] += np.pi/2
        print('negative degree: {}'.format(deg_neg))


def test_max_x():
    L1 = 0.425
    L2 = 0.3922
    y_const = 0.2

    theta_max = np.pi/6

    x_c = L1 * np.cos(np.pi/2 - theta_max)
    y_c = L1 * np.sin(np.pi/2 - theta_max)

    x_max = x_c + np.sqrt(L2**2 - (y_const - y_c)**2)

    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    x = np.arange(x_c-L2, x_c+L2+2*L2/1000, 2*L2/1000)
    y_pos = np.sqrt(L2**2 - (x-x_c)**2) + y_c
    y_neg = -np.sqrt(L2**2 - (x-x_c)**2) + y_c

    ax.plot(x, y_pos, c='r')
    ax.plot(x, y_neg, c='r')
    ax.plot(x, y_const*np.ones_like(x), c='g')
    ax.plot([0, x_c], [0, y_c], c='b')
    ax.plot([x_c, x_max], [y_c, y_const], c='b')
    ax.scatter(0, 0, c='b', marker='o')
    ax.scatter(x_c, y_c, marker='o')
    ax.scatter(x_max, y_const, c='b', marker='o')
    
    ax.set_aspect('equal', adjustable='box')

    plt.show()


if __name__ == '__main__':
    test_backward()
    test_max_x()