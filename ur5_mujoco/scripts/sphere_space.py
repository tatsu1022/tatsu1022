from gym.spaces import Space

import numpy as np


class DiskSpace(Space):
    def __init__(self, radius_low=0, radius_high=1, theta_low=0, theta_high=2*np.pi,
                origin=[0,0], shape=None, dtype=np.float32):
        self.radius_low = radius_low
        self.radius_high = radius_high
        assert theta_low < theta_high
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.origin = origin.copy()

        assert dtype is not None, 'dtype must be explicitly provided.'
        self.dtype = np.dtype(dtype)
        self.shape = shape
        super().__init__(self.shape, self.dtype)

    def sample(self):
        theta = (self.theta_high - self.theta_low) * np.random.rand() + self.theta_low
        u_low = self.radius_low**2
        u_high = self.radius_high**2
        r = np.sqrt(np.random.uniform(u_low, u_high))
        pos = np.array((r*np.cos(theta), r*np.sin(theta)))
        pos += self.origin
        return pos

    def contains(self, pos):
        pos = np.array(pos, copy=True) - self.origin
        r = np.sqrt(pos[0]**2 + pos[1]**2)
        theta = abs(np.arctan2(pos[1], pos[0]))

        ret = (self.radius_low < r < self.radius_high) and (self.theta_low <= theta < self.theta_high)
        return ret

    def get_range(self):
        return np.array([self.radius_low, self.radius_high, self.theta_low, self.theta_high])


class SphereSpace(Space):
    def __init__(self, radius_low=0, radius_high=1, theta_low=0, theta_high=np.pi, 
                phi_low=0, phi_high=2*np.pi, origin=[0,0,0], 
                shape=None, dtype=np.float32):
        self.radius_low = radius_low
        self.radius_high = radius_high
        assert theta_low < theta_high
        self.theta_low = theta_low
        self.theta_high = theta_high
        assert phi_low < phi_high
        self.phi_low = phi_low
        self.phi_high = phi_high
        self.origin = origin.copy()
        
        assert dtype is not None, 'dtype must be explicitly provided.'
        self.dtype = np.dtype(dtype)
        self.shape = shape
        super().__init__(self.shape, self.dtype)

    def sample(self):
        u_low = 1/2 - 1/2 * np.cos(self.theta_low)
        u_high = 1/2 - 1/2 * np.cos(self.theta_high)
        u = (u_high - u_low) * np.random.rand() + u_low        
        cos_theta = -2 * u + 1
        sin_theta = np.sqrt(1 - cos_theta**2)

        phi = (self.phi_high - self.phi_low) * np.random.rand() + self.phi_low

        v_low = self.radius_low**3
        v_high = self.radius_high**3
        r = np.cbrt(np.random.uniform(v_low, v_high))

        pos = np.array((r*sin_theta*np.cos(phi), r*sin_theta*np.sin(phi), r*cos_theta))
        pos += self.origin
        return pos

    def contains(self, p):
        pos = np.array(p, copy=True)
        pos -= self.origin
        r = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        theta = np.arccos(pos[2]/r)
        phi = np.sign(pos[1]) * np.arccos(pos[0] / np.sqrt(pos[0]**2 + pos[1]**2))

        ret = (self.radius_low <= r <= self.radius_high) and (self.phi_low <= abs(phi) < self.phi_high)
        return ret

    def get_range(self):
        return np.array([self.radius_low, self.radius_high, 
                        self.theta_low, self.theta_high,
                        self.phi_low, self.phi_high
                        ])


def test_disk_space():
    import matplotlib.pyplot as plt
    sp = DiskSpace(1,5,-3/4*np.pi,3/4*np.pi)
    
    IN = []
    OUT = []
    b = []
    for i in range(1000):
        sample = sp.sample()
        within = sp.contains(sample)
        if within: IN.append(sample)    
        else: OUT.append(sample)
        b.append(within)
    IN = np.asarray(IN)
    OUT = np.asarray(OUT)
    b = np.all(b)

    print('all points contained within the space?: ' + str(b))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if IN.shape[0] != 0: ax.scatter(IN[:,0], IN[:,1], c='b', marker='.')
    if OUT.shape[0] != 0: ax.scatter(OUT[:,0], OUT[:,1], c='r', marker='.')
    ax.set_aspect('equal', adjustable='box')

    plt.show()

def test_sphere_space(*params):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    sp = SphereSpace(*params) #theta_low=np.pi/6, theta_high=np.pi*5/6, phi_low=-np.pi/4, phi_high=np.pi/4)
    
    IN = []
    OUT = []
    b = []
    for i in range(1000):
        sample = sp.sample()
        within = sp.contains(sample)
        if within: IN.append(sample)    
        else: OUT.append(sample)
        b.append(within)
    IN = np.asarray(IN)
    OUT = np.asarray(OUT)
    b = np.all(b)

    print('all points contained within the space?: ' + str(b))

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    if IN.shape[0] != 0: ax.scatter(IN[:,0], IN[:,1], IN[:,2], s=13, c='b')
    if OUT.shape[0] != 0: ax.scatter(OUT[:,0], OUT[:,1], OUT[:,2], s=13, c='r')
    
    ax.set_aspect('auto')

    plt.show()


if __name__ == '__main__':
    test_disk_space()
    test_sphere_space(4, 6, 0, 1/2*np.pi, -1/4*np.pi, 1/4*np.pi)





