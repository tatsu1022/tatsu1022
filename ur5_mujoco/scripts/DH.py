import numpy as np

sin = np.sin
cos = np.cos
pi = np.pi


def rotate_around_z(vec, theta):
    T = np.array([  [cos(theta), -sin(theta),  0],
                    [sin(theta),  cos(theta),  0],
                    [         0,           0,  1]
                ])
    return np.dot(T, vec)


class Link:
    def __init__(self, theta, d, alpha, a, driven):
        self.theta = theta
        self.d = d
        self.alpha = alpha
        self.a = a

        self.DH_params = {'theta':self.theta, 'd':self.d, 'alpha':self.alpha, 'a':self.a}

        self.H = self.generate_homogeneous_transformation_matrix()

    def generate_homogeneous_transformation_matrix(self):
        theta, d, alpha, a = self.DH_params.values()

        H = np.array([  [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                        [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                        [         0,             sin(alpha),             cos(alpha),            d],
                        [         0,                      0,                      0,            1]
                    ])
        H = np.around(H, 5)

        return H

    def update_theta(self, q):
        self.theta = self.theta_const + q
        self.H = self.generate_homogeneous_transformation_matrix()
        return self.H


class Chain:
    def __init__(self, links):
        self.links = links.copy()
        self.T = np.eye(4)
        self.k = np.zeros(3)

    def forward(self, q):
        assert len(q) == len(self.links)
        T = np.eye(4)
        for i, link in enumerate(self.links):
            assert isinstance(link, Link)
            H = link.update_theta(q[i])
            T = np.matmul(T, H)
        k = T[:, 3][:3]

        self.T = T.copy()
        self.k = k.copy()
        return k

def main():
    q = [0.314, -0.314, -0.5]

    link0 = Link(pi/2, 0.0213, 0, 0)
    link1 = Link(0, 0.0679, -pi/2, 0)
    link2 = Link(-pi/2, 0.0743, pi, 0.425)
    link3 = Link(-pi/2, 0.0173, -pi/2, 0.3922)

    chain = Chain([link0, link1, link2, link3])

    tcp_position = chain.forward(q)
    print(tcp_position)


if __name__ == '__main__':
    main()  # incomplete
