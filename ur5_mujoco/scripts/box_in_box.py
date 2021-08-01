import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# sample points
rt3 = np.sqrt(3)/2
a = np.array([1, 1])
b = a + [rt3, .5]
c = b + [-.5, rt3]
d = c + [-rt3, -.5]

fig = plt.figure()
ax = plt.subplot(111)

def draw_rect(ax, a,b,c,d):
    l = np.vstack((a,b,c,d,a))
    clist = ['r', 'g', 'b', 'orange', 'r']
    for i in range(len(l)):
        ax.scatter(l[i,0], l[i,1], c=clist[i])

    ax.plot(l[:,0], l[:,1], c='k')

def sort_vertices(a,b,c,d):
    l = np.vstack((a,b,c,d))
    idx = np.argmin(l, axis=0)[1]
    y0 = l[idx]

    rest = np.delete(l, idx, axis=0)
    theta = []
    for p in rest:
        delta = p - y0
        theta.append(np.arctan2(delta[1], delta[0]))
    rest[np.argsort(theta)]
    
    sorted = np.vstack((y0, rest))
    return sorted


vlist = sort_vertices(b,c,d,a)
draw_rect(ax, *vlist)

ax.set_aspect('equal', adjustable='box')
plt.axis([-1,3,-1,3])
ax.spines['top'].set_color('none') 
ax.spines['right'].set_color('none') 
ax.spines['bottom'].set_position(('data', 0)) 
ax.spines['left'].set_position(('data', 0)) 

plt.show()






