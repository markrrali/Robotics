import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#np.cross(x, y) -- cross product
#np.dot(x, y) -- dot product
#sym.Matrix(np.zeros((4, 4)))
def eulerZYZ(phi, th, psi):
    A = transform(rotate('z', psi), np.zeros(3), 1)
    B = transform(rotate('y', th), np.zeros(3), 1)
    C = transform(rotate('z', phi), np.zeros(3), 1)
    return(np.matmul(C, np.matmul(B, A)))

def eulerXYZ(phi, th, psi):
    A = transform(rotate('x', psi), np.zeros(3), 1)
    B = transform(rotate('y', th), np.zeros(3), 1)
    C = transform(rotate('z', phi), np.zeros(3), 1)
    return(np.matmul(C, np.matmul(B, A)))

def eulerRPY(phi, th, psi):
    A = transform(rotate('x', psi), np.zeros(3), 1)
    B = transform(rotate('y', th), np.zeros(3), 1)
    C = transform(rotate('z', phi), np.zeros(3), 1)
    return(np.matmul(C, np.matmul(B, A)))
    
#angle between two 2d vectors
def find_angle(v, u):
    return(math.acos(np.linalg.norm(v)*np.linalg.norm(u)/np.dot(v, u)))

#Given Rotation, translation and scaling, create a transfomation matrix
def transform(R, t, s):
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    T[3, 3] = s
    T[3, 0:3] = np.array([0, 0, 0])
    return(T)

#inverse a transofmarion matrix
def transform_inv(T):
    TI = np.zeros((T.shape))
    R = T[0:3, 0:3].transpose()
    t = np.matmul(-R, T[0:3, 3])
    s = 1
    return(transform(R, t, s))

#Rotate by an = angle around axis ax
def rotate(ax, an):
    #ax = x or y or z and an is the angle
    M = np.identity(3)
    a = math.radians(an)
    sa = math.sin(a)
    ca = math.cos(a)
    if abs(ca) < 0.01:
        ca = 0
    if abs(sa) < 0.01:
        sa = 0
    
    if ax == 'x':
        M[1, 1] = ca
        M[2, 2] = ca
        M[1, 2] = -sa
        M[2, 1] = sa
        
    elif ax == 'y':
        M[0,0] = ca
        M[0,2] = sa
        M[2,0] = -sa
        M[2,2] = ca
        
    elif ax == 'z':
        M[0,0] = ca
        M[0,1] = -sa
        M[1,0] = sa
        M[1,1] = ca
    
    return(M)  
#Plot a 3d vector
def plot_3d(x):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    start = [0,0,0]
    ax.quiver(start[0], start[1], start[2], x[0], x[1], x[2])
    ax.set_xlim([0, 8])
    ax.set_xlabel('x')
    ax.set_ylim([0, 8])
    ax.set_ylabel('y')
    ax.set_zlim([0, 8])
    ax.set_zlabel('z')
    plt.show()
    