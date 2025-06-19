#import numpy as np
#import matplotlib.pyplot as plt
#import tqdm
from torch.func import jacfwd, jacrev, vmap
#import time
import torch
from torch import tensor, sum, log, max, cos, sin, pi, clone, sqrt, mean, std, sort
from scipy.constants import g

def L(t,tdot,m1,m2,l1,l2):
    t0 = sum(t*tensor([1,0]))
    t1 = sum(t*tensor([0,1]))
    td0 = sum(tdot*tensor([1,0]))
    td1 = sum(tdot*tensor([0,1]))
    #p1 = .5*(m1+m2)*(l1*tdot[0])**2 + .5*m2*(l2*tdot[1])**2
    #p2 = m2*l1*l2*tdot[0]*tdot[1]*cos(t[1]-t[0]) + (m1+m2)*g*l1*cos(t[0]) + m2*g*l2*cos(t[1])
    p1 = .5*(m1+m2)*(l1*td0)**2 + .5*m2*(l2*td1)**2
    p2 = m2*l1*l2*td0*td1*cos(t1-t0) + (m1+m2)*g*l1*cos(t0) + m2*g*l2*cos(t1)
    
    return p1 + p2

Lv = vmap(L,in_dims=(1,1,None,None,None,None))

def QDD(Q,QD,m1,m2,l1,l2):
    g_q2 = jacrev(L,argnums=0)(Q,QD,m1,m2,l1,l2)
    g_qdot2 = jacrev(L,argnums=1)(Q,QD,m1,m2,l1,l2)
    g_q_qdot2 = jacrev(jacrev(L,argnums=1),argnums=0)(Q,QD,m1,m2,l1,l2)
    g_qdot_q2 = jacrev(jacrev(L,argnums=0),argnums=1)(Q,QD,m1,m2,l1,l2)
    g_qdot_qdot2 = jacfwd(jacrev(L,argnums=1),argnums=1)(Q,QD,m1,m2,l1,l2)
    D = torch.linalg.inv(g_qdot_qdot2)
    return (D@(g_q2 - g_q_qdot2@QD.T).mT).reshape(Q.shape)

def QDD_true(t,tdot,m1,m2,l1,l2): #for testing if output is as expected for alpha=0
    tdd1 = -g*(2*m1+m2)*sin(t[0]) - m2*g*sin(t[0]-2*t[1]) - 2*sin(t[0]-t[1])*m2*(l2*tdot[1]**2 + l1*(tdot[0]**2)*cos(t[0]-t[1]))
    tdd1 = tdd1 / (l1* (2*m1 + m2 - m2*cos(2*t[0] - 2*t[1]) ) )
    tdd2 = 2*sin(t[0]-t[1])*( (m1+m2)*l1*tdot[0]**2 + g*(m1+m2)*cos(t[0]) + l2*m2*cos(t[0]-t[1])*tdot[1]**2 )
    tdd2 = tdd2 / ( l2 * (2*m1 + m2 - m2*cos(2*t[0] - 2*t[1])) )
    return tensor([tdd1,tdd2])

QDDv = vmap(QDD,in_dims=(1,1,None,None,None,None)) #this actually seems to work?

#def QDDv(Q,QD,m1,m2,l1,l2):
#    QDDv_inner = vmap(QDD,in_dims=(1,1,None,None,None,None))
#    return QDDv_inner(Q,QD,m1,m2,l1,l2)

def QDD_true_v(t,tdot,m1,m2,l1,l2): #pseudo-vectorized...
    N = t.shape[-1]
    outs = torch.zeros(t.shape)
    for n in range(N):
        outs[:,n] = QDD_true(t[:,n],tdot[:,n],m1,m2,l1,l2)
    return outs

def rand_sampling_q_no_qdot(t0_min,t0_max,t1_min,t1_max,n_points):
    t0_sampling = (t0_max - t0_min)*(torch.rand(n_points)) + t0_min
    t1_sampling = (t1_max - t1_min)*(torch.rand(n_points)) + t1_min
    tdot_sampling = (tdot_max - tdot_min)*(torch.rand([2,n_points])) + tdot_min
    
    Q0,ind = sort(t0_sampling)
    
    Q1 = t1_sampling[ind]
    QD0 = tdot_sampling[0,:]
    QD1 = tdot_sampling[1,:]

    Qv = torch.zeros(2,n_points)
    Qv[0,:] = Q0
    Qv[1,:] = Q1
    
    QDv = torch.zeros(2,n_points)
    #QDv[0,:] = QD0
    #QDv[1,:] = QD1
    return Qv, QDv

def grid_sampling_q_no_qdot(t0_min,t0_max,t1_min,t1_max,n_points):
    n_side = int(torch.sqrt(tensor(n_points)))
    t0_grid = torch.linspace(t0_min,t0_max,n_side)
    t1_grid = torch.linspace(t1_min,t1_max,n_side)
    T0, T1 = torch.meshgrid(t0_grid,t1_grid, indexing='ij')
    Qv = torch.zeros(2,n_side**2)
    Qv[0,:] = T0.reshape(-1)
    Qv[1,:] = T1.reshape(-1)
    QDv = torch.zeros(2,n_side**2)
    return Qv, QDv

def Euler_step(t,tdot,h,m1,m2,l1,l2):
    acc = QDD(t,tdot,m1,m2,l1,l2)
    t_new = t + h*tdot
    tdot_new = tdot + h*acc
    return t_new, tdot_new

def Euler_step_true(t,tdot,h,m1,m2,l1,l2):
    acc = QDD_true(t,tdot,m1,m2,l1,l2)
    
    t_new = t + h*tdot
    tdot_new = tdot + h*acc
    return t_new, tdot_new