import numpy as np

def skew(x):
    xx=np.array([[0,-x[2],x[1]],\
                 [x[2],0,-x[0]],\
                 [-x[1],x[0],0]])
    return xx

def outer_product(y,x):
    if y.shape is not (5,1):
        y = y.reshape((5,1))
    if x.shape is not (5,1):
        x = x.reshape((5,1))

    yx=skew(y[:3,0])
    yoi=y[3,0]*np.eye(3)
    yinfi=y[4,0]*np.eye(3)
    A=np.zeros((10,5))
    A[:3,:3]=yx
    A[3:6,0:3]=yoi
    A[3:6,3]=-y[0:3,0]
    A[6:9,0:3]=-yinfi
    A[6:9,4]=y[0:3,0]
    A[9,3]=-y[4,0]
    A[9,4]=y[3,0]
    val=A@x

    return val

def construct_p_from_batch(ps):
    N=ps.shape[0]
    D=np.zeros((5,N))
    for i, p in enumerate(ps):
        D[:3,i]=ps[i,:].T
        D[3,i]=1
        D[4,i]=0.5*np.linalg.norm(ps[i,:])**2
    M=np.block([[np.eye(3),np.zeros((3,2))],\
                [np.zeros((1,4)),-1],\
                [np.zeros((1,3)),-1,0]])
    P2=D@D.T@M
    P2=P2/N
    return P2

def construct_a_single_p(p):
    if p.shape is not (3,1):
        p = p.reshape((3,1))
    P=np.zeros((5,5))
    P[:3,:3]=p@p.T
    pnorm=np.linalg.norm(p)
    P[:3,3]=-0.5*pnorm**2*p.squeeze()
    P[:3,4]=-p.squeeze()
    P[3,:3]=p.T
    P[3,3]=-0.5*pnorm**2
    P[3,4]=-1
    P[4,:3]=0.5*pnorm**2*p.T
    P[4,3]=-0.25*pnorm**4
    P[4,4]=-0.5*pnorm**2

    return P

def recover_circle_parameter(e,verbose=False):
    if e.shape is not (10,):
        e=e.squeeze()
    ei=e[:3]
    eoi=e[3:6]
    einfi=e[6:9]
    eoinf=-e[9]

    alpha=np.linalg.norm(eoi)
    n1=-eoi/alpha
    n=-eoi

    B0=eoinf
    B1=ei[0]
    B2=ei[1]
    B3=ei[2]

    c=np.array([[B0,-B3,B2],\
                [B3,B0,-B1],\
                [-B2,B1,B0]])@n.reshape((3,1))/(np.linalg.norm(n)**2)
    c=c.squeeze()

    radius_square=np.linalg.norm(c)**2-2*n1.dot(einfi)/alpha-2*(c.dot(n1))**2
    radius=np.sqrt(radius_square)
    return (c,radius,n1)

def lsq_fit_3d_circle(points, verbose=False):
    if points.shape[0] < points.shape[1]:
        points=points.T

    M=construct_p_from_batch(points)

    evals,evecs=np.linalg.eig(M)

    indx=np.argsort(evals)
    indx1 = (evals[indx]>0).nonzero()[0]

    sol1=evecs[:,indx[indx1[0]]]
    sol2=evecs[:,indx[indx1[1]]]

    sol_final=outer_product(sol2,sol1)

    center, radius, normal=recover_circle_parameter(sol_final,verbose)

    return {'center':center,'radius':radius}

if __name__=='__main__':
    pass


