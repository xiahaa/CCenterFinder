import numpy as np
import ellipse_center_refinement as ellref

def normalize_mat_det_1(M):
    nr, nc = M.shape[:2]
    if nr==nc:
        d=np.linalg.det(M)
        s = np.sign(d) / abs(d)**(1/nr)
        N = s*M
    else:
        raise ValueError("Matrix is not Square!")
    return (N,s)

def constuctH(Q11,Q22,Q33,xc,yc):
    Hp=np.array([[1,0,0],[0,1,0],[Q11*xc,Q22*yc,Q33]])
    a = Q22*xc*yc / (Q11*xc**2+Q33)
    res = (Q22*Q33/(Q11+1e-10)*((Q11*xc**2+Q22*yc**2+Q33)/(Q11*xc**2+Q33+1e-10)**2)-a**2)
    if res > 1e-10:
        b = (Q22*Q33/(Q11+1e-10)*((Q11*xc**2+Q22*yc**2+Q33)/(Q11*xc**2+Q33+1e-10)**2)-a**2)**(1/2)
    else:
        b = 1e-10
    Ha = np.array([[1/b,-a/b,0],[0,1,0],[0,0,1]])
    x = (-xc/b+yc*a/b)/(Q11*xc**2+Q22*yc**2+Q33)
    y = -yc / (Q11*xc**2+Q22*yc**2+Q33)
    He=np.array([[1,0,x],[0,1,y],[0,0,1]])
    H=He@Ha@Hp
    return H

def constructG(Q11,Q22,Q33,u,v):
    r=np.sqrt(-Q22/Q11*(Q11*u**2+Q22*v**2+Q33))
    s=np.sqrt(-Q22*(1-Q11*u**2))
    Ginv=np.array([[-1,Q22*u*v,-u],[0,-Q11*u**2+1,-v],[-Q11*u,Q22*v,1]])@np.array([[r,0,0],[0,-1,0],[0,0,s]])
    return Ginv

def findTtoCanonicalQ(ell):
    poly = ellref.get_ellipse_polynomial_coeff(ell)
    C=np.array([[poly[0],poly[1]/2,poly[3]/2],\
                [poly[1]/2,poly[2],poly[4]/2],\
                [poly[3]/2,poly[4]/2,poly[5]]])
    ex=ell[0][0]
    ey=ell[0][1]
    etheta=ell[2]*np.pi/180.0

    A1 = np.array([[np.cos(etheta),np.sin(etheta),0],\
                   [-np.sin(etheta),np.cos(etheta),0],\
                   [0,0,1]])
    A2=np.array([[1,0,-ex],[0,1,-ey],[0,0,1]])
    T=A1@A2
    Q=np.linalg.inv(T).T@C@np.linalg.inv(T)

    return (T,Q,C)

def findHomography(ell,center):
    T,Q,_=findTtoCanonicalQ(ell)
    Q11=Q[0,0]
    Q22=Q[1,1]
    Q33=Q[2,2]

    o=center.copy()
    o=np.array([*o,1])
    o=o.reshape((3,1))
    no=T@o
    no=np.reshape(no,(3,))
    H=constuctH(Q11,Q22,Q33,no[0],no[1])

    return (H,Q,T)

if __name__=="__main__":
    pass
