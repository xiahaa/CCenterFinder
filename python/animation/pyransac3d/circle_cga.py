
import random

import numpy as np

import copy

from .aux import *

import time

def skew(x):
    xx=np.array([[0,-x[2],x[1]],\
                 [x[2],0,-x[0]],\
                 [-x[1],x[0],0]])
    return xx

def outer_product(y,x):
    if y.shape != (5,1):
        y = y.reshape((5,1))
    if x.shape != (5,1):
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
    if p.shape != (3,1):
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
    if e.shape != (10,):
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

    return center, radius, normal


class CircleCGA:
    def __init__(self):
        self.inliers = []
        self.center = []
        self.normal = []
        self.radius = 0

    def fit(self, pts, pcd_load, thresh=0.2, maxIteration=1000):
        def rotate_view(vis):
            ctr.rotate(0.1, 0.0)
            return False

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        n_points = pts.shape[0]
        best_eq = []
        best_inliers = []

        for it in range(maxIteration):
            # Samples 3 random points
            id_samples = random.sample(range(0, n_points), 5)
            pt_samples = pts[id_samples]

            center, radius, normal = lsq_fit_3d_circle(pt_samples, verbose=False)
            normal = normal / np.linalg.norm(normal)

            vecC = normal
            k = -np.sum(np.multiply(vecC, center))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            # Distance from a point to the circle's plane
            dist_pt_plane = (plane_eq[0]*pts[:,0]+plane_eq[1]*pts[:, 1]+plane_eq[2]*pts[:, 2]+plane_eq[3])/np.sqrt(plane_eq[0]**2+plane_eq[1]**2+plane_eq[2]**2)
            vecC_stakado =  np.stack([vecC]*n_points,0)
            # Distance from a point to the circle hull if it is infinite along its axis (perpendicular distance to the plane)
            dist_pt_inf_circle = np.cross(vecC_stakado, (center- pts))
            dist_pt_inf_circle = np.linalg.norm(dist_pt_inf_circle, axis=1) - radius

            # https://math.stackexchange.com/questions/31049/distance-from-a-point-to-circles-closest-point
            # The distance from a point to a circle will be the hipotenusa
            dist_pt = np.sqrt(np.square(dist_pt_inf_circle)+np.square(dist_pt_plane))

            # https://math.stackexchange.com/questions/31049/distance-from-a-point-to-circles-closest-point
            # The distance from a point to a circle will be the hipotenusa
            dist_pt = np.sqrt(np.square(dist_pt_inf_circle))

            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(dist_pt <= thresh)[0]

            if len(pt_id_inliers) > len(best_inliers):
                best_inliers = pt_id_inliers
                self.inliers = best_inliers
                self.center = center
                self.normal = normal
                self.radius = radius

            R2 = get_rotationMatrix_from_vectors([0, 0, 1], self.normal )
            mesh_cylinder2 = o3d.geometry.TriangleMesh.create_torus(torus_radius=self.radius, tube_radius=0.1)
            mesh_cylinder2.compute_vertex_normals()
            mesh_cylinder2.paint_uniform_color([0, 1, 0])
            mesh_cylinder2 = mesh_cylinder2.rotate(R2, center=[0, 0, 0])
            mesh_cylinder2 = mesh_cylinder2.translate((self.center[0], self.center[1], self.center[2]))

            R1 = get_rotationMatrix_from_vectors([0, 0, 1], normal )
            mesh_cylinder1 = o3d.geometry.TriangleMesh.create_torus(torus_radius=radius, tube_radius=0.1)
            mesh_cylinder1.compute_vertex_normals()
            mesh_cylinder1.paint_uniform_color([1, 0, 0])
            mesh_cylinder1 = mesh_cylinder1.rotate(R1, center=[0, 0, 0])
            mesh_cylinder1 = mesh_cylinder1.translate((center[0], center[1], center[2]))

            lin = pcd_load.select_by_index(pt_id_inliers).paint_uniform_color([1, 0, 0])
            lin2 = pcd_load.select_by_index(self.inliers).paint_uniform_color([0, 1, 0])
            vis.clear_geometries()
            #time.sleep(0.01)
            vis.add_geometry(copy.deepcopy(pcd_load))
            # vis.add_geometry(mesh_cylinder)
            vis.add_geometry(mesh_cylinder2)
            if(radius <= 1.2*self.radius):
                vis.add_geometry(mesh_cylinder1)
            vis.add_geometry(lin)
            vis.add_geometry(lin2)

            ctr = vis.get_view_control()
            ctr.rotate(-it*4, -it*4,)
            ctr.set_zoom(0.7)
            ctr.set_lookat([0, 0, 0])

            vis.poll_events()
            vis.update_renderer()

            # save vis to a path
            import os
            output_folder = 'output'
            os.makedirs(output_folder, exist_ok=True)
            filename = os.path.join(output_folder, 'circle_{:04d}.png'.format(it))
            vis.capture_screen_image(filename, do_render=True)

            time.sleep(0.04)

        return self.center, self.normal, self.radius, self.inliers
