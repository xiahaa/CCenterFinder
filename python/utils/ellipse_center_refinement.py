import logging

import numpy as np

def get_ellipse_polynomial_coeff(elps):
    a = elps[1][0]*0.5
    b = elps[1][1]*0.5
    theta = elps[2]*np.pi / 180.0
    M_PI = np.pi
    cx = elps[0][0]
    cy = elps[0][1]

    if a<b:
        theta += M_PI/2
        c=a
        a=b
        b=c

    A = (a**2) * (np.sin(theta)**2) + (b**2) * (np.cos(theta)**2)
    B = 2*((b**2)-(a**2))*np.sin(theta)*np.cos(theta)
    C = (a**2)*(np.cos(theta)**2)+(b**2)*(np.sin(theta)**2)
    D = -2*A*cx-B*cy
    E = -B*cx-2*C*cy
    F = A*(cx**2)+B*cx*cy+C*(cy**2)-(a**2)*(b**2)
    k = 1/F

    poly = np.array([A,B,C,D,E,F]) * k
    return poly

def get_ellipse_line_intersections(em, x0, y0, theta):
    p1 = np.zeros((2,))
    p2 = np.zeros((2,))
    if abs(theta - np.pi / 2) < 1e-6:
        p1[0] = x0
        p1[1] = (-(x0 * em[1]) - em[4] - np.sqrt((x0 * em[1] + em[4])**2 - 4 * em[2] * ((x0**2) * em[0] + x0 * em[3] + em[5]))) / (2 * em[2])
        p2[0] = x0
        p2[1] = (-(x0 * em[1]) - em[4] + np.sqrt((x0 * em[1] + em[4])**2 - 4 * em[2] * ((x0**2) * em[0] + x0 * em[3] + em[5]))) / (2 * em[2])
    else:
        lm = np.array([-np.tan(theta), 1.0, np.tan(theta) * x0 - y0])
        tmp0 = lm[0]
        tmp1 = lm[1]
        tmp2 = em[2]
        tmp3 = em[1]
        tmp4 = lm[2]
        tmp5 = em[4]
        tmp6 = em[3]
        tmp7 = tmp1**2
        tmp8 = tmp0**2
        tmp9 = tmp2 * tmp8
        tmp10 = -(tmp3 * tmp0)
        tmp11 = em[0]
        tmp12 = tmp11 * tmp1
        tmp13 = tmp10 + tmp12
        tmp14 = tmp1 * tmp13
        tmp15 = tmp9 + tmp14
        tmp16 = 1 / tmp15
        tmp17 = tmp3 * tmp4
        tmp18 = tmp5 * tmp0 * tmp1
        tmp19 = -(tmp6 * tmp7)
        tmp20 = -2 * tmp2 * tmp0
        tmp21 = tmp3 * tmp1
        tmp22 = tmp20 + tmp21
        tmp23 = tmp22 * tmp4
        tmp24 = tmp18 + tmp19 + tmp23
        tmp25 = tmp24**2
        tmp26 = em[5]
        tmp27 = tmp26 * tmp7
        tmp28 = -(tmp5 * tmp1)
        tmp29 = tmp2 * tmp4
        tmp30 = tmp28 + tmp29
        tmp31 = tmp4 * tmp30
        tmp32 = tmp27 + tmp31
        tmp33 = -4 * tmp15 * tmp32
        tmp34 = tmp25 + tmp33
        if tmp34 > 1e-10:
            tmp35 = np.sqrt(tmp34)
        else:
            tmp35 = 0
        tmp36 = 1 / tmp1
        tmp37 = tmp6 * tmp0
        tmp38 = -2 * tmp11 * tmp4
        tmp39 = tmp37 + tmp38
        tmp40 = tmp7 * tmp39
        tmp41 = -(tmp5 * tmp0)
        tmp42 = tmp41 + tmp17
        tmp43 = tmp0 * tmp1 * tmp42

        p1[0] = -(tmp16 * (tmp6 * tmp7 + 2 * tmp2 * tmp0 * tmp4 - tmp1 * (tmp5 * tmp0 + tmp17) + tmp35)) / 2.
        p1[1] = (tmp36 * tmp16 * (tmp40 + tmp43 + tmp0 * tmp35)) / 2.
        p2[0] = (tmp16 * (tmp18 + tmp19 - 2 * tmp2 * tmp0 * tmp4 + tmp3 * tmp1 * tmp4 + tmp35)) / 2.
        p2[1] = (tmp36 * tmp16 * (tmp40 + tmp43 - tmp0 * tmp35)) / 2.

    return (p1, p2)

def get_distance_given_center(elps, c, r, N, K):
    sum = 0
    sumsq = 0
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
    M_PI = np.pi
    cnt = 0
    for i in range(0,N):
        theta = i * M_PI / N
        p1, p2 = get_ellipse_line_intersections(elps, c[0], c[1], theta)
        if p1 is not None and p2 is not None:
            vr1 = np.array([(p1[0]-cx)/f,(p1[1]-cy)/f,1])
            vr2 = np.array([(p2[0] - cx) / f, (p2[1] - cy) / f, 1])
            vrc = np.array([(c[0] - cx) / f, (c[1] - cy) / f, 1])

            try:
                dotprod = np.dot(vr1/np.linalg.norm(vr1), vrc/np.linalg.norm(vrc))
                dotprod = np.clip(dotprod,-1,1)
                th1 = np.math.acos(dotprod)
                dotprod = np.dot(vr2/np.linalg.norm(vr2), vrc/np.linalg.norm(vrc))
                dotprod = np.clip(dotprod, -1, 1)
                th2 = np.math.acos(dotprod)
            except:
                raise RuntimeWarning("Something wrong, just continue")
                continue

            res = np.clip(3-2*np.cos(2*th1)-2*np.cos(2*th2)+np.cos(2*(th1+th2)),0,1e6)
            curd = (np.sqrt(2)*r*np.sin(th1+th2))/np.sqrt(res)

            sum+=curd
            sumsq+=curd**2
            cnt+=1
    if cnt != 0:
        mu = sum /cnt
        if (sumsq/cnt - mu**2) < 1e-10:
            std = 0
        else:
            std = np.sqrt(sumsq/cnt-mu**2)
    else:
        mu = sum / N
        std = np.sqrt(sumsq / N - mu ** 2)

    if np.isnan(std):
        mu = 0
        std = 0

    return (mu, std)

def eval_distance_f0(outer, x, K, marker_diamater, N=4):
    mu, std = get_distance_given_center(outer, x, marker_diamater / 2, N, K)
    ret = std
    # could add some regularization term here

    return ret

def eval_distance_f(outer, x, x0, K, marker_diamater, N=4):
    mu, std = get_distance_given_center(outer, x, marker_diamater / 2, N, K)
    ret = std
    ret +=1e-4*((x[0]-x0[0])**2+(x[1]-x0[1])**2)

    return ret

def get_distance_with_gradient_descent(outer, x0, step, plambda, tolx, tolfunc, K, marker_diameter, verbose=Flase):
    x = x0.copy()
    it = 1
    g = np.zeros((2,))
    xs = []
    xs.append(x0)
    while True:
        fplus = eval_distance_f(outer, x+np.array([step,0]), x0, K, marker_diameter)
        fmius = eval_distance_f(outer, x-np.array([step,0]), x0, K, marker_diameter)
        g[0] = (fplus-fmius)/2/step
        fplus = eval_distance_f(outer, x + np.array([0, step]), x0, K, marker_diameter)
        fmius = eval_distance_f(outer, x - np.array([0, step]), x0, K, marker_diameter)
        g[1] = (fplus - fmius) / 2 / step
        f = eval_distance_f(outer, x, x0, K, marker_diameter)

        is_reduce = False
        step_down = False

        while(plambda*np.linalg.norm(g)>tolx and step_down is False):
            newx = x - plambda * g
            newf = eval_distance_f(outer, newx, x0, K, marker_diameter)
            if newf < f:
                x = newx
                f = newf
                if is_reduce is False:
                    plambda*=2
                step_down = True
            else:
                plambda/=2
                is_reduce = True

        if step_down is False:
            if verbose:
                logging.INFO("plambda * norm(g): {}".format(plambda*np.linalg.norm(g)))
                logging.INFO("cannot reduce!")
            break

        if abs(f-newf) < tolfunc:
            if verbose:
                logging.INFO("exit because of tolfunc")
            break
        xs.append(x)
        it+=1

        if it>1e5:
            logging.INFO("exit because of max iteration!")
            break
    if verbose:
        logging.INFO("after {} iterations!".format(it))
    return (x, xs)

if __name__ == "__main__":
    # todo add some unit test here
    pass
