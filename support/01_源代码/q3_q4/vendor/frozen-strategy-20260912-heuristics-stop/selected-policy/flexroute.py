"""Move Q3 discovery ring probes while preserving continuous annulus cover.

The origin covers radii <= 1000. Every direction is assigned a ring probe
within 999.9 m at BOTH annulus endpoints r=1000 and r=1800; convexity of
squared distance in r then covers the entire segment. Adjacent angular
intervals must overlap. Normal floating point, explicitly slackened.
"""
import math
import numpy as np
from scipy.optimize import minimize
from routing import route

def margins(points):
    radii=np.linalg.norm(points,axis=1)
    angles=np.unwrap(np.arctan2(points[:,1],points[:,0]))
    alpha=np.minimum(np.arccos(np.clip((1800**2+radii**2-999.9**2)/(3600*radii),-1,1)),
                     np.arccos(np.clip((1000**2+radii**2-999.9**2)/(2000*radii),-1,1)))
    gaps=np.diff(np.r_[angles,angles[0]+2*math.pi])
    return np.r_[alpha+np.roll(alpha,-1)-gaps,gaps]

def adjust(ring,remaining,centers,position,iterations=20):
    free=[i for i,p in enumerate(ring) if p in remaining]
    if not free:return ring,remaining
    points=np.array(ring);radii=np.linalg.norm(points,axis=1)
    theta=np.unwrap(np.arctan2(points[:,1],points[:,0]))
    seed=np.r_[radii[free],np.zeros(len(free))]
    nodes=[('ring',i,tuple(p)) for i,p in enumerate(points) if i in free]
    nodes += [('source',i,p) for i,p in enumerate(centers)]
    tour=route(nodes,position,4)
    def unpack(x):
        copy=points.copy();n=len(free)
        for j,i in enumerate(free):
            a=theta[i]+x[n+j]/1300
            copy[i]=(x[j]*math.cos(a),x[j]*math.sin(a))
        return copy
    def cost(x):
        changed=unpack(x);last=position;total=0.
        for kind,i,p in tour:
            q=changed[i] if kind=='ring' else p
            total+=math.dist(last,q);last=q
        return total
    result=minimize(cost,seed,method='SLSQP',bounds=[(1125,1620)]*len(free)+[(-500,500)]*len(free),
        constraints=[dict(type='ineq',fun=lambda x:margins(unpack(x))*1000)],
        options=dict(maxiter=iterations,ftol=.05))
    changed=[tuple(round(float(v),6) for v in p) for p in unpack(result.x)]
    if min(margins(np.array(changed)))<-1e-7 or cost(result.x)>=cost(seed):return ring,remaining
    replacement={ring[i]:changed[i] for i in free}
    return changed,[replacement.get(p,p) for p in remaining]
