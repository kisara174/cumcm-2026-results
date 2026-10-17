"""Joint sampled set cover over mandatory service points and optional probes.

Discovery can be split across several already-required service destinations.
The objective approximates measurement plus route insertion cost. No actual
source layout/count is available; finite coverage is not a continuous proof.
"""
import math,warnings
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import csc_matrix
from routing import route


def choose(design,remaining,original,centers,position,scan_cost=300):
    if not design.unseen.any():return []
    candidates=list(dict.fromkeys(tuple(round(float(x),6) for x in p)
                    for p in [position]+list(centers)+list(remaining)+list(original)))
    masks=np.column_stack([design.visible(p)[design.unseen] for p in candidates])
    useful=masks.any(axis=0);masks=masks[:,useful]
    candidates=[p for p,good in zip(candidates,useful) if good]
    if not masks.any(axis=1).all():return remaining
    # Equal coverage rows define the same binary constraint.
    _,indices=np.unique(np.packbits(masks,axis=1),axis=0,return_index=True)
    matrix=masks[indices].astype(float)
    service=route([('source',i,p) for i,p in enumerate(centers)],position,4)
    path=[position]+[n[2] for n in service]
    costs=[]
    for j,p in enumerate(candidates):
        insertion=math.dist(path[-1],p)
        for a,b in zip(path,path[1:]):insertion=min(insertion,math.dist(a,p)+math.dist(p,b)-math.dist(a,b))
        costs.append(scan_cost+max(0,insertion)+j*1e-7)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected')
        result=milp(np.array(costs),integrality=np.ones(len(costs)),bounds=Bounds(0,1),
            constraints=LinearConstraint(csc_matrix(matrix),1,np.inf),
            options=dict(time_limit=.5,mip_rel_gap=.01,threads=1))
    if result.x is None:return remaining
    chosen=result.x>.5
    if not masks[:,chosen].any(axis=1).all():return remaining
    return [p for p,keep in zip(candidates,chosen) if keep]
