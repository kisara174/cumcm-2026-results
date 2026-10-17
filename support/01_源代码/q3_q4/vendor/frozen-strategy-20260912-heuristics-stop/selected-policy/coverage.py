"""Conservative spatial-cell discovery certificate from measurement positions.

For Q4, every cell is inside the convex hull of probes within 1000 m of
EVERY cell point. Thus every possible reception half-plane contains a probe.
Angular uncertainty includes the entire square cell, not just its center.
"""
import math
import numpy as np

class Coverage:
    def __init__(self,problem,step=80):
        self.problem=problem
        self.radius=step/math.sqrt(2)+1e-5
        axis=np.arange(-1800,1800+step,step)
        self.cells=np.array([(x,y) for x in axis for y in axis
                             if math.hypot(x,y)<=1800+self.radius])
        self.points=[]
        self.mask=np.zeros(len(self.cells),dtype=bool)

    def covered(self,points):
        if not points:return np.zeros(len(self.cells),dtype=bool)
        delta=np.array(points)[None,:,:]-self.cells[:,None,:]
        dist=np.linalg.norm(delta,axis=2)
        valid=dist<1000-self.radius
        if self.problem==3:return valid.any(axis=1)
        angles=np.arctan2(delta[:,:,1],delta[:,:,0])%(2*math.pi)
        # An observer inside a cell gives no whole-cell angular guarantee.
        valid &= dist>self.radius
        errors=np.arcsin(np.minimum(1,self.radius/np.maximum(dist,1e-10)))
        angles=np.where(valid,angles,np.inf)
        order=np.argsort(angles,axis=1)
        angles=np.take_along_axis(angles,order,axis=1)
        errors=np.take_along_axis(errors,order,axis=1)
        count=valid.sum(axis=1)
        result=np.zeros(len(self.cells),dtype=bool)
        for n in np.unique(count):
            if n<3:continue
            rows=count==n
            a=angles[rows,:n];e=errors[rows,:n]
            gaps=np.roll(a,-1,axis=1)-a
            gaps[:,-1]+=2*math.pi
            result[rows]=(gaps+e+np.roll(e,-1,axis=1)<math.pi-1e-8).all(axis=1)
        return result

    def add(self,p):
        self.points.append(p)
        self.mask=self.covered(self.points)

    @property
    def complete(self):return bool(self.mask.all())

    def gain(self,p):
        if self.problem==3:return int(np.count_nonzero(self.covered(self.points+[p]) & ~self.mask))
        # Reward partial angular coverage as well as completing cells.
        delta=self.cells[~self.mask]-p
        return int(np.count_nonzero(np.linalg.norm(delta,axis=1)<1000-self.radius))
