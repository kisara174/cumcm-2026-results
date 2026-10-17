"""Planning-only quadrature prior; never a completion/exclusion certificate."""
import math
import numpy as np

class SearchPrior:
    def __init__(self,problem):
        points=[];radii=[];directions=[];weights=[]
        for i in range(1,13):
            r=1800*math.sqrt((i-.5)/12)
            for j in range(36):
                p=(r*math.cos(j*math.pi/18),r*math.sin(j*math.pi/18))
                for receive in (1000,1250,1500):
                    configs=[(-1,1.)] if problem==3 else [(-1,.5)]+[(k,.5/12) for k in range(12)]
                    for k,w in configs:
                        points.append(p);radii.append(receive);directions.append(k);weights.append(w)
        self.points=np.array(points);self.radii=np.array(radii);self.directions=np.array(directions)
        self.weights=np.array(weights);self.weights/=self.weights.sum()
        self.active=np.ones(len(points),dtype=bool)
        self.centers,self.center_ids=np.unique(self.points,axis=0,return_inverse=True)
    def visible(self,p):
        delta=p-self.points
        distance=np.linalg.norm(delta,axis=1)
        projection=delta[:,0]*np.cos(self.directions*math.pi/6)+delta[:,1]*np.sin(self.directions*math.pi/6)
        return (distance<=self.radii)&((self.directions<0)|(projection>=0))
    def add(self,p):self.active &= ~self.visible(np.array(p))
    def gain(self,p,unknown_channels):
        mass=self.weights[self.active].sum()
        hit=self.weights[self.active & self.visible(np.array(p))].sum()
        return unknown_channels*.65*hit/(.35+.65*mass)
    def forecast(self,unknown_channels,scale):
        weights=np.bincount(self.center_ids,weights=self.weights*self.active,minlength=len(self.centers))
        mass=weights.sum()
        if mass<1e-10:return []
        expected=unknown_channels*.65*mass/(.35+.65*mass)
        count=min(8,max(0,round(expected*scale)))
        if not count:return []
        centers=[self.centers[int(np.argmax(weights))]]
        for _ in range(count-1):
            d=((self.centers[:,None,:]-np.array(centers)[None,:,:])**2).sum(axis=2).min(axis=1)
            centers.append(self.centers[int(np.argmax(weights*d))])
        centers=np.array(centers)
        for _ in range(12):
            labels=((self.centers[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
            for i in range(count):
                mask=labels==i
                if weights[mask].sum():centers[i]=(self.centers[mask]*weights[mask,None]).sum(axis=0)/weights[mask].sum()
        return [tuple(p) for p in centers]
