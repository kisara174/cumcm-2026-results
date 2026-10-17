"""Planning-only discovery quadrature; not a coverage proof or case input.

Uses the supplied generator's public position/range distribution. Miss mass
is an empirical design criterion, never a statement that a case is clear.
"""
import math
import numpy as np
from scipy.stats import qmc


class EmpiricalDiscovery:
    def __init__(self,problem,tolerance,power=13,stress=0):
        samples=qmc.Halton(5,scramble=True,seed=902611).random(2**power)
        r=1770*np.sqrt(samples[:,0]);a=2*math.pi*samples[:,1]
        self.points=np.column_stack([r*np.cos(a),r*np.sin(a)])
        self.receive2=(1000+500*samples[:,2])**2
        directions=2*math.pi*samples[:,3]
        self.axes=np.column_stack([np.cos(directions),np.sin(directions)])
        self.omni=np.ones(len(r),dtype=bool) if problem==3 else samples[:,4]<.5
        if stress:
            # Public-support stress configurations emphasize short reception
            # radii and boundary/tangent directions missed by average priors.
            radial=np.linspace(250 if stress==1 else 150,1770,7 if stress==1 else 12)
            na=48 if stress==1 else 96;nd=24 if stress==1 else 48
            rr,aa,dd=np.meshgrid(radial,np.arange(na),np.arange(nd),indexing='ij')
            theta=2*math.pi*(aa/na+(rr-radial[0])/(1770*na))
            direction=theta+2*math.pi*dd/nd
            extra=np.column_stack([(rr*np.cos(theta)).ravel(),(rr*np.sin(theta)).ravel()])
            self.points=np.concatenate([self.points,extra])
            self.receive2=np.r_[self.receive2,np.full(len(extra),1000.**2)]
            self.axes=np.concatenate([self.axes,np.column_stack([np.cos(direction).ravel(),np.sin(direction).ravel()])])
            self.omni=np.r_[self.omni,np.full(len(extra),problem==3)]
        self.unseen=np.ones(len(self.points),dtype=bool)
        self.tolerance=tolerance
        self.cache={}

    def visible(self,p):
        key=tuple(round(float(x),4) for x in p)
        if key not in self.cache:
            d=np.asarray(p)-self.points
            self.cache[key]=((d*d).sum(axis=1)<=self.receive2)&(self.omni|((d*self.axes).sum(axis=1)>=0))
        return self.cache[key]

    def add(self,p):
        self.unseen &= ~self.visible(p)

    def adjust(self,remaining,targets,position,blend=False):
        if not remaining:return remaining
        pending=list(remaining)
        budget=int(self.tolerance*len(self.unseen))
        def miss(points):
            unseen=self.unseen.copy()
            for p in points:unseen &= ~self.visible(p)
            return int(unseen.sum())
        # Keep at least the original planned quality when its residual mass
        # exceeds the requested empirical threshold.
        bound=max(budget,miss(pending))
        for p in sorted(pending,key=lambda p:-math.dist(position,p)):
            others=[q for q in pending if q!=p]
            if miss(others)<=bound:pending=others
        # A source-service point substitutes for a discovery point only if
        # the whole remaining empirical design stays within its miss budget.
        for target in targets:
            if not pending:break
            p=min(pending,key=lambda p:math.dist(p,target))
            if math.dist(p,target)<2 or math.dist(p,target)>850:continue
            others=[q for q in pending if q!=p]
            for fraction in ((1.,.75,.5,.25) if blend else (1.,)):
                q=tuple(round(p[k]+fraction*(target[k]-p[k]),6) for k in (0,1))
                if miss(others+[q])<=bound:
                    pending=others+[q]
                    break
        # Bounded cache: old proposed positions are no longer needed.
        if len(self.cache)>1000:self.cache={}
        return pending
