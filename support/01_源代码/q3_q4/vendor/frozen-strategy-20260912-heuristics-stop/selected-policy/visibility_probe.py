"""Predict reception of proposed probes by integrating receipt-consistent priors.

This numerical planning likelihood is not a proof of reception or exclusion.
Actual observations alone update the retained geometric source polygon.
"""
import math
import numpy as np


def orientation_mass(angles):
    angles=np.sort(angles,axis=1)
    counts=np.isfinite(angles).sum(axis=1)
    mass=np.zeros(len(angles))
    for n in np.unique(counts):
        ix=counts==n;a=angles[ix,:n]
        if not n:mass[ix]=1;continue
        gaps=np.diff(np.concatenate([a,a[:,:1]+2*math.pi],axis=1),axis=1)
        mass[ix]=np.maximum(0,gaps.max(axis=1)-math.pi)/(2*math.pi)
    return mass


def probabilities(poly,history,geometry,candidates):
    mid,rad,u,v,width=geometry
    pos=[(p,o) for k,p,o in history if k=='measure' and o['kind'] in ('near','direction')]
    neg=[p for k,p,o in history if k=='measure' and o['kind']=='no_signal']
    if not pos:return [1.]*len(candidates)
    polygon=np.asarray(poly);u=np.asarray(u);v=np.asarray(v)
    x=polygon@u;y=polygon@v
    pts=(np.linspace(x.min(),x.max(),97)[:,None,None]*u+
         np.linspace(y.min(),y.max(),9)[None,:,None]*v).reshape(-1,2)
    keep=(pts*pts).sum(axis=1)<=1770**2
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        e=b-a;d=pts-a;keep &= e[0]*d[:,1]-e[1]*d[:,0]>=-1e-6
    for k,p,o in history:
        if k=='clear' and o['kind']!='success':keep &= np.linalg.norm(pts-p,axis=1)>20
    pts=pts[keep]
    if not len(pts):return [1.]*len(candidates)
    pd=np.array([p for p,o in pos])[None,:,:]-pts[:,None,:]
    pa=np.arctan2(pd[:,:,1],pd[:,:,0])%(2*math.pi)
    minimum=np.maximum(1000,np.linalg.norm(pd,axis=2).max(axis=1))
    logweight=np.zeros(len(pts))
    for j,(p,o) in enumerate(pos):
        if o['kind']=='direction':
            error=(np.degrees(pa[:,j])+180-o['bearing']+180)%360-180
            logweight-=.5*(error/.4)**2
    weight=np.exp(logweight-logweight.max())*np.maximum(0,1500-minimum)/500
    nd=np.array(neg)[None,:,:]-pts[:,None,:] if neg else np.empty((len(pts),0,2))
    nr=np.linalg.norm(nd,axis=2)
    na=np.arctan2(-nd[:,:,1],-nd[:,:,0])%(2*math.pi)
    qd=np.asarray(candidates)[None,:,:]-pts[:,None,:]
    qr=np.linalg.norm(qd,axis=2);qa=np.arctan2(qd[:,:,1],qd[:,:,0])%(2*math.pi)
    base=0.;numerator=np.zeros(len(candidates))
    for fraction in (.125,.375,.625,.875):
        receive=minimum+fraction*np.maximum(0,1500-minimum)
        inside=nr<=receive[:,None]
        angles=np.concatenate([pa,np.where(inside,na,np.inf)],axis=1)
        omni=(~inside.any(axis=1))*.46
        base+=weight@(.54*orientation_mass(angles)+omni)
        for j in range(len(candidates)):
            new=np.concatenate([angles,qa[:,j,None]],axis=1)
            likelihood=(.54*orientation_mass(new)+omni)*(qr[:,j]<=receive)
            numerator[j]+=weight@likelihood
    return list(numerator/base) if base>1e-20 else [1.]*len(candidates)
