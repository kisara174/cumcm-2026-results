"""Approximate posterior from public noise geometry and action receipts only.

Grid-node errors are modeled by their known covariance, with a Gaussian
likelihood approximation. No seed, channel hash, scenario or true count input.
Credible radii are exploratory; the caller retains its full polygon fallback.
"""
import math
import numpy as np


def noise_weights(p):
    x,y=np.asarray(p)/150
    ix,iy=math.floor(x),math.floor(y)
    tx,ty=x-ix,y-iy
    sx,sy=tx*tx*(3-2*tx),ty*ty*(3-2*ty)
    return {(ix,iy):(1-sx)*(1-sy),(ix+1,iy):sx*(1-sy),
            (ix,iy+1):(1-sx)*sy,(ix+1,iy+1):sx*sy}


def estimate(poly,history,problem,geometry,mode=0):
    mid,rad,u,v,width=geometry
    positive=[(p,o) for k,p,o in history if k=='measure' and o['kind'] in ('direction','near')]
    bearings=[(p,o['bearing']) for p,o in positive if o['kind']=='direction']
    if len(bearings)<2:return geometry
    polygon=np.asarray(poly);u=np.asarray(u);v=np.asarray(v)
    xx=polygon@u;yy=polygon@v
    nx,ny=161,17
    xs=np.linspace(xx.min(),xx.max(),nx);ys=np.linspace(yy.min(),yy.max(),ny)
    points=(xs[:,None,None]*u+ys[None,:,None]*v).reshape(-1,2)
    keep=(points*points).sum(axis=1)<=1770**2
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        e=b-a;d=points-a
        keep &= e[0]*d[:,1]-e[1]*d[:,0]>=-1e-6
    failures=[p for k,p,o in history if k=='clear' and o['kind']!='success']
    for p in failures:keep &= np.linalg.norm(points-p,axis=1)>20
    points=points[keep]
    if not len(points):return geometry
    bp=np.array([p for p,b in bearings]);observed=np.array([b for p,b in bearings])
    d=points[:,None,:]-bp
    predicted=np.degrees(np.arctan2(d[:,:,1],d[:,:,0]))
    residual=(predicted-observed+180)%360-180
    w=[noise_weights(p) for p in bp]
    covariance=np.array([[sum(value*b.get(key,0) for key,value in a.items())/3 for b in w] for a in w])
    if mode==1:covariance=np.diag(np.diag(covariance))
    covariance+=np.eye(len(w))*(.01**2/12+1e-6)
    inverse=np.linalg.inv(covariance)
    loglike=-.5*np.einsum('ni,ij,nj->n',residual,inverse,residual)
    # Integrate unknown receive radius and common orientation. This uses all
    # positive and negative receipts, including those preceding discovery.
    posdelta=np.array([p for p,o in positive])[None,:,:]-points[:,None,:]
    posdist=np.linalg.norm(posdelta,axis=2)
    minimum=np.maximum(1000,posdist.max(axis=1))
    negatives=[p for k,p,o in history if k=='measure' and o['kind']=='no_signal']
    nd=np.array(negatives)[None,:,:]-points[:,None,:] if negatives else np.empty((len(points),0,2))
    nr=np.linalg.norm(nd,axis=2)
    if problem==3:
        maximum=np.minimum(1500,nr.min(axis=1)) if negatives else np.full(len(points),1500.)
        likelihood=np.maximum(0,maximum-minimum)/500
    else:
        angles=np.arctan2(posdelta[:,:,1],posdelta[:,:,0])%(2*math.pi)
        negangles=np.arctan2(-nd[:,:,1],-nd[:,:,0])%(2*math.pi)
        likelihood=np.zeros(len(points))
        for f in (0.0625,.1875,.3125,.4375,.5625,.6875,.8125,.9375):
            receive=minimum+f*np.maximum(0,1500-minimum)
            inside=nr<=receive[:,None]
            joined=np.concatenate([angles,np.where(inside,negangles,np.inf)],axis=1)
            joined.sort(axis=1);counts=np.isfinite(joined).sum(axis=1)
            probability=np.zeros(len(points))
            for n in np.unique(counts):
                ix=counts==n;a=joined[ix,:n]
                gaps=np.diff(np.concatenate([a,a[:,:1]+2*math.pi],axis=1),axis=1)
                probability[ix]=np.maximum(0,gaps.max(axis=1)-math.pi)/(2*math.pi)
            # Marginal directional proportion under public N=10..16 prior.
            likelihood+=(.46*(~inside.any(axis=1))+.54*probability)/8
        likelihood*=np.maximum(0,1500-minimum)/500
    loglike+=np.log(np.maximum(likelihood,1e-300))
    if likelihood.max()<=0:return geometry
    weights=np.exp(loglike-loglike.max());weights/=weights.sum()
    center=weights@points
    distance=np.linalg.norm(points-center,axis=1)
    order=np.argsort(distance)
    confidence={2:.95,3:.999,4:.8,5:.5}.get(mode,.999)
    credible=distance[order[min(len(order)-1,np.searchsorted(np.cumsum(weights[order]),confidence))]]
    padding=math.hypot((xs[-1]-xs[0])/(nx-1),(ys[-1]-ys[0])/(ny-1))
    radius=float(credible+padding) if mode>=2 else float(distance.max()+padding)
    return tuple(center),radius,tuple(u),tuple(v),float(min(width,2*radius))
