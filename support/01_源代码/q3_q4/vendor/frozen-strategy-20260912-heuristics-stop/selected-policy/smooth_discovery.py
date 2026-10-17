"""Shorten a probe/service tour while retaining its sampled discovery cover.

Assign each latent source configuration to a probe. Its admissible probe
region is the intersection of a reception disk and a directional half-plane.
Move each probe toward the adjacent route segment up to the first boundary.
Reassign and repeat. This only protects finite planning samples, not a proof.
"""
import math
import numpy as np
from routing import route


def adjust(design,remaining,centers,position,passes=4,guard=0):
    if not remaining:return remaining
    source=design.points[design.unseen];r2=design.receive2[design.unseen]
    axes=design.axes[design.unseen];omni=design.omni[design.unseen]
    if not len(source):return []
    probes=np.asarray(remaining,dtype=float).copy()
    initial=probes.copy()
    for iteration in range(passes):
        delta=probes[None,:,:]-source[:,None,:]
        distance=np.linalg.norm(delta,axis=2)
        projection=(delta*axes[:,None,:]).sum(axis=2)
        margin=np.minimum(np.sqrt(r2)[:,None]-distance,np.where(omni[:,None],np.inf,projection))
        owner=margin.argmax(axis=1)
        if (margin.max(axis=1)<-1e-7).any():return remaining
        nodes=[('anchor',i,tuple(p)) for i,p in enumerate(probes)]
        nodes += [('source',i,p) for i,p in enumerate(centers)]
        tour=route(nodes,position,4 if iteration==0 else 1)
        moved=False
        for j,node in enumerate(tour):
            if node[0]!='anchor':continue
            i=node[1];p=probes[i]
            def live(n):return probes[n[1]] if n[0]=='anchor' else np.asarray(n[2])
            left=np.asarray(position) if j==0 else live(tour[j-1])
            right=live(tour[j+1]) if j+1<len(tour) else left
            line=right-left;length2=float(line@line)
            target=left+np.clip(float((p-left)@line)/length2,0,1)*line if length2>1e-9 else left
            d=target-p;a=float(d@d)
            if a<1:continue
            ix=owner==i
            if not ix.any():
                probes[i]=target;moved=True;continue
            relative=p-source[ix];b=2*(relative@d);dist2=(relative*relative).sum(axis=1)
            allowed=r2[ix]
            if guard:allowed=np.maximum(np.sqrt(dist2),np.sqrt(allowed)-guard)**2
            c=dist2-allowed
            roots=(-b+np.sqrt(np.maximum(0,b*b-4*a*c)))/(2*a)
            fraction=min(1.,max(0.,float(roots.min())))
            oriented=~omni[ix]
            if oriented.any():
                normals=axes[ix][oriented];s=(relative[oriented]*normals).sum(axis=1);rate=normals@d
                if guard:s=s-np.minimum(s,guard)
                closing=rate<0
                if closing.any():fraction=min(fraction,max(0.,float((s[closing]/-rate[closing]).min())))
            fraction=max(0.,fraction-1e-7)
            if fraction*math.sqrt(a)>1:
                probes[i]=p+fraction*d;moved=True
        if not moved:break
    changed=[tuple(round(float(x),6) for x in p) for p in probes]
    # Independent direct recomputation after rounding; never rely on cache.
    for start in range(0,len(source),4096):
        d=np.asarray(changed)[None,:,:]-source[start:start+4096,None,:]
        visible=(d*d).sum(axis=2)<=r2[start:start+4096,None]
        visible &= omni[start:start+4096,None]|((d*axes[start:start+4096,None,:]).sum(axis=2)>=0)
        if not visible.any(axis=1).all():return remaining
    return changed
