"""Planning-only posterior event rollouts; no dataset/seed/true-world input.

Future localizations are approximated as revealing position after a paid
second measurement. These forecasts are not execution or completion proofs.
The real controller still retains its full geometric fallback and receipts.
"""
import math
import numpy as np
from routing import route

def sample_polygon(poly,rng):
    p=np.array(poly);origin=p[0]
    areas=np.abs(np.cross(p[1:-1]-origin,p[2:]-origin))
    if areas.sum()<1e-8:return tuple(p.mean(axis=0))
    for _ in range(12):
        i=int(rng.choice(len(areas),p=areas/areas.sum()))+1
        a,b=rng.random(2)
        if a+b>1:a,b=1-a,1-b
        point=origin+a*(p[i]-origin)+b*(p[i+1]-origin)
        if np.linalg.norm(point)<=1800:return tuple(point)
    return tuple(p.mean(axis=0))

def sample_world(prior,known,cleared,rng):
    known_count=len(known)+len(cleared)
    active_weights=prior.weights*prior.active
    mass=float(active_weights.sum())
    choices=list(range(max(10,known_count),17))
    probs=np.array([math.comb(20-known_count,n-known_count)/math.comb(20,n)*mass**(n-known_count) for n in choices])
    if not probs.sum():number=known_count
    else:number=int(rng.choice(choices,p=probs/probs.sum()))
    sources={c:(sample_polygon(poly,rng),1500,-1) for c,poly in known.items()}
    channels=[c for c in range(1,21) if c not in known and c not in cleared]
    missing=number-known_count
    if missing:
        selected=rng.choice(channels,size=missing,replace=False)
        particles=rng.choice(len(active_weights),size=missing,p=active_weights/mass)
        for c,j in zip(selected,particles):
            sources[int(c)]=(tuple(prior.points[j]),float(prior.radii[j]),int(prior.directions[j]))
    return sources,number

def visible(source,p):
    g,r,theta=source;dx=p[0]-g[0];dy=p[1]-g[1]
    return math.hypot(dx,dy)<=r and (theta<0 or dx*math.cos(theta*math.pi/6)+dy*math.sin(theta*math.pi/6)>=0)

def future_cost(first,anchors,estimates,sources,position,cleared_count,number):
    remaining=list(anchors);known=dict(estimates);hidden={c:s for c,s in sources.items() if c not in known}
    cost=0.;choice=first;newly_cleared=0
    while remaining or known:
        kind,key,estimate=choice
        if kind=='source':
            # Current known sources may still need information at their estimate.
            g=sources[key][0]
            cost+=math.dist(position,estimate)/5+math.dist(estimate,g)/5+5
            position=g;known.pop(key);newly_cleared+=1
        else:
            p=key;remaining.remove(p)
            cost+=math.dist(position,p)/5+6*(20-cleared_count-newly_cleared-len(known))
            position=p;found=[]
            for c,s in hidden.items():
                if visible(s,p):found.append(c)
            if found:cost+=15+5*len(found)
            for c in found:known[c]=hidden.pop(c)[0]
        if cleared_count+newly_cleared+len(known)==16:remaining=[]
        nodes=[('source',c,p) for c,p in known.items()]+[('anchor',p,p) for p in remaining]
        if not nodes:break
        choice=route(nodes,position,1)[0]
    return cost/max(1,number)

def choose(nodes,baseline,position,known,cleared,prior,worlds,decision):
    candidates=[baseline]
    candidates += sorted(nodes,key=lambda n:math.dist(position,n[2]))[:3]
    anchors=[n for n in nodes if n[0]=='anchor']
    if anchors:
        candidates.append(max(anchors,key=lambda n:prior.gain(n[2],20-len(known)-len(cleared))/(100+math.dist(position,n[2]))))
    candidates=list(dict.fromkeys(candidates))
    if len(candidates)<2:return baseline
    rng=np.random.default_rng(19301+decision)
    samples=[sample_world(prior,known,cleared,rng) for _ in range(worlds)]
    estimates={n[1]:n[2] for n in nodes if n[0]=='source'}
    best=None;best_value=math.inf
    for first in candidates:
        value=sum(future_cost(first,[n[1] for n in anchors],estimates,sources,position,len(cleared),number)
                  for sources,number in samples)/worlds
        if value<best_value:best=first;best_value=value
    return best
