"""Full-policy counterfactual rollouts in sampled, receipt-conditioned worlds.

These worlds and synthetic future bearing errors are approximations, not
observations or certification. No arena, scenario, actual seed or count input.
"""
import math
import numpy as np


def sample_known(poly,history,problem,rng):
    polygon=np.asarray(poly)
    triangles=np.asarray([(polygon[0],polygon[j],polygon[j+1]) for j in range(1,len(polygon)-1)])
    if not len(triangles):
        point=polygon.mean(axis=0)
        return dict(point=point,radius=1500.,direction=0.,omni=True,fallback=True)
    sides=triangles[:,1:]-triangles[:,:1]
    areas=np.abs(sides[:,0,0]*sides[:,1,1]-sides[:,0,1]*sides[:,1,0])
    weights=areas/areas.sum() if areas.sum()>1e-12 else np.full(len(areas),1/len(areas))
    n=4096;selected=triangles[rng.choice(len(triangles),n,p=weights)]
    z=rng.random((n,2));s=np.sqrt(z[:,0])
    points=(1-s[:,None])*selected[:,0]+(s*(1-z[:,1]))[:,None]*selected[:,1]+(s*z[:,1])[:,None]*selected[:,2]
    radius=1000+500*rng.random(n);angles=rng.uniform(-math.pi,math.pi,n)
    axes=np.column_stack([np.cos(angles),np.sin(angles)])
    omni=np.ones(n,dtype=bool) if problem==3 else rng.random(n)<.5
    valid=(points*points).sum(axis=1)<=1770**2+1e-5
    likelihood=np.zeros(n)
    for kind,p,o in history:
        d=np.asarray(p)-points;distance=np.linalg.norm(d,axis=1)
        visible=(distance<=radius)&(omni|((d*axes).sum(axis=1)>=0))
        if kind=='clear':
            if o['kind']!='success':valid &= distance>20-1e-5
        elif o['kind']=='no_signal':valid &= ~visible
        else:
            valid &= visible
            if o['kind']=='near':valid &= distance<=5+1e-5
            else:
                error=(np.degrees(np.arctan2(-d[:,1],-d[:,0]))-o['bearing']+180)%360-180
                valid &= np.abs(error)<=1.011
                likelihood-=.5*(error/.4)**2
    ix=np.flatnonzero(valid)
    fallback=not len(ix)
    if fallback:
        # A forecast-only fallback; it never alters the real source polygon.
        positives=[np.asarray(p) for kind,p,o in history if kind=='measure' and o['kind']!='no_signal']
        point=polygon.mean(axis=0);direction=np.mean(positives,axis=0)-point if positives else np.array([1.,0.])
        return dict(point=point,radius=1500.,direction=math.atan2(direction[1],direction[0]),omni=problem==3,fallback=True)
    w=np.exp(likelihood[ix]-likelihood[ix].max());i=int(rng.choice(ix,p=w/w.sum()))
    return dict(point=points[i],radius=float(radius[i]),direction=float(angles[i]),omni=bool(omni[i]),fallback=False)


def sample_worlds(problem,known,cleared,history,prior,number,decision):
    rng=np.random.default_rng(911223+decision*193)
    k=len(known)+len(cleared)
    mass=float(prior.unseen.mean())
    counts=np.arange(max(10,k),17)
    probabilities=np.array([math.comb(int(n),k)*max(mass,1e-12)**(int(n)-k) for n in counts])
    probabilities/=probabilities.sum()
    pool=np.flatnonzero(prior.unseen)
    channels=sorted(set(range(1,21))-set(known)-set(cleared))
    worlds=[]
    for _ in range(number):
        n=int(rng.choice(counts,p=probabilities)) if len(pool) else k
        world={c:sample_known(poly,history[c],problem,rng) for c,poly in known.items()}
        for c in rng.choice(channels,n-k,replace=False) if n>k else []:
            i=int(rng.choice(pool))
            world[int(c)]=dict(point=prior.points[i],radius=float(math.sqrt(prior.receive2[i])),
                direction=math.atan2(prior.axes[i,1],prior.axes[i,0]),omni=bool(prior.omni[i]),fallback=False)
        worlds.append((world,max(n,1),float(rng.uniform(0,2*math.pi))))
    return worlds


def choose(solve,problem,base_variant,nodes,default,known,cleared,history,prior,
           prefix,forced,world_count,decision):
    position=prefix[-1][1] if prefix else (0.,0.)
    candidates=[default]
    for kind in ('source','anchor'):
        options=[n for n in nodes if n[0]==kind and n not in candidates]
        if options:candidates.append(min(options,key=lambda n:math.dist(position,n[2])))
    if len(candidates)<2:return default,dict(simulations=0,fallback_world_sources=0)
    worlds=sample_worlds(problem,known,cleared,history,prior,world_count,decision)
    scores=[];fallbacks=sum(s['fallback'] for w,n,phase in worlds for s in w.values())
    for candidate in candidates:
        values=[]
        for sources,total,phase in worlds:
            state=dict(index=0,position=position,detector=1,time=0.,actions=0)
            for kind,p,c,o in prefix:
                if kind=='measure':state['detector']=c
            active=dict(sources)
            def action(kind,p,c):
                index=state['index'];state['index']+=1
                if index<len(prefix):
                    old_kind,old_p,old_c,old_o=prefix[index]
                    assert kind==old_kind and c==old_c and math.dist(p,old_p)<2e-6,'prefix mismatch'
                    return dict(old_o)
                state['actions']+=1
                if state['actions']>750:raise TimeoutError('forecast action cap')
                state['time']+=math.dist(state['position'],p)/5;state['position']=p
                source=active.get(c)
                distance=math.dist(p,source['point']) if source else math.inf
                if kind=='clear':
                    success=source is not None and distance<=20
                    state['time']+=5 if success else 3
                    if success:active.pop(c)
                    return dict(kind='success' if success else 'no_target_in_range')
                state['time']+=5+int(state['detector']!=c);state['detector']=c
                visible=source is not None and distance<=source['radius']
                if visible and not source['omni']:
                    d=np.asarray(p)-source['point']
                    visible=d[0]*math.cos(source['direction'])+d[1]*math.sin(source['direction'])>=0
                if not visible:return dict(kind='no_signal')
                if distance<=5:return dict(kind='near')
                d=source['point']-np.asarray(p)
                error=.6*math.sin(p[0]/170+phase+c)*math.cos(p[1]/170-phase-c)
                bearing=(math.degrees(math.atan2(d[1],d[0]))+error)%360
                return dict(kind='direction',bearing=round(bearing,2)%360)
            try:
                solve(problem,action,base_variant,_forced=forced+[candidate])
                cost=state['time']/total+5000*len(active)
            except TimeoutError:
                cost=state['time']/total+5000*len(active)
            values.append(cost)
        scores.append(float(np.mean(values)))
    best=int(np.argmin(scores))
    # Require a small margin over the baseline to reduce Monte Carlo chatter.
    if scores[0]-scores[best]<2:best=0
    return candidates[best],dict(simulations=len(candidates)*len(worlds),
        fallback_world_sources=fallbacks,scores=scores,changed=best!=0)
