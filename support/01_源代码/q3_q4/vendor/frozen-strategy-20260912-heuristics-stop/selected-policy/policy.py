"""Experimental receipt-only policy; no scenario, seed, source count or truth input.

Uses exploratory measurements/clears with unchanged belief after no_signal or
failed clear, and a retained complete cover fallback. Floating geometry is a
numerical enclosure audit, not an outward-rounded formal certificate.
"""
import math
import numpy as np
from scipy.spatial import Delaunay,ConvexHull
import core

def hex_grid():
    s=970.
    points=[(s*(i+j/2),s*math.sqrt(3)*j/2) for i in range(-2,3) for j in range(-2,3)
            if max(abs(i),abs(j),abs(i+j))<=2]
    points += [(1875*math.cos(math.radians(30+60*k)),1875*math.sin(math.radians(30+60*k))) for k in range(6)]
    array=np.array(points)
    hull=ConvexHull(array)
    assert all(-b/np.linalg.norm(n)>1800 for *n,b in hull.equations)
    triangles=Delaunay(array).simplices
    max_edge=max(math.dist(points[i],points[j]) for tri in triangles for i in tri for j in tri)
    assert max_edge<1000
    return points

def center_radius(poly):
    angle=core.principal(poly)
    u,v=core.axes(angle)
    xs=[core.dot(p,u) for p in poly];ys=[core.dot(p,v) for p in poly]
    middle=core.world((min(xs)+max(xs))/2,(min(ys)+max(ys))/2,u,v)
    return middle,max(math.dist(middle,p) for p in poly),u,v,max(xs)-min(xs)

def solve(problem,action,variant,_forced=None,_observer=None):
    if variant=='tenth-v1':variant='corrsensor-2-10' if problem==3 else 'visibilityguard-1'
    if variant=='ninth-v1':variant='corrsensor-2-10' if problem==3 else 'visibleprobe-1'
    corr_clear=variant.startswith('corrclear-')
    corr_clear_radius=float(variant.split('-')[1]) if corr_clear else 0
    if corr_clear:variant='corrsensor-2-10'
    corr_sensor=variant.startswith('corrsensor-')
    sensor_parts=variant.split('-') if corr_sensor else []
    if corr_sensor:variant='correlated-'+sensor_parts[1]
    if variant=='eighth-v1':variant='correlated-0' if problem==3 else 'visibleprobe-1'
    visible_grid=variant.startswith('visiblegrid-')
    prior_guard=variant.startswith('visibilityguard-')
    prior_guard_level=int(variant.split('-')[1]) if prior_guard else 0
    if visible_grid or prior_guard:variant='visibleprobe-1'
    visible_probe=variant.startswith('visibleprobe-')
    visible_mode=int(variant.split('-')[1]) if visible_probe else 0
    if visible_probe:variant='merged-1' if problem==3 or visible_grid else 'edense-15-0'
    travel_micro=variant.startswith('travelmicro-')
    travel_mode=int(variant.split('-')[1]) if travel_micro else 0
    if travel_micro:variant='merged-1' if problem==3 else 'edense-15-0'
    correlated=variant.startswith('correlated-')
    correlated_mode=int(variant.split('-')[1]) if correlated else 0
    if correlated:variant='merged-1' if problem==3 else 'edense-15-0'
    controls=variant.startswith('controls-')
    control_values=tuple(map(float,variant.split('-')[1:])) if controls else ()
    if controls:variant='merged-1' if problem==3 else 'edense-15-0'
    q4_action=variant.startswith('q4act-')
    q4_action_threshold=int(variant.split('-')[1]) if q4_action else 0
    if q4_action:variant='edense-15-0'
    cover_plan=variant.startswith('coverplan-')
    cover_weight=float(variant.split('-')[1]) if cover_plan else 0
    if cover_plan:variant='eadapt-0' if problem==3 else 'edense-15-0'
    value_policy=variant.startswith('value1-')
    value_margin=float(variant.split('-')[1]) if value_policy else 0
    last_value_key=None
    if value_policy:variant='merged-1' if problem==3 else 'edense-15-0'
    staged=variant.startswith('staged-')
    stage_limit=int(variant.split('-')[1]) if staged else 0
    stage_scans=0
    if staged:variant='merged-1' if problem==3 else 'edense-15-0'
    safety_check=variant.startswith('scheck-')
    safety_done=False;safety_mode=False
    if safety_check:variant='smooth-'+variant.split('-')[1]
    harvest=variant.startswith('harvest-')
    harvest_margin=float(variant.split('-')[1]) if harvest else 0
    if harvest:variant='eadapt-0' if problem==3 else 'edense-15-0'
    stress_smooth=variant.startswith('sstress-')
    stress_level=int(variant.split('-')[1]) if stress_smooth else 0
    if stress_smooth:variant='smooth-'+variant.split('-')[2]
    guarded_smooth=variant.startswith('sguard-')
    smooth_guard=float(variant.split('-')[1]) if guarded_smooth else 0
    if guarded_smooth:variant='smooth-'+variant.split('-')[2]
    smooth_filter=variant.startswith('smoothf-')
    smooth_filter_mode=int(variant.split('-')[2]) if smooth_filter else 0
    if smooth_filter:variant='smooth-'+variant.split('-')[1]
    smooth_dense=variant.startswith('sdense-')
    smooth_density=int(variant.split('-')[1]) if smooth_dense else 0
    if smooth_dense:variant='smooth-'+variant.split('-')[2]
    smooth=variant.startswith('smooth-')
    smooth_passes=int(variant.split('-')[1]) if smooth else 0
    if smooth:variant='eadapt-0' if problem==3 else 'edense-15-0'
    micro_route=variant.startswith('mroute-')
    micro_route_weight=float(variant.split('-')[1]) if micro_route else 0
    if micro_route:variant='merged-1' if problem==3 else 'edense-15-0'
    lookahead=variant.startswith('lookahead-')
    lookahead_worlds=int(variant.split('-')[1]) if lookahead else 0
    if lookahead:variant='merged-1' if problem==3 else 'edense-15-0'
    lookahead_base=variant
    decision_index=0;forced_history=[];action_prefix=[];last_plan_key=None
    navigation=variant.startswith('nav-')
    navigation_bias=float(variant.split('-')[1]) if navigation else 0
    if navigation:variant='merged-1' if problem==3 else 'edense-15-0'
    negative_filter=variant.startswith('filter-') or smooth_filter or correlated
    filter_mode=correlated_mode if correlated else smooth_filter_mode if smooth_filter else int(variant.split('-')[1]) if negative_filter else 0
    if negative_filter and not smooth_filter and not correlated:variant='merged-1' if problem==3 else 'edense-15-0'
    if variant=='empirical-v4':
        variant='eadapt-0.0005' if problem==3 else 'edense-15-0'
    empirical_dense=variant.startswith('edense-')
    empirical_power=int(variant.split('-')[1]) if empirical_dense else 13
    if smooth_dense:empirical_power=smooth_density
    if empirical_dense:variant='eblend-'+variant.split('-')[2]
    empirical_blend=variant.startswith('eblend-')
    if empirical_blend:variant='eadapt-'+variant.split('-')[1]
    sticky=variant.startswith('sticky-')
    sticky_mode=int(variant.split('-')[1]) if sticky else 0
    retained_tour=[];last_discovery=-1
    if sticky:variant='merged-1'
    service_scan=variant.startswith('service-')
    service_values=tuple(map(float,variant.split('-')[1:])) if service_scan else ()
    if service_scan:variant='eadapt-'+str(service_values[0])
    empirical_config=variant.startswith('econf-')
    config_values=tuple(map(float,variant.split('-')[1:])) if empirical_config else ()
    if empirical_config:variant='eadapt-'+str(config_values[0])
    empirical_adapt=variant.startswith('eadapt-')
    empirical_tolerance=float(variant.split('-')[1]) if empirical_adapt else 0
    if empirical_adapt:variant='merged-1'
    empirical=variant.startswith('emp-')
    empirical_parts=variant.split('-')[1:] if empirical else []
    empirical_guard_done=False
    if empirical:variant='merged-1'
    rollout=variant.startswith('rollout-')
    rollout_worlds=int(variant.split('-')[1]) if rollout else 0
    rollout_decision=0
    if rollout:variant='merged-1'
    forecast=variant.startswith('forecast-')
    forecast_scale=float(variant.split('-')[1]) if forecast else 0
    if forecast:variant='skip-50' if problem==4 else 'skip-2000'
    merged=variant.startswith('merged-')
    merged_mode=int(variant.split('-')[1]) if merged else 0
    if merged:variant='skip-2000' if problem==3 else 'skip-50'
    informed=variant.startswith('informed-')
    information_weight=float(variant.split('-')[1]) if informed else 0
    if informed:variant='skip-50' if problem==4 else 'skip-2000'
    skip_known=variant.startswith('skip-')
    skip_radius=float(variant.split('-')[1]) if skip_known else 0
    if staged:skip_radius=50
    if skip_known:variant='micro-100' if problem==4 else 'micro-150'
    micro=variant.startswith('micro-')
    micro_distance=float(variant.split('-')[1]) if micro else 0
    if corr_sensor:micro_distance=float(sensor_parts[2])
    if controls:micro_distance*=control_values[0]
    if micro:variant='cleartry-45'
    if empirical_config:micro_distance=config_values[1]
    cleartry=variant.startswith('cleartry-')
    exploratory_radius=float(variant.split('-')[1]) if cleartry else 45
    if corr_clear:exploratory_radius=corr_clear_radius
    if controls:exploratory_radius=control_values[1]
    if cleartry:variant='bracket-6' if problem==4 else 'count-tune-90-250'
    if empirical_config:exploratory_radius=config_values[2]
    layered=variant.startswith('layered-')
    layer_radius=float(variant.split('-')[1]) if layered else 0
    layer_outer=False
    if layered:variant='bracket-6' if problem==4 else 'count-tune-90-250'
    early=variant.startswith('early-')
    negative_limit=int(variant.split('-')[1]) if early else 3
    if controls:negative_limit=int(control_values[4])
    if early:variant='bracket-6'
    bracket=variant.startswith('bracket-')
    bracket_steps=int(variant.split('-')[1]) if bracket else 0
    if bracket:variant='count-prefetch-1' if problem==4 else 'count-tune-90-250'
    count_stop=variant.startswith('count-')
    if count_stop:variant=variant.removeprefix('count-')
    prefetch=variant.startswith('prefetch-')
    prefetch_count=int(variant.split('-')[1]) if prefetch else 0
    if staged:prefetch_count=0
    if prefetch:variant='rings-8' if problem==4 else 'multi-8'
    belief=variant.startswith('belief-')
    belief_mode=variant.split('-')[1] if belief else ''
    if belief:variant='tune-90-250'
    base_geometry=globals()['center_radius']
    def center_radius(poly):
        middle,radius,u,v,width=base_geometry(poly)
        if controls and control_values[5] and radius>40:
            area=sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1]))
            if abs(area)>1e-6:
                centroid=tuple(sum((a[k]+b[k])*(a[0]*b[1]-b[0]*a[1]) for a,b in zip(poly,poly[1:]+poly[:1]))/(3*area) for k in (0,1))
                middle=tuple(middle[k]+control_values[5]*(centroid[k]-middle[k]) for k in (0,1))
                radius=max(math.dist(middle,p) for p in poly)
        if belief_mode in ('mean','both') and radius>80:
            twice_area=sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1]))
            if abs(twice_area)>1e-6:
                middle=tuple(sum((a[k]+b[k])*(a[0]*b[1]-b[0]*a[1]) for a,b in zip(poly,poly[1:]+poly[:1]))/(3*twice_area) for k in (0,1))
                radius=max(math.dist(middle,p) for p in poly)
        return middle,radius,u,v,width
    angular=variant.startswith('angular-')
    angular_gain=float(variant.split('-')[1])*math.pi/180 if angular else 0
    if angular:variant='tune-90-250'
    weighted=variant.startswith('weighted-')
    scan_weight=float(variant.split('-')[1]) if weighted else 0
    if weighted:variant='rings-8' if problem==4 else 'multi-8'
    rings=variant.startswith('rings-')
    rings_starts=int(variant.split('-')[1]) if rings else 0
    if rings:variant='tune-90-250'
    stationary=variant.startswith('stationary-')
    stationary_ratio=float(variant.split('-')[1]) if stationary else 0
    if stationary:variant='tune-90-250'
    polar=variant.startswith('polar-')
    polar_iterations=int(variant.split('-')[1]) if polar else 0
    if polar:variant='tune-90-250'
    if merged and problem==3:polar=True;polar_iterations=40
    if empirical or empirical_adapt:polar=False
    multi=variant.startswith('multi-')
    route_starts=int(variant.split('-')[1]) if multi else 0
    if multi:variant='tune-90-250'
    tuned=variant.startswith('tune-')
    tuning=tuple(map(float,variant.split('-')[1:])) if tuned else None
    if tuned:variant='act300' if problem==3 else 'seed100'
    if q4_action:variant='act'+str(q4_action_threshold)
    adaptive=variant.startswith('adaptive')
    routed=variant.startswith(('route','joint','prune','compact','flex','seed','act'))
    actionlevel=variant.startswith('act')
    seeded=variant.startswith('seed') or actionlevel
    seed_done=False
    if micro:seed_done=True
    if prefetch:seed_done=True
    flex=variant.startswith('flex')
    prune=variant.startswith('prune')
    if prune:
        from coverage import Coverage
        coverage=Coverage(problem,20)
    if flex:
        from coverage import Coverage
        coverage=Coverage(problem,40 if problem==3 else 20)
    joint=variant.startswith('joint')
    lateral=routed and 'side' in variant
    if adaptive:
        from coverage import Coverage
        coverage=Coverage(problem)
    anchors=list(core.POINTS) if problem==3 else hex_grid()
    if rings:
        assert problem==4
        from discovery import ring_grid
        anchors=ring_grid()
    if seeded and problem==3:
        anchors=[(0.,0.)]+[(1150*math.cos(k*math.pi/3),1150*math.sin(k*math.pi/3)) for k in range(6)]
    if flex and problem==3:
        anchors=[(0.,0.)]+[(1250*math.cos(k*math.pi/3),1250*math.sin(k*math.pi/3)) for k in range(6)]
    if variant.startswith('compact') and problem==3:
        r=float(variant.removeprefix('compact'))
        anchors=[(0.,0.)]+[(r*math.cos(k*math.pi/3),r*math.sin(k*math.pi/3)) for k in range(6)]
    if variant=='active36' and problem==4:
        anchors=[(float(x),float(y)) for x in range(-1800,1801,720) for y in range(-1800,1801,720)]
    fallback_anchors=list(anchors)
    if empirical:
        if empirical_parts[0]=='ring':
            n=int(empirical_parts[1]);r=float(empirical_parts[2])
            anchors=[(0.,0.)]+[(round(r*math.cos(2*math.pi*k/n),6),round(r*math.sin(2*math.pi*k/n),6)) for k in range(n)]
        elif empirical_parts[0]=='box':
            x,y,central=map(float,empirical_parts[1:])
            anchors=[(0.,central),(0.,-central)]+[(a*x,b*y) for a in (-1,1) for b in (-1,1)]
        elif empirical_parts[0]=='q4':
            from discovery import ring_grid
            outer,inner,ri,ro=map(float,empirical_parts[1:])
            anchors=ring_grid(outer=int(outer),inner=int(inner),r_inner=ri,r_outer=ro)
        else:raise ValueError('unknown empirical geometry')
    remaining=list(anchors)
    if safety_check:
        from empirical_discovery import EmpiricalDiscovery
        safety_prior=EmpiricalDiscovery(problem,0,13,2)
    if lookahead and _forced is None:
        from empirical_discovery import EmpiricalDiscovery
        rollout_prior=EmpiricalDiscovery(problem,0,12)
    if empirical_adapt:
        from empirical_discovery import EmpiricalDiscovery
        empirical_design=EmpiricalDiscovery(problem,empirical_tolerance,empirical_power,prior_guard_level or stress_level)
    if informed or forecast or rollout:
        from search_prior import SearchPrior
        discovery_prior=SearchPrior(problem)
    if angular:
        assert problem==3
        from angular import AngularCover
        angle_cover=AngularCover();remaining=[(0.,0.)]
    if polar:
        assert problem==3
        ring=anchors[1:]
    if adaptive:
        # Dense retained fallback supports the conservative cell certificate.
        remaining=[(float(x),float(y)) for x in range(-2000,2001,500)
                   for y in range(-2000,2001,500) if math.hypot(x,y)<2600]
    position=(0.,0.);cleared=set();known={};history={c:[] for c in range(1,21)}
    counts={'discovery':0,'localize':0,'exploratory_clear':0,'guaranteed_clear':0,'fallback_clear':0}
    if value_policy:counts.update(value_decisions=0,value_changes=0)
    progress={};negative_counts={};fallbacks={}
    geometry_cache={}
    def source_geometry(c):
        geometry=center_radius(known[c])
        if not negative_filter:return geometry
        key=len(history[c])
        if c not in geometry_cache or geometry_cache[c][0]!=key:
            if correlated:
                from correlated_bearing import estimate
            else:
                from belief_filter import estimate
            geometry_cache[c]=(key,estimate(known[c],history[c],problem,geometry,filter_mode))
        return geometry_cache[c][1]
    def send(kind,p,c,purpose):
        nonlocal position
        p=tuple(round(float(v),6) for v in p)
        obs=action(kind,p,c)
        if lookahead and _forced is None:action_prefix.append((kind,p,c,dict(obs)))
        counts[purpose]+=1
        history[c].append((kind,p,obs))
        position=p
        if kind=='measure' and obs['kind'] in ('near','direction'):
            known[c]=core.initial_polygon(p,obs) if c not in known else core.posterior(known[c],p,obs)
            if belief_mode in ('disk','both'):
                for k in range(64):
                    normal=(math.cos(k*math.pi/32),math.sin(k*math.pi/32))
                    known[c]=core.clip(known[c],normal,1800+1e-5)
            if not known[c]:raise ValueError('empty belief')
        if kind=='clear' and obs['kind']=='success':
            cleared.add(c);known.pop(c,None)
        return obs
    def measure_once(p,c,purpose):
        p=tuple(round(float(v),6) for v in p)
        for kind,q,obs in history[c]:
            if kind=='measure' and q==p:return obs
        return send('measure',p,c,purpose)
    def scan(p):
        prior_known=set(known)
        for c in range(1,21):
            if c not in cleared and (not (adaptive or prune) or c not in known):
                if skip_known and c in known and center_radius(known[c])[1]<skip_radius:continue
                measure_once(p,c,'discovery' if c not in known else 'localize')
        if adaptive or prune or flex:coverage.add(p)
        if angular:angle_cover.add(p)
        if informed or forecast or rollout:discovery_prior.add(p)
        if empirical_adapt:empirical_design.add(p)
        if safety_check:safety_prior.add(p)
        if lookahead and _forced is None:rollout_prior.add(p)
        if micro and not ((service_scan or harvest) and scan_at_service) and not (staged and stage_scans<stage_limit):
            newly=[c for c in known if c not in prior_known and center_radius(known[c])[1]>100]
            if newly:
                angles=[k*math.pi/6 for k in range(12)]
                angles += [math.atan2(center_radius(known[c])[0][1]-p[1],center_radius(known[c])[0][0]-p[0])+sign*math.pi/2
                           for c in newly for sign in (-1,1)]
                options=[(p[0]+micro_distance*math.cos(a),p[1]+micro_distance*math.sin(a)) for a in angles]
                if micro_route:
                    options=[p]+[(p[0]+r*math.cos(a),p[1]+r*math.sin(a)) for r in (75,150,300,450) for a in angles]
                    from routing import route
                    route_nodes=[('source',c,source_geometry(c)[0]) for c in known]
                    route_nodes += [('anchor',anchor,anchor) for anchor in remaining]
                def score(q):
                    gain=0.
                    for c in newly:
                        mid,r,*_=center_radius(known[c])
                        bearing=math.degrees(math.atan2(mid[1]-q[1],mid[0]-q[0]))%360
                        poly=core.posterior(known[c],q,dict(kind='direction',bearing=bearing))
                        if poly:gain+=math.log(max(r,1)/max(center_radius(poly)[1],1))
                    if micro_route:
                        tour=route(route_nodes,q,1)
                        path=[p,q]+[n[2] for n in tour]
                        length=sum(math.dist(a,b) for a,b in zip(path,path[1:]))
                        return micro_route_weight*gain-length
                    return gain
                if travel_micro:
                    targets=[center_radius(known[c])[0] for c in newly]
                    if travel_mode<4:
                        target=min(targets,key=lambda t:math.dist(p,t))
                        a=math.atan2(target[1]-p[1],target[0]-p[0])
                        fraction=(.5,.75,1.,1.25)[travel_mode]
                        near=tuple(p[k]+fraction*(target[k]-p[k]) for k in (0,1))
                        options=[(near[0]+sign*50*math.sin(a),near[1]-sign*50*math.cos(a)) for sign in (-1,1)]
                    else:
                        from routing import route
                        route_nodes=[('source',c,source_geometry(c)[0]) for c in known]
                        route_nodes += [('anchor',anchor,anchor) for anchor in remaining]
                        for target in targets:
                            a=math.atan2(target[1]-p[1],target[0]-p[0])
                            for fraction in (.5,.75,1.):
                                near=tuple(p[k]+fraction*(target[k]-p[k]) for k in (0,1))
                                options.extend((near[0]+sign*50*math.sin(a),near[1]-sign*50*math.cos(a)) for sign in (-1,1))
                        original_score=score
                        def score(q):
                            tour=route(route_nodes,q,1)
                            path=[p,q]+[n[2] for n in tour]
                            length=sum(math.dist(a,b) for a,b in zip(path,path[1:]))
                            return (100,300,600,1000)[travel_mode-4]*original_score(q)-length
                q=max(options,key=score)
                for c in newly:measure_once(q,c,'localize')
    def clear_source(c):
        if stationary:
            mid,rad,*_=center_radius(known[c])
            if rad>30 and not any(kind=='measure' and math.dist(q,position)<.01 for kind,q,o in history[c]):
                angle=math.degrees(math.atan2(mid[1]-position[1],mid[0]-position[0]))%360
                predicted=core.posterior(known[c],position,dict(kind='direction',bearing=angle))
                if predicted and center_radius(predicted)[1]<rad*stationary_ratio:
                    measure_once(position,c,'localize')
                    return
        if actionlevel and c in fallbacks:
            q=min(fallbacks[c],key=lambda p:math.dist(position,p));fallbacks[c].remove(q)
            if send('clear',q,c,'fallback_clear')['kind']=='success':return
            if not fallbacks[c]:raise ValueError('complete cover exhausted without success')
            return
        negatives=negative_counts.get(c,0) if actionlevel else 0
        start=progress.get(c,0) if actionlevel else 0
        for attempt in range(start,min(start+1,8) if actionlevel else 8):
            progress[c]=attempt+1
            middle,radius,u,v,width=source_geometry(c)
            safe_radius=max(math.dist(middle,p) for p in known[c]) if negative_filter else radius
            if radius<19.8 or (radius<exploratory_radius and (attempt>=1 or cleartry)):
                clear_point=middle
                if merged and merged_mode and safe_radius<19.8:
                    distance=math.dist(position,middle)
                    step=min(distance,19.99-safe_radius)
                    if distance>1e-8:clear_point=tuple(middle[k]+(position[k]-middle[k])*step/distance for k in (0,1))
                obs=send('clear',clear_point,c,'guaranteed_clear' if safe_radius<19.8 else 'exploratory_clear')
                if obs['kind']=='success':return
            # Geometry-driven transverse measurement, approached from nearest end.
            offset=min(120,max(30,width*.15))
            advance=min(100,width*.20)
            if tuned:
                offset=min(tuning[0],max(10,width*.15))
                advance=min(tuning[1],width*.20)
            if controls:
                offset=min(tuning[0],max(10,width*control_values[2]))
                advance=min(tuning[1],width*control_values[3])
            candidates=[(middle[0]+sign*v[0]*offset+end*u[0]*advance,
                         middle[1]+sign*v[1]*offset+end*u[1]*advance)
                        for sign in (-1,1) for end in (-1,1)]
            if lateral and attempt==0 and radius>100:
                positives=[(q,o) for kind,q,o in history[c] if kind=='measure' and o['kind']=='direction']
                if len(positives)==1:
                    first,o=positives[0];along,across=core.axes(o['bearing'])
                    distance=float(variant.split('side')[-1] or 150)
                    candidates=[(first[0]+sign*across[0]*distance+along[0]*50,
                                 first[1]+sign*across[1]*distance+along[1]*50) for sign in (-1,1)]
            if visible_probe and problem==4:
                positives=[p for k,p,o in history[c] if k=='measure' and o['kind'] in ('direction','near')]
                for p in positives:
                    distance=math.dist(p,middle)
                    if distance<1:continue
                    along=tuple((p[k]-middle[k])/distance for k in (0,1))
                    across=(-along[1],along[0])
                    for side in (-1,1):
                        for forward in (.5,1.,2.):
                            candidates.append(tuple(middle[k]+forward*offset*along[k]+side*offset*.5*across[k] for k in (0,1)))
            used=[q for kind,q,o in history[c] if kind=='measure']
            candidates=[p for p in candidates if all(math.dist(p,q)>2 for q in used)]
            if not candidates:break
            if visible_probe and problem==4:
                from visibility_probe import probabilities
                ps=probabilities(known[c],history[c],(middle,radius,u,v,width),candidates)
                penalty=(0,200,600,1500,5000)[visible_mode]
                q=min(zip(candidates,ps),key=lambda pair:math.dist(position,pair[0])+penalty*(1-pair[1]))[0]
            else:q=min(candidates,key=lambda p:math.dist(position,p))
            obs=measure_once(q,c,'localize')
            if actionlevel and int(variant.removeprefix('act')):
                for other in list(known):
                    if other==c:continue
                    _,other_radius,*_=center_radius(known[other])
                    if other_radius>int(variant.removeprefix('act')):measure_once(q,other,'localize')
            if joint:
                for other in list(known):
                    if other==c:continue
                    _,other_radius,*_=center_radius(known[other])
                    threshold=float(variant.removeprefix('joint') or 40)
                    if other_radius>threshold:measure_once(q,other,'localize')
            if obs['kind']=='no_signal':negatives+=1
            negative_counts[c]=negatives
            if negatives>=negative_limit:break
        if actionlevel and progress.get(c,0)<8 and negatives<negative_limit:return
        if c in cleared:return
        if bracket and negatives>=negative_limit:
            positives=[(p,o) for k,p,o in history[c] if k=='measure' and o['kind']=='direction']
            negatives_seen=[p for k,p,o in history[c] if k=='measure' and o['kind']=='no_signal']
            if positives and negatives_seen:
                p,o=min(positives,key=lambda pair:math.dist(position,pair[0]))
                along,_=core.axes(o['bearing'])
                distances=[core.dot((q[0]-p[0],q[1]-p[1]),along) for q in negatives_seen]
                distances=[d for d in distances if d>10]
                if distances:
                    low=0.;high=min(distances)
                    q=(p[0]+along[0]*high,p[1]+along[1]*high)
                    first=measure_once(q,c,'localize')
                    if first['kind']=='no_signal':
                        for _ in range(bracket_steps):
                            mid=(low+high)/2
                            q=(p[0]+along[0]*mid,p[1]+along[1]*mid)
                            if high-low<35:
                                if send('clear',q,c,'exploratory_clear')['kind']=='success':return
                                break
                            obs=measure_once(q,c,'localize')
                            if obs['kind']=='no_signal':high=mid
                            else:low=mid
                            if obs['kind']=='near':
                                if send('clear',q,c,'exploratory_clear')['kind']=='success':return
        # Purely exploratory estimate is never counted as a guaranteed clear.
        middle,*_=source_geometry(c)
        if send('clear',middle,c,'exploratory_clear')['kind']=='success':return
        if actionlevel:
            if c not in fallbacks:fallbacks[c]=list(core.cover(known[c],position,position).points)
            plan=[fallbacks[c].pop(0)] if fallbacks[c] else []
        else:plan=core.cover(known[c],position,position).points
        for q in plan:
            if send('clear',q,c,'fallback_clear')['kind']=='success':return
        if actionlevel and fallbacks.get(c):return
        raise ValueError('complete cover exhausted without success')
    # Each anchor is visited, but active-source service and future anchor order
    # follow current receipt-derived geometry. No hidden source-count stopping.
    scan_at_service=False
    while remaining or known or (empirical and not empirical_guard_done) or (safety_check and not safety_done):
        if len(cleared)==16:return dict(completed=True,reason='maximum_count_reached',counts=counts)
        if safety_check and not remaining and not known:
            if not safety_prior.unseen.any():
                safety_done=True;break
            safety_mode=True
            points=safety_prior.points[safety_prior.unseen]
            choices=[tuple(p) for p in np.unique(points,axis=0)]
            if len(choices)>128:choices=choices[::max(1,len(choices)//128)]
            choices += fallback_anchors
            def guard_gain(p):
                hit=int((safety_prior.unseen & safety_prior.visible(p)).sum())
                return hit/(math.dist(position,p)+200)
            remaining=[max(choices,key=guard_gain)]
            if guard_gain(remaining[0])<=0:raise ValueError('safety sample cover stalled')
        if empirical and not remaining and not known:
            empirical_guard_done=True
            if len(cleared)<10:
                remaining=list(fallback_anchors)
            else:break
        if count_stop and len(cleared)+len(known)==16:remaining=[]
        if staged and remaining and stage_scans<stage_limit:
            from routing import route
            q=route([('anchor',p,p) for p in remaining],position,8)[0][1]
            remaining.remove(q);scan(q);stage_scans+=1
            continue
        if prefetch and remaining and counts['discovery'] and prefetch_count:
            q=min(remaining,key=lambda p:math.dist(position,p))
            remaining.remove(q);scan(q);prefetch_count-=1
            continue
        if seeded and not seed_done and known:
            seed_done=True
            distance=100. if actionlevel else float(variant.removeprefix('seed'))
            options=[(distance*math.cos(k*math.pi/6),distance*math.sin(k*math.pi/6)) for k in range(12)]
            def score(q):
                gain=0
                for c in known:
                    mid,r,*_=center_radius(known[c])
                    bearing=math.degrees(math.atan2(mid[1]-q[1],mid[0]-q[0]))%360
                    poly=core.posterior(known[c],q,dict(kind='direction',bearing=bearing))
                    if poly:
                        new=center_radius(poly)[1]
                        gain+=max(0,math.log(max(r,1)/max(new,1)))
                return gain
            q=max(options,key=score)
            for c in list(known):measure_once(q,c,'localize')
        if angular and angle_cover.points:
            remaining=angle_cover.pending(position)
            if not remaining and not known:break
        if flex and remaining and known:
            # Relocate discovery waypoints towards receipt-derived source
            # estimates only when the whole planned cover remains valid.
            for c in known:
                target=center_radius(known[c])[0]
                p=min(remaining,key=lambda p:math.dist(p,target))
                if math.dist(p,target)>600 or math.dist(p,target)<5:continue
                for fraction in (1.,.75,.5,.25):
                    q=tuple(round(p[i]+fraction*(target[i]-p[i]),6) for i in range(2))
                    if coverage.covered(coverage.points+[x for x in remaining if x!=p]+[q]).all():
                        remaining.remove(p);remaining.append(q);break
        if prune and remaining:
            for p in sorted(remaining,key=lambda p:-math.dist(position,p)):
                if coverage.covered(coverage.points+[q for q in remaining if q!=p]).all():
                    remaining.remove(p)
        if adaptive and coverage.complete and not known:
            return dict(completed=True,reason='adaptive_coverage_complete',counts=counts,
                        discovery_points=len(coverage.points),cleared=sorted(cleared))
        choice=None
        if empirical_adapt and counts['discovery'] and not safety_mode:
            if cover_plan and len(known)+len(cleared)<16:
                from probe_cover import choose as choose_cover
                remaining=choose_cover(empirical_design,remaining,fallback_anchors,
                    [source_geometry(c)[0] for c in known],position,30*(20-len(known)-len(cleared))*cover_weight)
            else:
                remaining=empirical_design.adjust(remaining,[source_geometry(c)[0] for c in known],position,empirical_blend)
            if smooth and remaining:
                from smooth_discovery import adjust as smooth_adjust
                remaining=smooth_adjust(empirical_design,remaining,[source_geometry(c)[0] for c in known],position,smooth_passes,smooth_guard)
            if not remaining and not known:break
        if polar and known:
            from flexroute import adjust
            ring,remaining=adjust(ring,remaining,[source_geometry(c)[0] for c in known],position,polar_iterations)
        if routed:
            nodes=[('source',c,min(fallbacks[c],key=lambda p:math.dist(position,p)) if c in fallbacks
                    else source_geometry(c)[0]) for c in known]
            nodes += [('anchor',p,p) for p in remaining]
            if forecast:
                nodes += [('forecast',i,p) for i,p in enumerate(discovery_prior.forecast(20-len(known)-len(cleared),forecast_scale))]
            if layered and not layer_outer:
                inner_nodes=[n for n in nodes if math.hypot(*n[2])<
                             (layer_radius if n[0]=='source' else (1100 if problem==4 else 1))]
                if inner_nodes:nodes=inner_nodes
                else:layer_outer=True
            pending=list(nodes);tour=[];last=position
            while pending:
                n=min(pending,key=lambda n:math.dist(last,n[2]));pending.remove(n)
                tour.append(n);last=n[2]
            for _ in range(5):
                improved=False
                for i in range(len(tour)-1):
                    a=position if i==0 else tour[i-1][2]
                    for j in range(i+1,len(tour)):
                        b=tour[i][2];cpoint=tour[j][2]
                        before=math.dist(a,b);after=math.dist(a,cpoint)
                        if j+1<len(tour):
                            d=tour[j+1][2];before+=math.dist(cpoint,d);after+=math.dist(b,d)
                        if after<before-1e-6:
                            tour[i:j+1]=reversed(tour[i:j+1]);improved=True
                if not improved:break
            choice=next(n for n in tour if n[0]!='forecast')
            if multi or (rings and rings_starts):
                from routing import route
                choice=next(n for n in route(nodes,position,rings_starts if rings else route_starts,scan_weight) if n[0]!='forecast')
            if sticky:
                from routing import route,persistent_route
                if not retained_tour or (sticky_mode==2 and counts['discovery']!=last_discovery):
                    retained_tour=route(nodes,position,8)
                else:retained_tour=persistent_route(nodes,retained_tour,position,sticky_mode==3)
                last_discovery=counts['discovery']
                choice=retained_tour[0]
            if navigation:
                from routing import route
                if navigation_bias>=1000:
                    choice=min(nodes,key=lambda n:math.dist(position,n[2]))
                else:
                    def cost(first):
                        rest=route([n for n in nodes if n!=first],first[2],2)
                        path=[position,first[2]]+[n[2] for n in rest]
                        return sum(math.dist(a,b) for a,b in zip(path,path[1:]))+navigation_bias*math.dist(position,first[2])
                    choice=min(nodes,key=cost)
            if informed:
                from routing import route
                best_score=math.inf
                for first in nodes:
                    q=first[2]
                    rest=route([n for n in nodes if n!=first],q,2)
                    path=[position,q]+[n[2] for n in rest]
                    value=sum(math.dist(a,b) for a,b in zip(path,path[1:]))
                    if first[0]=='anchor':value-=information_weight*discovery_prior.gain(q,20-len(known)-len(cleared))
                    if value<best_score:best_score=value;choice=first
        if rollout:
            from rollout_planner import choose
            choice=choose(nodes,choice,position,known,cleared,discovery_prior,rollout_worlds,rollout_decision)
            rollout_decision+=1
        if value_policy:
            key=(len(known),len(cleared),counts['discovery'])
            if known and len(nodes)>2 and key!=last_value_key:
                from routing_features import candidates_and_features
                from routing_model_v1 import predict
                options,features=candidates_and_features(dict(problem=problem,position=position,
                    nodes=nodes,choice=choice,known=known,cleared=cleared,history=history))
                predicted=[predict(problem,x) for x in features]
                best=min(range(len(options)),key=lambda i:predicted[i])
                counts['value_decisions']+=1
                if predicted[best]<predicted[0]-value_margin:
                    choice=options[best];counts['value_changes']+=int(best!=0)
                last_value_key=key
        if _forced is not None and decision_index<len(_forced):
            selected=_forced[decision_index]
            matches=[n for n in nodes if n[0]==selected[0] and n[1]==selected[1]]
            assert len(matches)==1,'forced node mismatch'
            choice=matches[0]
        if lookahead and _forced is None:
            key=(len(cleared),counts['discovery'])
            if known and len(nodes)>2 and key!=last_plan_key:
                from policy_lookahead import choose
                choice,diagnostic=choose(solve,problem,lookahead_base,nodes,choice,known,cleared,
                    history,rollout_prior,action_prefix,forced_history,lookahead_worlds,decision_index)
                last_plan_key=key
        if _observer is not None:
            _observer(dict(problem=problem,decision=decision_index,position=position,
                nodes=nodes,choice=choice,known=known,cleared=cleared,history=history,
                counts=counts,forced=forced_history))
        forced_history.append(choice)
        decision_index+=1
        if known and (not routed or choice[0]=='source'):
            c=min(known,key=lambda c:math.dist(position,center_radius(known[c])[0]))
            if routed:c=choice[1]
            target=center_radius(known[c])[0]
            nearest=min(remaining,key=lambda p:math.dist(position,p)) if remaining else None
            if routed or adaptive or nearest is None or math.dist(position,target)<math.dist(position,nearest)*1.3:
                clear_source(c)
                if harvest and c in cleared and remaining:
                    from routing import route
                    centers=[source_geometry(channel)[0] for channel in known]
                    def planned_length(probes):
                        nodes=[('source',i,p) for i,p in enumerate(centers)]+[('anchor',p,p) for p in probes]
                        tour=route(nodes,position,4)
                        path=[position]+[n[2] for n in tour]
                        return sum(math.dist(a,b) for a,b in zip(path,path[1:]))
                    old_length=planned_length(remaining)
                    unseen_before=empirical_design.unseen.copy()
                    empirical_design.unseen &= ~empirical_design.visible(position)
                    trial=empirical_design.adjust(remaining,centers,position,empirical_blend)
                    empirical_design.unseen=unseen_before
                    channels_now=sum(channel not in cleared and not (skip_known and channel in known and center_radius(known[channel])[1]<skip_radius) for channel in range(1,21))
                    # Estimate absent-channel cost with the public count prior,
                    # never the actual number of sources in this scenario.
                    absent=max(4,20-max(13,len(cleared)+len(known)))
                    action_delta=30*(channels_now-absent*(len(remaining)-len(trial)))
                    if old_length-planned_length(trial)>action_delta+harvest_margin:
                        scan_at_service=True;scan(position);scan_at_service=False
                        remaining=trial
                if service_scan and c in cleared and remaining:
                    hit=int((empirical_design.unseen & empirical_design.visible(position)).sum())
                    if hit/len(empirical_design.unseen)>=service_values[1]:
                        scan_at_service=True
                        scan(position)
                        scan_at_service=False
                if angular and c in cleared and angle_cover.gain(position)>angular_gain:
                    scan(position)
                if prune and remaining:scan(position)
                if adaptive and not coverage.complete:
                    scan(position)
                if variant in ('opportunistic','opportunistic36'):
                    scan(position)
                continue
        if routed:q=choice[1]
        elif adaptive:
            q=max(remaining,key=lambda p:coverage.gain(p)/(math.dist(position,p)+float(variant.removeprefix('adaptive') or 300)))
        else:q=min(remaining,key=lambda p:math.dist(position,p))
        remaining.remove(q);scan(q)
    if angular:
        assert angle_cover.points and not angle_cover.gaps()
        return dict(completed=True,reason='angular_annulus_covered',counts=counts,
                    discovery_points=len(angle_cover.points),discovery_positions=angle_cover.points,cleared=sorted(cleared))
    return dict(completed=True,reason='empirical_scan_exhausted' if empirical or empirical_adapt else 'guaranteed_grid_exhausted',counts=counts,
                discovery_points=len(anchors),cleared=sorted(cleared))
