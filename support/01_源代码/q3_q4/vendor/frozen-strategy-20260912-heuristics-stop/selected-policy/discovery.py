"""Whole-region directional discovery audits for compact concentric grids."""
import math
import numpy as np

def ring_grid(outer=12,inner=8,r_outer=1865,r_inner=999,phase=0):
    return [(0.,0.)]+[(round(r*math.cos(2*math.pi*k/n+a),6),round(r*math.sin(2*math.pi*k/n+a),6))
        for n,r,a in [(outer,r_outer,0),(inner,r_inner,phase)] for k in range(n)]

def certified_cells(cells,radius,points):
    delta=np.array(points)[None,:,:]-cells[:,None,:]
    dist=np.linalg.norm(delta,axis=2)
    valid=(dist<1000-radius-1e-5)&(dist>radius)
    angles=np.where(valid,np.arctan2(delta[:,:,1],delta[:,:,0])%(2*math.pi),np.inf)
    errors=np.arcsin(np.minimum(1,radius/np.maximum(dist,1e-10)))
    order=np.argsort(angles,axis=1)
    angles=np.take_along_axis(angles,order,axis=1)
    errors=np.take_along_axis(errors,order,axis=1)
    count=valid.sum(axis=1);result=np.zeros(len(cells),dtype=bool)
    for n in np.unique(count):
        if n<3:continue
        rows=count==n;a=angles[rows,:n];e=errors[rows,:n]
        gaps=np.roll(a,-1,axis=1)-a;gaps[:,-1]+=2*math.pi
        result[rows]=(gaps+e+np.roll(e,-1,axis=1)<math.pi-1e-9).all(axis=1)
    return result

def certify(points,min_side=.1):
    cells=np.array([[0.,0.]]);side=3600.;levels=[]
    while len(cells):
        mask=certified_cells(cells,side/math.sqrt(2),points)
        uncertain=cells[~mask]
        levels.append(dict(side_m=side,cells=len(cells),certified=int(mask.sum()),uncertain=len(uncertain)))
        if not len(uncertain):return dict(covered=True,formal_interval_proof=False,levels=levels)
        if side<=min_side:
            return dict(covered=False,formal_interval_proof=False,levels=levels,
                        unresolved_centers=uncertain[:100].tolist())
        side/=2
        cells=np.concatenate([uncertain+np.array([x,y])*side/2 for x,y in [(-1,-1),(-1,1),(1,-1),(1,1)]])
        nearest=np.maximum(np.abs(cells)-side/2,0)
        cells=cells[(nearest**2).sum(axis=1)<=1800**2+1e-6]

if __name__=='__main__':
    import json
    from pathlib import Path
    report=certify(ring_grid())
    print(json.dumps(report))
    output=Path(__file__).with_name('ring21-r999-certificate.json')
    assert not output.exists()
    output.write_text(json.dumps(dict(points=ring_grid(),**report),indent=2)+'\n')
