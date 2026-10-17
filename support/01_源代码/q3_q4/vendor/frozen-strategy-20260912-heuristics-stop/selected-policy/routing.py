"""Deterministic multi-start open-route improvement; no source truth inputs."""
import math
import random


def persistent_route(nodes,previous,position,improve=False):
    """Retain the route direction; insert new receipt-derived service nodes."""
    pending=list(nodes);tour=[]
    for old in previous:
        options=[n for n in pending if n[0]==old[0] and
                 (n[0]!='source' or n[1]==old[1])]
        if not options:continue
        n=min(options,key=lambda n:math.dist(n[2],old[2]))
        if n[0]=='anchor' and math.dist(n[2],old[2])>850:continue
        pending.remove(n);tour.append(n)
    for n in pending:
        def insertion(j):
            a=position if not j else tour[j-1][2]
            return math.dist(a,n[2])+(math.dist(n[2],tour[j][2])-math.dist(a,tour[j][2]) if j<len(tour) else 0)
        j=min(range(len(tour)+1),key=insertion);tour.insert(j,n)
    if improve:
        # Preserve the far end while improving local ordering.
        for _ in range(5):
            changed=False
            for i in range(len(tour)-2):
                a=position if i==0 else tour[i-1][2];b=tour[i][2]
                for j in range(i+1,len(tour)-1):
                    c=tour[j][2];d=tour[j+1][2]
                    if math.dist(a,c)+math.dist(b,d)<math.dist(a,b)+math.dist(c,d)-1e-6:
                        tour[i:j+1]=reversed(tour[i:j+1]);changed=True;break
            if not changed:break
    return tour

def route(nodes,position,starts=8,scan_weight=0):
    if len(nodes)<2:return nodes
    points=[position]+[n[2] for n in nodes]
    dist=[[math.dist(a,b) for b in points] for a in points]
    source=[False]+[n[0]=='source' for n in nodes]
    def cost(p):
        total=sum(dist[a][b] for a,b in zip([0]+p,p));anchors=0
        for i in p:
            if source[i]:total+=scan_weight*anchors
            else:anchors+=1
        return total
    def improve(p):
        for iteration in range(15):
            gain=0;change=None
            for i in range(len(p)-1):
                a=0 if i==0 else p[i-1];b=p[i]
                anchors=int(not source[b]);sources=int(source[b]);inversions=0
                for j in range(i+1,len(p)):
                    c=p[j];d=p[j+1] if j+1<len(p) else None
                    if source[c]:inversions+=anchors;sources+=1
                    else:anchors+=1
                    delta=dist[a][c]-dist[a][b]
                    if d is not None:delta+=dist[b][d]-dist[c][d]
                    delta+=scan_weight*(anchors*sources-2*inversions)
                    if delta<gain-1e-6:gain=delta;change=(i,j)
            if change is None:break
            i,j=change;p[i:j+1]=reversed(p[i:j+1])
        # Single-node relocation can escape a 2-opt local optimum.
        for iteration in range(8):
            gain=0;change=None
            for i,x in enumerate(p):
                a=0 if i==0 else p[i-1];b=p[i+1] if i+1<len(p) else None
                removed=-dist[a][x]+(dist[a][b]-dist[x][b] if b is not None else 0)
                short=p[:i]+p[i+1:]
                for j in range(len(short)+1):
                    c=0 if j==0 else short[j-1];d=short[j] if j<len(short) else None
                    delta=removed+dist[c][x]+(dist[x][d]-dist[c][d] if d is not None else 0)
                    if scan_weight:
                        crossed=p[i+1:j+1] if j>i else p[j:i]
                        inversion_delta=sum(int(not source[k]) if source[x] else -int(source[k]) for k in crossed)
                        delta+=scan_weight*inversion_delta*(1 if j>i else -1)
                    if delta<gain-1e-6:gain=delta;change=(i,j)
            if change is None:break
            i,j=change;x=p.pop(i);p.insert(j,x)
        return p
    randomizer=random.Random(193)
    best=None;best_cost=math.inf
    for attempt in range(starts):
        pending=list(range(1,len(points)));p=[];last=0
        while pending:
            options=sorted(pending,key=lambda i:dist[last][i])
            k=0 if attempt==0 else randomizer.randrange(min(3,len(options))) if len(p)<4 else 0
            n=options[k];p.append(n);pending.remove(n);last=n
        p=improve(p);value=cost(p)
        if value<best_cost:best=p;best_cost=value
    return [nodes[i-1] for i in best]
