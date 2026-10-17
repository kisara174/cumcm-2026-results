"""Continuous Q3 annulus discovery using source-service positions as probes."""
import math

class AngularCover:
    def __init__(self):self.points=[]
    @staticmethod
    def interval(p):
        r=math.hypot(*p)
        if r<1:return None
        bounds=[(s*s+r*r-999.9**2)/(2*s*r) for s in (1000,1800)]
        if max(bounds)>=1:return None
        alpha=math.acos(max(-1,max(bounds)))
        theta=math.atan2(p[1],p[0])%(2*math.pi)
        return theta-alpha,theta+alpha
    def gaps(self,extra=()):
        intervals=[]
        for p in self.points+list(extra):
            bounds=self.interval(p)
            if bounds is None:continue
            a,b=bounds
            for shift in (-2*math.pi,0,2*math.pi):
                lo=max(0,a+shift);hi=min(2*math.pi,b+shift)
                if lo<hi:intervals.append((lo,hi))
        gaps=[];end=0
        for lo,hi in sorted(intervals):
            if lo>end+1e-9:gaps.append((end,lo))
            end=max(end,hi)
        if end<2*math.pi-1e-9:gaps.append((end,2*math.pi))
        if len(gaps)>1 and gaps[0][0]==0 and gaps[-1][1]==2*math.pi:
            first=gaps.pop(0);last=gaps.pop();gaps.append((last[0],first[1]+2*math.pi))
        return gaps
    def gain(self,p):
        return sum(b-a for a,b in self.gaps())-sum(b-a for a,b in self.gaps([p]))
    def add(self,p):self.points.append(p)
    def pending(self,position):
        result=[];r=1250
        alpha=self.interval((r,0))[1]
        for a,b in self.gaps():
            n=math.ceil((b-a)/(2*alpha-1e-5))
            for i in range(n):
                t=a+(i+.5)*(b-a)/n
                result.append((round(r*math.cos(t),6),round(r*math.sin(t),6)))
        return result
