"""Compare the GPU solver with a scalar reference of the upstream equations."""
import math
from pathlib import Path
import tempfile
from gpu import ROOT, run

P = [[5.,5.,5.,.35],[5.7,5.,5.,.35],[.35,.36,.35,.35],[15.65,11.64,9.65,.35]]
V = [[1.,0.,0.],[-1.,0.,0.],[-2.,-3.,-4.],[2.,3.,4.]]
DT, H = 1/90, 1.7

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def sub(a,b): return [x-y for x,y in zip(a,b)]
def grad(r):
    d = math.sqrt(dot(r,r))
    return [0.,0.,0.] if d <= 1e-6 or d >= H else [(-45/(math.pi*H**6))*(H-d)**2*x/d for x in r]
def reference():
    gradients = [[grad(sub(p[:3],q[:3])) for q in P] for p in P]
    rho = [sum(315/(64*math.pi*H**9)*max(H*H-dot(sub(p[:3],q[:3]),sub(p[:3],q[:3])),0)**3 for q in P) for p in P]
    factors = []
    for gs in gradients:
        total = [sum(g[k] for g in gs) for k in range(3)]
        denom = dot(total,total) + sum(dot(g,g) for g in gs)
        factors.append(1/denom if denom > 1e-6 else 0)
    def project(v,density):
        rates = [sum(dot(sub(v[i],v[j]),gradients[i][j]) for j in range(4)) for i in range(4)]
        c = [max(rho[i]+DT*rates[i]-1 if density else rates[i],0)*factors[i]/(DT*DT if density else DT) for i in range(4)]
        return [[v[i][k]-DT*sum((c[i]+c[j])*gradients[i][j][k] for j in range(4)) for k in range(3)] for i in range(4)]
    v = project(V,False)
    bounds = [16.,12.,10.]
    clamp = lambda x: max(0,min(1,x))
    v = [[v[i][k]*math.exp(-.001*DT)+DT*((-9.8 if k==1 else 0)+40*(clamp(1-(P[i][k]-.35)/.35)**2-clamp(1-(bounds[k]-.35-P[i][k])/.35)**2)) for k in range(3)] for i in range(4)]
    v = project(v,True)
    positions, velocities = [], []
    for i in range(4):
        x = [P[i][k]+DT*v[i][k] for k in range(3)]
        velocities.extend([max(v[i][k],0) if x[k] <= .35 else min(v[i][k],0) if x[k] >= bounds[k]-.35 else v[i][k] for k in range(3)]+[0])
        positions.extend([max(.35,min(bounds[k]-.35,x[k])) for k in range(3)]+[.35])
    return positions, velocities

if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='wyn-fluid-test-') as temp:
        temp = Path(temp)
        for target in ['spirv', 'wgsl']:
            outputs = run(ROOT/'fluid-simulation/test/step.wyn', temp, target)
            assert len(outputs) == 2
            for actual, expected in zip(outputs, reference()):
                assert len(actual)==len(expected)
                for i,(a,e) in enumerate(zip(actual,expected)):
                    assert math.isfinite(a) and math.isclose(a,e,rel_tol=2e-4,abs_tol=2e-4),(target,i,a,e)
            print(f'{target}: GPU step matches reference, including both wall collisions')
