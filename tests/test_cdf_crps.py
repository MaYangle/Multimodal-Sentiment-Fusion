import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.integrate import quad
from jevaffect import cdf_crps as q
from jevaffect import shape_controls as s


class CdfCrpsTests(unittest.TestCase):
    def test_uniform_and_atomic_analytic_limits_include_boundary_observations(self):
        m=torch.tensor([-1.,-.4,0.,.7,1.],dtype=torch.float64)
        y=torch.tensor([1.,-.1,0.,-.8,-1.],dtype=torch.float64)
        out=dict(parameters=dict(anchor=m,bound=1.))
        with patch.object(q,'cdf',side_effect=lambda _,x: ((x+1)/2).clamp(0,1)):
            torch.testing.assert_close(q.per_sample(out,y),y.square()/2+1/6,atol=1e-12,rtol=0)
        with patch.object(q,'cdf',side_effect=lambda _,x: (x >= m[:,None]).double()):
            torch.testing.assert_close(q.per_sample(out,y),(m-y).abs(),atol=1e-12,rtol=0)

    def test_all_families_match_independent_adaptive_scalar_integral(self):
        torch.manual_seed(191)
        x=torch.randn(4,257)
        m=torch.tensor([-1.,-.37,.41,1.])
        y=torch.tensor([-.1,.8,-.6,0.])
        for case in ('scale','mixture','median_mdn','bern_d6'):
            model=s.ShapeHead(case,dropout=0).eval()
            out=model(x,m)
            got=q.per_sample(out,y,48).detach().numpy()
            for i in range(len(m)):
                # Independent adaptive quadrature visits only this sample.
                row={'kind':out['kind'], 'parameters':{
                    k:v[i:i+1] if torch.is_tensor(v) and v.ndim and v.shape[0]==len(m) else v
                    for k,v in out['parameters'].items()}}
                def integrand(t):
                    with torch.no_grad():
                        f=float(s.cdf(row,t)[0])
                    return (f-float(t >= float(y[i])))**2
                points=sorted(set([-1.,float(m[i]),float(y[i]),1.]))
                expected=sum(quad(integrand,a,b,epsabs=1e-10,epsrel=1e-10)[0] for a,b in zip(points[:-1],points[1:]))
                self.assertAlmostEqual(got[i],expected,places=8,msg=case)

    def test_gradient_matches_finite_difference_of_final_median_mdn(self):
        torch.manual_seed(71)
        model=s.ShapeHead('median_mdn',feature_dim=3,hidden=4,dropout=0).double().eval()
        x=torch.randn(2,3,dtype=torch.float64)
        m=torch.tensor([-.2,.4],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([-.1,.8],dtype=torch.float64)
        score=q.per_sample(model(x,m),y).mean();score.backward()
        self.assertIsNone(m.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        param=model.head.bias; original=float(param[0].detach()); analytic=param.grad[0].clone(); eps=1e-5
        with torch.no_grad():
            param[0]=original+eps;plus=q.per_sample(model(x,m),y).mean()
            param[0]=original-eps;minus=q.per_sample(model(x,m),y).mean();param[0]=original
        torch.testing.assert_close(analytic,(plus-minus)/(2*eps),atol=1e-8,rtol=1e-5)

    def test_no_rounding_or_support_rejection_and_order_guards(self):
        out=s.ShapeHead('scale',feature_dim=2,dropout=0)(torch.zeros(2,2),torch.tensor([0.,1.]))
        result=q.per_sample(out,torch.tensor([-.1,.123456]),48)
        self.assertTrue(torch.isfinite(result).all())
        for nodes in (0,7,True,1.5):
            with self.assertRaises(ValueError):q.per_sample(out,torch.zeros(2),nodes)
        with self.assertRaises(ValueError):q.per_sample(out,torch.tensor([0.,1.1]))


if __name__=='__main__':unittest.main()
