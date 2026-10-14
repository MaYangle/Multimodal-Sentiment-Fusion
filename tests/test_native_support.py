import unittest
import numpy as np
import torch
from scipy.optimize import minimize
from jevaffect import native_support as s


class NativeSupportTests(unittest.TestCase):
    def test_projection_is_exact_median_and_independent_kl_optimum(self):
        p = torch.tensor([[.25,.22,.16,.1,.08,.06,.04,.025,.025,.025,.015]],dtype=torch.float64)
        m = torch.tensor([.2],dtype=torch.float32,requires_grad=True)
        q = s.project_median(p,m)
        scores = s.SUPPORT.astype(np.float64)
        below = scores < float(m.detach()[0]); above = scores > float(m.detach()[0])
        self.assertLessEqual(float(q[0,below].sum()),.5+1e-12)
        self.assertLessEqual(float(q[0,above].sum()),.5+1e-12)
        base = p[0].numpy()
        objective = lambda x: np.sum(x*np.log(x/base))
        optimum = minimize(objective,np.full(11,1/11),method='SLSQP',bounds=[(1e-10,1)]*11,
                           constraints=[{'type':'eq','fun':lambda x:x.sum()-1},
                                        {'type':'ineq','fun':lambda x:.5-x[below].sum()},
                                        {'type':'ineq','fun':lambda x:.5-x[above].sum()}],
                           options={'ftol':1e-12,'maxiter':500})
        self.assertTrue(optimum.success, optimum.message)
        np.testing.assert_allclose(q[0].numpy(),optimum.x,atol=1e-6,rtol=1e-5)

    def test_neural_cases_preserve_support_gradients_and_native_labels(self):
        torch.manual_seed(71)
        x = torch.randn(5,257)
        m = torch.tensor([-1.,-.31,0.,.41,1.],requires_grad=True)
        target = torch.tensor([-1.,-.2,0.,.4,1.])
        for case in s.NEURAL_CASES:
            model = s.NativeSupportHead(case,dropout=0)
            out = model(x,m); p = out['probability']
            s.validate_probability(p)
            objective = s.nll(p,target); objective.backward()
            self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters()),case)
            self.assertIsNone(m.grad)
            qs = s.quantiles(p,[.025,.5,.975]).numpy()
            self.assertTrue(np.isin(qs,s.SUPPORT.astype(np.float64)).all())
            self.assertTrue(np.all(qs[:,1:]>=qs[:,:-1]))
            if case.endswith('_projected'):
                below = torch.tensor(s.SUPPORT)[None] < m.detach()[:,None]
                above = torch.tensor(s.SUPPORT)[None] > m.detach()[:,None]
                self.assertTrue(torch.all((p*below).sum(-1)<=.5+1e-12))
                self.assertTrue(torch.all((p*above).sum(-1)<=.5+1e-12))
        with self.assertRaisesRegex(ValueError,'no rounding'):
            s.target_indices(torch.tensor([.1]))

    def test_atomic_hit_and_crps_are_score_faithful(self):
        p = torch.zeros(3,11,dtype=torch.float64)
        p[0,5]=1; p[1,0]=1; p[2,10]=1
        target = torch.tensor([0.,-1.,1.])
        torch.testing.assert_close(s.crps(p,target),torch.zeros(3,dtype=torch.float64))
        point = torch.tensor([.1,-1.05,1.2])
        torch.testing.assert_close(s.hit_probability(p,point),torch.tensor([1.,1.,0.],dtype=torch.float64))
        torch.testing.assert_close(s.author_bins(p),torch.eye(5,dtype=torch.float64)[[2,0,4]])
        p2 = torch.zeros(1,11,dtype=torch.float64);p2[0,10]=1
        torch.testing.assert_close(s.crps(p2,torch.tensor([-1.])),torch.tensor([2.],dtype=torch.float64))
        p3 = torch.zeros(1,11,dtype=torch.float64);p3[0,5]=1
        offgrid = torch.tensor([-.1],dtype=torch.float32)
        torch.testing.assert_close(s.crps(p3,offgrid),offgrid.double().abs())
        with self.assertRaisesRegex(ValueError,'no rounding'):
            s.nll(p3,offgrid)

    def test_projection_gradient_matches_finite_difference_and_offgrid_median(self):
        logits = torch.linspace(-1.,1.,11,dtype=torch.float64)[None].requires_grad_()
        anchor = torch.tensor([.13],dtype=torch.float32)
        target = torch.tensor([0.],dtype=torch.float32)
        def objective(z):
            return s.nll(s.project_median(z.softmax(-1),anchor),target)
        objective(logits).backward()
        eps=1e-5
        for k in (0,5,10):
            plus,minus=logits.detach().clone(),logits.detach().clone()
            plus[0,k]+=eps;minus[0,k]-=eps
            fd=(objective(plus)-objective(minus))/(2*eps)
            torch.testing.assert_close(logits.grad[0,k],fd,atol=1e-8,rtol=1e-5)
        q=s.project_median(logits.detach().softmax(-1),anchor)
        lower=torch.tensor(s.SUPPORT)[None]<anchor[:,None]
        torch.testing.assert_close((q*lower).sum(-1),torch.tensor([.5],dtype=torch.float64))
        self.assertNotEqual(float(s.quantiles(q,[.5])[0,0]),float(anchor[0]))


if __name__ == '__main__':
    unittest.main()
