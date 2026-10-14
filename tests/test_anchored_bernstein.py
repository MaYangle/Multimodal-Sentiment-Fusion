import unittest
import numpy as np
import torch
from jevaffect.anchored_bernstein import parameters,cdf,beta_basis_cdf,bin_probabilities,error_band_probability,quantile


class AnchoredBernsteinTests(unittest.TestCase):
    def test_integer_beta_basis_and_uniform_mixture_identity(self):
        t=torch.linspace(0,1,101,dtype=torch.float64)
        for degree in (1,3,6,12):
            basis=beta_basis_cdf(t,degree)
            torch.testing.assert_close(basis.mean(-1),t,rtol=1e-13,atol=1e-13)
            torch.testing.assert_close(basis[:,0],1-(1-t).pow(degree+1),rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(basis[:,-1],t.pow(degree+1),rtol=1e-12,atol=1e-12)

    def test_median_and_censored_probability_law_at_native_bounds(self):
        torch.manual_seed(5)
        m=torch.tensor([-1.,-.7,-.1,0,.2,.7,1.])
        params=parameters(m,torch.randn(7,2),torch.randn(7,2,7))
        grid=torch.linspace(-1.1,1.1,401,dtype=torch.float64)[None].expand(7,-1)
        f=cdf(params,grid)
        self.assertTrue(torch.all(f>=0) and torch.all(f<=1))
        self.assertTrue(torch.all(f[:,1:]-f[:,:-1]>=-1e-14))
        torch.testing.assert_close(cdf(params,m)[:-1],torch.full((6,),.5,dtype=torch.float64),atol=1e-14,rtol=0)
        self.assertEqual(float(cdf(params,m)[-1]),1.)
        torch.testing.assert_close(quantile(params,.5).flatten(),m.double(),rtol=0,atol=0)
        p=bin_probabilities(params,np.array([-.7,-.1,.1,.7],np.float32).astype(float))
        torch.testing.assert_close(p.sum(-1),torch.ones(7,dtype=torch.float64),rtol=0,atol=1e-14)
        self.assertTrue(torch.all(p>=0))

    def test_endpoint_mass_is_pushforward_and_intervals_include_boundaries(self):
        m=torch.tensor([0.])
        params=parameters(m,torch.tensor([[10.,10.]]),torch.zeros(1,2,7))
        self.assertGreater(float(cdf(params,-1.)),.4)
        self.assertLess(float(cdf(params,torch.tensor([1.-1e-8],dtype=torch.float64))),.6)
        q=quantile(params,[.05,.95])
        torch.testing.assert_close(q,torch.tensor([[-1.,1.]],dtype=torch.float64),rtol=0,atol=1e-10)
        endpoint=parameters(torch.tensor([-1.,1.]),torch.zeros(2,2),torch.zeros(2,2,7))
        self.assertTrue(torch.all(error_band_probability(endpoint,.1)>=.5))

    def test_native_unit_scaling_and_independent_probability_gradients(self):
        m=torch.tensor([-.8,0,.6],requires_grad=True)
        raw=torch.zeros(3,2,requires_grad=True);logits=torch.randn(3,2,7,requires_grad=True)
        params=parameters(m,raw,logits)
        p=bin_probabilities(params,[-.7,-.1,.1,.7])
        loss=-p[torch.arange(3),torch.tensor([0,2,4])].log().mean();loss.backward()
        self.assertIsNone(m.grad)
        self.assertTrue(torch.isfinite(raw.grad).all() and torch.isfinite(logits.grad).all())
        self.assertGreater(int(torch.count_nonzero(raw.grad)),0)
        self.assertGreater(int(torch.count_nonzero(logits.grad)),0)
        widths=params["widths"].detach()*3
        # Compare the same distribution in native[-3,3] units, not shared labels.
        scaled={**params,"anchor":params["anchor"]*3,"widths":widths,"bound":3.}
        torch.testing.assert_close(cdf(params,m.double()),cdf(scaled,m.double()*3),rtol=0,atol=1e-13)
        torch.testing.assert_close(error_band_probability(params,.1),error_band_probability(scaled,.3),rtol=1e-12,atol=1e-12)

    def test_tied_side_shapes_are_reflections_about_anchor(self):
        weights=torch.tensor([[2.,-1.,.4,1.,.1,-.2,.6]])
        params=parameters(torch.tensor([0.]),torch.zeros(1,2),weights[:,None].expand(-1,2,-1))
        x=torch.linspace(-.9,.9,31,dtype=torch.float64)[None]
        torch.testing.assert_close(cdf(params,-x),1-cdf(params,x),rtol=1e-12,atol=1e-12)


if __name__=="__main__":unittest.main()
