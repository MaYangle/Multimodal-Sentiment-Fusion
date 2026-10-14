import math
import unittest
import numpy as np
import torch
from jevaffect.native_training import PooledRegressor
from jevaffect.source_robust import SourceRiskModel,source_group_means,reverse_gradient
from jevaffect.source_robust_training import fit_source_groups,make_fit_loader


class SourceRobustOperatorTests(unittest.TestCase):
    def test_group_dro_matches_exp_weights_and_absent_groups(self):
        losses=torch.tensor([.2,.4,.8],dtype=torch.float64,requires_grad=True)
        groups=torch.tensor([0,0,2])
        means,counts=source_group_means(losses,groups,4)
        torch.testing.assert_close(means,torch.tensor([.3,0,.8,0],dtype=torch.float64))
        model=SourceRiskModel({"text":2},4,3,0.)
        with torch.no_grad():
            model.log_source_probability.copy_(torch.tensor([.1,.2,.3,.4],dtype=torch.float64).log())
            model.regressor.head[-1].weight.zero_();model.regressor.head[-1].bias.zero_()
        previous=model.source_probability().numpy()
        expected=previous*np.exp(.01*means.detach().numpy());expected/=expected.sum()
        risk,_=model.training_objective({"text":torch.zeros(3,2)},losses,groups,"group_dro",0)
        np.testing.assert_allclose(model.source_probability().numpy(),expected,rtol=1e-14)
        risk.backward()
        torch.testing.assert_close(losses.grad,torch.tensor([expected[0]/2,expected[0]/2,expected[2]],dtype=torch.float64))
        self.assertAlmostEqual(expected[1]/expected[3],previous[1]/previous[3])

    def test_reversal_preserves_forward_and_flips_feature_gradient(self):
        x=torch.tensor([.2,.7],requires_grad=True)
        y=reverse_gradient(x,.4)
        torch.testing.assert_close(y,x)
        (y*torch.tensor([2.,3.])).sum().backward()
        torch.testing.assert_close(x.grad,torch.tensor([-.8,-1.2]))

    def test_added_domain_head_preserves_point_rng_and_prediction(self):
        torch.manual_seed(77);base=PooledRegressor({"text":4},8,.3);rng=torch.get_rng_state().clone()
        torch.manual_seed(77);model=SourceRiskModel({"text":4},3,8,.3)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in base.state_dict().items():
            self.assertTrue(torch.equal(value,model.regressor.state_dict()[key]))
        x={"text":torch.randn(7,4)}
        torch.manual_seed(9);a=base(x)
        torch.manual_seed(9);b,_=model.point_context(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_uniform_source_sampler_and_exact_draw_state(self):
        ids=["a/0","b/0","b/1","b/2"]
        sources,groups,counts=fit_source_groups(ids)
        self.assertEqual(sources,["a","b"])
        self.assertEqual(counts.tolist(),[1,3])
        ds=torch.utils.data.TensorDataset(torch.arange(4))
        cfg={"batch_size":4}
        g=torch.Generator().manual_seed(8)
        loader=make_fit_loader(ds,ids,cfg,"balanced_erm",g)
        weights=loader.sampler.weights.numpy()
        self.assertAlmostEqual(weights[0],sum(weights[1:]))
        state=g.get_state();a=list(iter(loader.sampler));g.set_state(state);b=list(iter(loader.sampler))
        self.assertEqual(a,b)

    def test_domain_probe_has_no_feature_gradient_from_source_loss(self):
        model=SourceRiskModel({"text":3},2,4,0.)
        x={"text":torch.randn(5,3)};target=torch.linspace(-.8,.8,5);groups=torch.tensor([0,1,0,1,0])
        loss,_=model.training_objective(x,target,groups,"domain_probe",.5)
        loss.backward();grad=[p.grad.clone() for p in model.regressor.parameters()]
        self.assertTrue(any(torch.count_nonzero(p.grad)>0 for p in model.domain_classifier.parameters()))
        model.zero_grad(set_to_none=True)
        model.training_objective(x,target,groups,"balanced_erm",.5)[0].backward()
        for a,p in zip(grad,model.regressor.parameters()):
            torch.testing.assert_close(a,p.grad,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.domain_classifier.parameters()))


if __name__=="__main__":
    unittest.main()
