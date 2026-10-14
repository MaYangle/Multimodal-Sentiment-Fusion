import unittest
import torch
from jevaffect.sequence_gates import CASES,SequenceGateRegressor,UnimodalSequenceHeads


class SequenceGatesTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        self.dims={"text":7,"audio":5,"vision":3}
        self.x={m:torch.randn(3,6,d) for m,d in self.dims.items()}
        self.x["lengths"]={"text":torch.tensor([6,3,0]),"audio":torch.tensor([4,0,0]),"vision":torch.tensor([5,2,0])}

    def model(self,case):
        return SequenceGateRegressor(self.dims,case,dim=16,queries=3,depth=1,heads=4,ff_dim=32,dropout=0)

    def test_padding_missing_and_target_free_inference(self):
        for case in CASES:
            model=self.model(case).eval()
            expected=model(self.x)["prediction"]
            altered={m:v.clone() for m,v in self.x.items() if m!="lengths"}
            altered["lengths"]=self.x["lengths"]
            for m in self.dims:
                for i,n in enumerate(self.x["lengths"][m]):
                    altered[m][i,int(n):]=float('nan')
            torch.testing.assert_close(model(altered)["prediction"],expected)
            self.assertTrue(torch.isfinite(expected).all())
            with self.assertRaisesRegex(ValueError,"target"):
                model(self.x,torch.zeros(3))

    def test_matching_parameters_and_nonzero_qk_gradients(self):
        counts=[]
        for case in CASES:
            model=self.model(case)
            model.freeze_knowledge()
            out=model(self.x,torch.tensor([.3,-.2,0.]))
            (out["prediction"].square().mean()+.01*out["contrastive"]+.1*out["distillation"]).backward()
            grad=model.blocks[0].cross['text'].attention.in_proj_weight.grad
            self.assertGreater(grad[:16].norm().item(),0)
            self.assertGreater(grad[16:32].norm().item(),0)
            for branch in model.branches.values():
                self.assertTrue(all(p.grad is None for p in branch.adapter.parameters()))
                self.assertTrue(all(p.grad is None for p in branch.decoder.parameters()))
            counts.append(sum(p.numel() for p in model.parameters()))
            self.assertTrue(torch.allclose(out["ratios"][:2].sum(1),torch.ones(2)))
            self.assertEqual(out["ratios"][2].sum().item(),0)
        self.assertEqual(len(set(counts)),1)

    def test_unimodal_checkpoint_and_target_detach(self):
        teacher=UnimodalSequenceHeads(self.dims,dim=16)
        model=self.model("post_norm")
        model.load_unimodal_state(teacher.state_dict())
        model.freeze_knowledge()
        target=torch.tensor([.1,-.7,0.],requires_grad=True)
        model(self.x,target)["prediction"].sum().backward()
        self.assertIsNone(target.grad)
        for k,v in teacher.state_dict().items():
            torch.testing.assert_close(model.state_dict()[k],v)

    def test_ratio_placement_probe_at_zero_epsilon(self):
        pre=self.model("pre_norm").eval()
        post=self.model("post_norm").eval()
        post.load_state_dict(pre.state_dict())
        for model in (pre,post):
            for layer in model.blocks[0].branch_norm.values():
                layer.eps=0
        # Exclude all-missing rows: eps=0 zero-variance dummy norm is undefined.
        x={m:v[:2] for m,v in self.x.items() if m!="lengths"}
        x["lengths"]={m:torch.full((2,),6) for m in self.dims}
        a=torch.tensor([[.1,.2,.7]]).repeat(2,1)
        b=torch.tensor([[.7,.2,.1]]).repeat(2,1)
        torch.testing.assert_close(pre.probe(x,a)["prediction"],pre.probe(x,b)["prediction"],atol=1e-6,rtol=0)
        self.assertGreater((post.probe(x,a)["prediction"]-post.probe(x,b)["prediction"]).abs().max().item(),1e-4)

    def test_explicit_nonprefix_masks_preserve_native_valid_rows(self):
        model=self.model("static").eval()
        x={m:v.clone() for m,v in self.x.items() if m!="lengths"}
        x["masks"]={m:torch.tensor([[False,True,True,False,True,False]]).repeat(3,1) for m in self.dims}
        expected=model(x)["prediction"]
        for m in self.dims:
            x[m][~x["masks"][m]]=float('nan')
        torch.testing.assert_close(model(x)["prediction"],expected)
        x["lengths"]={m:torch.full((3,),2) for m in self.dims}
        with self.assertRaisesRegex(ValueError,"disagree"):
            model(x)

    def test_distilled_policy_is_target_free_and_same_supervision_control(self):
        policy=self.model("distilled_post_norm")
        control=self.model("distilled_static")
        control.load_state_dict(policy.state_dict())
        target=torch.tensor([.5,-.8,0.],requires_grad=True)
        policy_out=policy(self.x,target)
        control_out=control(self.x,target)
        torch.testing.assert_close(policy_out["distillation"],control_out["distillation"])
        self.assertTrue(torch.isfinite(policy_out["distillation"]))
        (policy_out["prediction"].square().sum()+.1*policy_out["distillation"]).backward()
        self.assertIsNone(target.grad)
        self.assertGreater(policy.policy[0].weight.grad.norm().item(),0)
        policy.eval()
        eval_out=policy(self.x)
        torch.testing.assert_close(eval_out["ratios"],policy_out["policy_ratios"])
        self.assertFalse(torch.allclose(eval_out["ratios"][0],torch.ones(3)))
        self.assertTrue(torch.isfinite(policy.probe_source_none(self.x)["prediction"]).all())


if __name__=='__main__':unittest.main()
