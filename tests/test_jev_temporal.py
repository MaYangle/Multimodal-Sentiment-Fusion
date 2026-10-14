"""Boundary tests for masked, time-based five-class fusion."""
from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.jev_temporal import JevTemporalOrdinal

class TemporalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.batch={
            'text':torch.randn(2,5,4),'audio':torch.randn(2,8,3),'vision':torch.randn(2,4,2),
            'text_mask':torch.tensor([[1,1,0,0,0],[0,0,0,0,0]],dtype=torch.bool),
            'audio_mask':torch.tensor([[1,1,1,0,0,0,0,0],[1,1,0,0,0,0,0,0]],dtype=torch.bool),
            'vision_mask':torch.tensor([[1,1,0,0],[1,0,0,0]],dtype=torch.bool),
            'audio_times':torch.tensor([[0.05,0.3,0.8,0,0,0,0,0],[0.05,0.3,0,0,0,0,0,0]]),
            'vision_times':torch.tensor([[0,0.5,0,0],[0,0,0,0]])}
        self.model=JevTemporalOrdinal(text_dim=4,audio_dim=3,vision_dim=2,hidden=16,windows=4,dropout=0).eval()
    def test_padding_and_missing_modalities(self):
        with torch.no_grad():a=self.model(self.batch)
        self.assertEqual(tuple(a['logits'].shape),(2,5))
        self.assertTrue(torch.isfinite(a['logits']).all().item())
        self.assertTrue(torch.all(a['gate'][~a['valid_windows']]==0).item())
        altered={key:value.clone() for key,value in self.batch.items()}
        for key in ('text','audio','vision'):
            altered[key][~altered[key+'_mask']]=torch.randn_like(altered[key][~altered[key+'_mask']])*100
        with torch.no_grad():b=self.model(altered)
        self.assertTrue(torch.allclose(a['logits'],b['logits'],atol=1e-7))
        empty={key:value.clone() for key,value in self.batch.items()}
        for key in ('text','audio','vision'):empty[key+'_mask'].zero_()
        with torch.no_grad():c=self.model(empty)
        self.assertTrue(torch.allclose(c['logits'],self.model.prior_logits.expand(2,-1),atol=1e-7))
    def test_occlusion_retains_original_time_grid(self):
        with torch.no_grad():
            whole=self.model(self.batch)
            original_duration=whole['time_windows_seconds'][:,-1,1]
            removed={key:value.clone() for key,value in self.batch.items()}
            removed['audio_mask'][0,2]=False
            masked=self.model(removed,duration_override=original_duration)
        self.assertTrue(torch.allclose(whole['time_windows_seconds'],masked['time_windows_seconds']))
        self.assertTrue(torch.isfinite(masked['logits']).all().item())

    def test_no_time_control_and_optional_utility_gradients(self):
        model=JevTemporalOrdinal(text_dim=4,audio_dim=3,vision_dim=2,hidden=16,windows=4,
                                 dropout=0,gate_kind='global_jsd',temporal=False).eval()
        other={key:value.clone() for key,value in self.batch.items()}
        other['audio_times']+=42;other['vision_times']+=42
        with torch.no_grad():a=model(self.batch);b=model(other)
        self.assertTrue(torch.allclose(a['logits'],b['logits'],atol=1e-7))
        target=torch.zeros((2,4,2));target[0,0,0]=0.5
        self.model.train()
        output=self.model(self.batch,utility_targets=target,utility_weight=0.2)
        loss=torch.nn.functional.cross_entropy(output['logits'],torch.tensor([1,4]))+output['utility_loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(self.model.utility_head[-1].weight.grad).all().item())
        self.assertTrue(torch.isfinite(self.model.text_projection[1].weight.grad).all().item())

if __name__=='__main__':unittest.main()
