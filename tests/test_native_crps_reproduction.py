import copy
import unittest
import torch
from scripts.compare_native_crps_reproduction import exact_state,identity_without_output_location


class NativeCrpsReproductionTests(unittest.TestCase):
    def test_only_output_location_and_config_file_hash_are_ignored(self):
        identity={'config_sha256':'a','config':{'run_root':'one','config_sha256':'a','max_epochs':100},
                  'implementation_sha256':{'code':'pin'},'plan_sha256':'plan','normalization':{'fit_indices':[0,1]}}
        relocated=copy.deepcopy(identity);relocated['config_sha256']='b'
        relocated['config'].update(run_root='two',config_sha256='b')
        self.assertEqual(identity_without_output_location(identity),identity_without_output_location(relocated))
        relocated['config']['max_epochs']=50
        self.assertNotEqual(identity_without_output_location(identity),identity_without_output_location(relocated))
        self.assertEqual(identity['config']['run_root'],'one')

    def test_optimizer_comparison_preserves_exact_values_and_dtype(self):
        value={'state':{0:{'exp_avg':torch.tensor([.2,.4]),'step':torch.tensor(12.)}},'param_groups':[{'lr':.001}]}
        cloned=copy.deepcopy(value)
        self.assertTrue(exact_state(value,cloned))
        cloned['state'][0]['exp_avg'][0]+=.0001
        self.assertFalse(exact_state(value,cloned))
        cloned=copy.deepcopy(value);cloned['state'][0]['exp_avg']=cloned['state'][0]['exp_avg'].double()
        self.assertFalse(exact_state(value,cloned))


if __name__=='__main__':unittest.main()
