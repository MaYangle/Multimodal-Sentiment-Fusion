import unittest
import numpy as np
from scripts.diagnose_scalar_gate_normalization import fuse,layernorm,singleton_attention_gradient


class ScalarGateNormalizationTests(unittest.TestCase):
    def test_positive_scalar_cancels_at_zero_epsilon_but_relative_branch_gates_survive(self):
        z=np.random.default_rng(12).normal(size=(5,3,17))
        first,second=np.array([.1,.2,.7]),np.array([.7,.2,.1])
        np.testing.assert_allclose(fuse(z,first,"inside_norm",0),fuse(z,second,"inside_norm",0),atol=1e-14,rtol=0)
        self.assertGreater(np.max(np.abs(fuse(z,first,"after_norm",0)-fuse(z,second,"after_norm",0))),.1)

    def test_finite_epsilon_formula_and_low_variance_qualification(self):
        z=np.array([[-1.,0.,1.]])
        eps=1e-5
        variance=2/3
        expected=z*2/np.sqrt(4*variance+eps)
        np.testing.assert_allclose(layernorm(2*z,eps),expected,atol=1e-14)
        ordinary=np.max(np.abs(layernorm(z,eps)-layernorm(2*z,eps)))
        lowvar=np.max(np.abs(layernorm(z*1e-4,eps)-layernorm(z*2e-4,eps)))
        self.assertGreater(lowvar,ordinary*100)

    def test_singleton_qk_zero_gradient_and_multikey_positive_gradient(self):
        single=singleton_attention_gradient(1)
        multi=singleton_attention_gradient(2)
        self.assertEqual(single["Q_gradient_norm"],0)
        self.assertEqual(single["K_gradient_norm"],0)
        self.assertGreater(single["V_gradient_norm"],0)
        self.assertGreater(multi["Q_gradient_norm"],0)
        self.assertGreater(multi["K_gradient_norm"],0)


if __name__=="__main__":unittest.main()
