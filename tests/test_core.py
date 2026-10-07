import sys,unittest,importlib.util
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'experiments/benchmarks/source'))
from tei_kan.faithfulness import evidence_units,select_span
spec=importlib.util.spec_from_file_location('compact_model',R/'experiments/benchmarks/compact_stage/source/model.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
class CoreTests(unittest.TestCase):
 def test_selector(self):
  self.assertEqual(select_span([[0],[1],[2]],np.array([-3.,-2.,-1.]),.2),[2])
  self.assertEqual(select_span([[0],[1],[2]],np.zeros(3),.5),[0,1])
  class Tok:
   def convert_ids_to_tokens(self,ids):return ['[CLS]','play','##ing','[SEP]','##gap']
  self.assertEqual(evidence_units(np.array([False,True,True,False,True]),np.arange(5),Tok()),[[1,2],[4]])
 def test_compact_replay(self):
  torch.manual_seed(42)
  model=m.TEIKAN(m.ModelConfig(input_dims=(3,4,5),bottleneck_dim=4,fusion='structured_mlp',interaction=False,deterministic=True,evidence_pool=False)).eval()
  mask=torch.ones(2,7,3,dtype=torch.bool);mask[1,:,2]=False
  with torch.no_grad():
   out=model(torch.randn(2,7,3),torch.randn(2,7,4),torch.randn(2,7,5),mask)
   replay=model.fuse(out['evidence'].sum(2))['scores']
  self.assertTrue(torch.allclose(replay,out['scores'],atol=1e-6))
  self.assertTrue(torch.allclose(out['reg'],3*torch.tanh(replay[:,3]/3)))
  self.assertEqual(float(out['evidence'][1,2].abs().sum()),0.)
if __name__=='__main__':unittest.main()
