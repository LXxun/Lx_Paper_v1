"""Reconstruct two follow-up tables from hash-pinned summaries; no training or bootstrap."""
from pathlib import Path
import json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1]
HASHES={'RESULTS.json': '4fbf067bc636ea74fe2abf610a60145524f741ea6daf7067e77a98ba589e76dd', 'TRAINING_SUMMARY.json': 'ef76de36c936882157f091f6da44008f31100eb9f4bfc8c23e9f8fad98d66409'}
def build():
 for n,h in HASHES.items():assert hashlib.sha256((ROOT/'data/recurrent_extension'/n).read_bytes()).hexdigest()==h,n
 r=json.loads((ROOT/'data/recurrent_extension/RESULTS.json').read_text());training=json.loads((ROOT/'data/recurrent_extension/TRAINING_SUMMARY.json').read_text())
 a=[r'\begin{table}[!htbp]',r'\centering\small',r'\caption{Recurrent-encoder follow-up: text matching contrasts $U_e(O_e)-U_e(O_i)$. Bounded output, zero evidence reference and 20\% budget. Intervals are 98.75\% Bonferroni-adjusted video-bootstrap percentile intervals for this separate family of four comparisons. MOSI: 686 segments/31 videos; MOSEI: 300 segments/300 videos.}',r'\label{tab:recurrent-primary}',r'\begin{tabular}{llrrr}',r'\toprule',r'Dataset & Operation & Mean & Lower bound & Upper bound \\',r'\midrule']
 for ds,v in r['summary'].items():
  for q in v['primary']:a.append(f"{ds} & {q['operation'].capitalize()} & {q['mean']:.5f} & {q['ci_lower']:.5f} & {q['ci_upper']:.5f} "+r'\\')
 a +=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
 b=[r'\begin{table}[!htbp]',r'\centering\small',r'\caption{All recurrent-encoder validation runs. MAE uses the bounded regression output; Pearson correlation is computed on the same predictions. These are validation metrics, not additional test-prediction results.}',r'\label{tab:recurrent-validation}',r'\begin{tabular}{llrrr}',r'\toprule',r'Dataset & Seed & Epochs run & MAE & Pearson \\',r'\midrule']
 for q in training:b.append(f"{q['dataset']} & {q['seed']} & {q['completion']['epochs']} & {q['MAE']:.5f} & {q['Pearson']:.5f} "+r'\\')
 b +=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
 return {'recurrent_primary.tex':'\n'.join(a)+'\n','recurrent_validation.tex':'\n'.join(b)+'\n'}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--check',action='store_true');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);tables=build()
 for name,text in tables.items():
  (args.out/name).write_text(text,encoding='utf-8')
  if args.check:assert (ROOT/'tables'/name).read_text(encoding='utf-8')==text,name
 (args.out/'CHECK.json').write_text(json.dumps({'files':list(tables),'rows':10,'matched':args.check,'source_hashes':HASHES},indent=2));print('two tables / ten rows verified' if args.check else 'two tables generated')
