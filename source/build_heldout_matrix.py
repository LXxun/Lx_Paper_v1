"""Figure 3 from hash-pinned held-out summaries. No new inference or bootstrap."""
from pathlib import Path
import json,hashlib
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/final_test/sensitivity_means.json'
pin=json.loads((ROOT/'reproduction/INPUT_HASHES.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==pin['data/final_test/sensitivity_means.json']
rows=json.loads(source.read_text()); primary=json.loads((ROOT/'data/final_test/primary_six.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(8.8,6.8));fig.subplots_adjust(left=.14,right=.99,top=.80,bottom=.17,hspace=.75,wspace=.35)
records=[]
for idx,ax in enumerate(axes.flat):
 ds=['MOSI','MOSEI'][idx//2];op=['delete','retain'][idx%2]
 vals=[]
 for space in ['input','evidence']:
  line=[]
  for method in ['group_occlusion_input','group_occlusion_evidence']:
   rr=[x for x in rows if x['dataset']==ds and x['configuration'][2:]==['zero','intensity',.2,'T',method,op,space]]
   assert len(rr)==6
   assert len({tuple(x['configuration'][:2]) for x in rr})==6
   line.append(mean(x['mean'] for x in rr)*(1 if op=='delete' else -1))
  vals.append(line)
 effect=next(x['mean'] for x in primary if x['id']==f'{ds}_{op}_matching');assert abs(vals[1][1]-vals[1][0]-effect)<1e-12
 records.append({'dataset':ds,'operation':op,'values':vals,'primary_difference':effect})
 ax.set(xlim=(0,2),ylim=(2,0));ax.axis('off')
 ax.set_title(f"({chr(97+idx)}) {ds} | {'Deletion' if op=='delete' else 'Retention'}",loc='left',fontweight='bold',pad=34)
 for j,label in enumerate(['Input ranking','Evidence ranking']):ax.text(j+.5,-.12,label,ha='center',va='bottom',fontsize=9)
 for i,label in enumerate(['Input\nevaluation','Evidence\nevaluation']):
  ax.text(-.08,i+.5,label,ha='right',va='center',fontsize=9)
  for j in range(2):
   ax.add_patch(Rectangle((j+.015,i+.025),.97,.95,facecolor=['#e6f1f7','#fff0e5'][i],edgecolor='#b5bfc5',linewidth=.7))
   if i==j:ax.add_patch(Rectangle((j+.04,i+.05),.92,.9,fill=False,edgecolor=['#0072b2','#b95a15'][i],linewidth=1.8))
   ax.text(j+.5,i+.5,f'{vals[i][j]:.4f}',ha='center',va='center',fontsize=13,fontweight='bold' if i==j else 'normal',color='#18242a')
 ax.text(1,2.17,f'Evidence-row difference: +{effect:.4f}',ha='center',va='top',fontsize=9,color='#884212')
fig.text(.5,.97,'Held-out cross-space evaluation',ha='center',fontsize=15,fontweight='bold')
fig.text(.5,.055,'Higher utility is better within each row. Outlines mark matched spaces.',ha='center',fontsize=9)
fig.text(.5,.023,'Adjusted inference: evidence-row contrasts only (Figure 2). Input-row contrasts are descriptive.',ha='center',fontsize=8.5)
out=ROOT/'figures/redesigned/figure3_heldout_space_matrix'
fig.savefig(str(out)+'.pdf');fig.savefig(str(out)+'.png',dpi=170);plt.close(fig)
(ROOT/'reproduction/FIGURE3_HELDOUT_CHECK.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records,'new_inference':False,'status':'passed'},indent=2))
print('four matrices checked; primary residuals < 1e-12')
