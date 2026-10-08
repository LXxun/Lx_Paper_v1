"""Rebuild six archived-result tables (five main-text tables and Appendix E) from archived JSON using Python >=3.9, standard library only.
No inference, network, torch, raw participant records, or bootstrap execution.
Outputs must be outside the manuscript's tables/data/source directories.
"""
from pathlib import Path
from statistics import mean, stdev
import argparse, hashlib, json, re, sys
ROOT=Path(__file__).resolve().parents[1]

def read(rel):
    return json.loads((ROOT/rel).read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def body_rows(text):
    body=text.split('\\midrule',1)[1].split('\\bottomrule',1)[0]
    return [re.sub(r'\s+','',x) for x in body.split('\\\\') if x.strip()]
def row(*cells): return ' & '.join(str(c) for c in cells)+r' \\'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True,help='New output directory; never overwrites manuscript tables.')
    ap.add_argument('--check',action='store_true',help='Compare all reconstructed displayed rows to checked-in tables.')
    args=ap.parse_args();out=args.out.resolve()
    for folder in ['tables','data','source','reproduction']:
        q=(ROOT/folder).resolve()
        if out==q or q in out.parents: ap.error('Output must not be in '+folder)
    if out.exists() and any(out.iterdir()): ap.error('Use an empty/new output directory.')
    inputs=['data/final_test/DATA_AUDIT.json','data/final_test/FINAL_AUDIT.json','data/final_test/primary_six.json','data/final_test/sensitivity_means.json','data/mult_space_validation/ANALYSIS.json','data/expanded_locked_validation_20261002/summary_expanded.json']
    pin=read('reproduction/INPUT_HASHES.json')
    for name in inputs:
        if digest(ROOT/name)!=pin[name]: raise ValueError('Archived input hash mismatch: '+name)
    tables={};a=read(inputs[0]);tables['dataset']=[]
    for ds in ['MOSI','MOSEI']:
        x=a[ds];tables['dataset'].append(row(ds,*[f"{x['splits'][s]['samples']:,}" for s in ['train','valid','test']],f"{x['explanation_samples']:,} ({x['explanation_videos']})"))
    pred=read(inputs[1])['prediction_statuses'];tables['final_prediction']=[]
    for ds in ['MOSI','MOSEI']:
        for v,label in [('main_only','Main only'),('main_pair','Main + pair'),('concat','Concat')]:
            rr=[x for x in pred if '/'+ds+'/'+v+'/' in x['path']];assert len(rr)==3
            values=[]
            for k in ['MAE','Pearson','Acc2','weighted_F1']:
                scale=100 if k in ['Acc2','weighted_F1'] else 1;digits=2 if scale==100 else 4
                y=[x[k]*scale for x in rr]
                values.append(f'${mean(y):.{digits}f} \\pm {stdev(y):.{digits}f}$')
            tables['final_prediction'].append(row(ds,label,*values))
    primary=read(inputs[2]);tables['final_primary']=[]
    for x in primary:
        sensitivity='reference_vs' in x['id'];scale=100 if sensitivity else 1
        label='Reference minus target (pp)' if sensitivity else ('Deletion matching' if 'delete' in x['id'] else 'Retention matching')
        tables['final_primary'].append(row(x['id'].split('_')[0],label,f"{x['mean']*scale:.4f}",f"[{x['adjusted_ci'][0]*scale:.4f}, {x['adjusted_ci'][1]*scale:.4f}]"))
    summaries=read(inputs[3]);tables['final_space_matrix']=[];tables['final_comparators']=[];matrix=[]
    def utility(ds,op,space,method):
        matches=[x for x in summaries if x['dataset']==ds and x['configuration'][2:]==['zero','intensity',.2,'T',method,op,space]]
        assert len(matches)==6
        assert {(x['configuration'][0],x['configuration'][1]) for x in matches}=={(h,s) for h in ['main_only','main_pair','concat'] for s in ['mask_mean','unk_zero']}
        # Archived retention entries are positive absolute losses; display utility = -loss.
        return mean(x['mean'] for x in matches)*(1 if op=='delete' else -1)
    for ds in ['MOSI','MOSEI']:
        for op in ['delete','retain']:
            values=[utility(ds,op,s,m) for s,m in [('input','group_occlusion_input'),('input','group_occlusion_evidence'),('evidence','group_occlusion_input'),('evidence','group_occlusion_evidence')]]
            locked=next(x['mean'] for x in primary if x['id']==f'{ds}_{op}_matching')
            err=values[3]-values[2]-locked;assert abs(err)<1e-12
            matrix.append(dict(dataset=ds,operation=op,UiOi=values[0],UiOe=values[1],UeOi=values[2],UeOe=values[3],primary_residual=err))
            tables['final_space_matrix'].append(row(ds,op.capitalize(),*[f'{x:.6f}' for x in values]))
            c=utility(ds,op,'evidence','coalition')
            tables['final_comparators'].append(row(ds,op.capitalize(),*[f"{c-utility(ds,op,'evidence',m):.6f}" for m in ['group_occlusion_input','group_occlusion_evidence','ig_evidence']]))
    mul=read(inputs[4])['summary'];tables['mult_space_validation']=[]
    for ds in ['MOSI','MOSEI']:
        for op,label in [('delete','Deletion'),('retain','Signed retention')]:
            rr=[x for x in mul if x['dataset']==ds and x['modality']=='T' and x['space']=='representation' and x['operation']==op and x['primary']];assert len(rr)==1
            x=rr[0];ci=x['ci98_75_bonferroni4']
            tables['mult_space_validation'].append(row(ds,label,f"{x['mean']:.5f}",f'[{ci[0]:.5f}, {ci[1]:.5f}]'))
    cells=read(inputs[5])['cells'];gaps=[]
    for ref in ['zero','encoded']:
        rr=[x for x in cells if x['modality']=='T' and x['reference']==ref and x['target']=='intensity' and x['budget']==.2];assert len(rr)==24
        assert {(x['dataset'],x['variant'],x['substitution'],x['operation']) for x in rr}=={(d,h,s,o) for d in ['MOSI','MOSEI'] for h in ['main_only','main_pair','concat'] for s in ['mask_mean','unk_zero'] for o in ['delete','retain']}
        gaps.append(dict(reference=ref,cells=24,mean_absolute_outside_gap=mean(x['coalition_mean_absolute_gap']['outside_reencoding'] for x in rr),weights='24 cells equal; within-cell absolute gaps average 3 seeds and eligible segments'))
    # Validate manuscript-rounded diagnostic values, not a new statistical test.
    assert [f"{x['mean_absolute_outside_gap']:.5f}" for x in gaps]==['0.28828','0.28678']
    rendered={};checks={}
    for name,rows in tables.items():
        template=(ROOT/f'source/table_templates/{name}.tex').read_text(encoding='utf-8');assert template.count('@@ROWS@@')==1
        text=template.replace('@@ROWS@@','\n'.join(rows));rendered[name]=text
        ok=body_rows(text)==body_rows((ROOT/f'tables/{name}.tex').read_text(encoding='utf-8'));checks[name]=dict(rows=len(rows),matches_manuscript=ok)
        if args.check and not ok: raise ValueError('Displayed rows differ: '+name)
    out.mkdir(parents=True,exist_ok=True)
    for name,text in rendered.items(): (out/f'{name}.tex').write_text(text,encoding='utf-8')
    report=dict(python=sys.version.split()[0],script_sha256=digest(Path(__file__)),input_hashes={n:digest(ROOT/n) for n in inputs},checks=checks,matrix=matrix,hybrid=gaps,ci_handling='Archived intervals copied; no raw-data bootstrap rerun',status='passed' if all(x['matches_manuscript'] for x in checks.values()) else 'mismatch')
    (out/'REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],tables=len(tables),rows=sum(len(v) for v in tables.values()),output=str(out)),indent=2))

if __name__=='__main__': main()
