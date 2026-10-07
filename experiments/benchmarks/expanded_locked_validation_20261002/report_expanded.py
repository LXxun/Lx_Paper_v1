import json
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;s=json.loads((S/'summary_expanded.json').read_text());cells=s['cells'];parse=json.loads((S/'analysis/parse_summary.json').read_text());statuses=[z for arr in parse['statuses'].values() for z in arr]
primary=[x for x in cells if x['modality']=='T' and x['target']=='intensity' and x['reference']=='zero']
lines=['# 扩大验证结果（2026-10-02）','',f"18 个模型均完成：529 条验证片段，{s['attribution_games']} 个归因游戏，{s['records']} 条干预记录。覆盖 10%、20%、30% 预算、两种输出目标、两种表示参考、两种输入替换。干预记录不是独立样本。未重训，未使用测试集。",'', 'MOSI 为全部 229 条验证片段、10 个视频；MOSEI 为300个验证视频各一条固定哈希样本，其中291条有可用视觉观测。9条视觉不可用样本仍保留，局部视觉评价分母单列。扩大样本与此前诊断样本有重叠，不能称为独立确认性测试。','', '## 原定文本稳定性门槛','', '保留有界强度、零证据参考、原始输入评价的既定门槛：相对随机区间的平均配对优势为正，且至少2/3种子均值为正。每种结构24个配置=2数据集×2输入替换×3预算×删除/保留。','', '| 结构 | MOSI通过数 | MOSEI通过数 | 总计 |','|---|---|---|---|']
for v in ['main_only','main_pair','concat']:
 vals=[]
 for ds in ['MOSI','MOSEI']:
  xx=[x for x in primary if x['variant']==v and x['dataset']==ds];vals.append((sum(x['stability_gate_input'] for x in xx),len(xx)))
 lines.append(f'| {v} | {vals[0][0]}/{vals[0][1]} | {vals[1][0]}/{vals[1][1]} | {sum(x[0] for x in vals)}/{sum(x[1] for x in vals)} |')
fails=[{k:x[k] for k in ['dataset','variant','substitution','budget','operation']}|dict(comparison=x['comparisons']['random']['input']) for x in primary if not x['stability_gate_input']]
if fails:lines+=['','未通过的配置（完整保留）：',json.dumps(fails,ensure_ascii=False,indent=2)]
lines+=['','## 匹配空间后的对照','', '下表固定主要有界输出、零参考，按24个配置等权平均。正差表示原方法更好；配置含重复条件，不是24个独立实验。逐配置视频聚类区间和逐种子值见 summary_expanded.json。','', '| 预算 | 文本：原方法-IG（证据） | 文本：原方法-输入遮挡（输入） | 文本：原方法-证据遮挡（证据） | 文本：原方法-输入遮挡（证据） |','|---|---|---|---|---|']
for bu in [.1,.2,.3]:
 xx=[x for x in primary if x['budget']==bu];vv=[]
 for method,space in [('ig_evidence','evidence'),('group_occlusion_input','input'),('group_occlusion_evidence','evidence'),('group_occlusion_input','evidence')]:vv.append(np.mean([x['comparisons'][method][space]['mean'] for x in xx]))
 lines.append('| '+str(bu)+' | '+' | '.join(f'{v:.8g}' for v in vv)+' |')
lines+=['','## 选区对目标和参考的敏感性','', '以下是原方法文本选区的同区间比例，按模型—片段配对数量加权。参考/替换条件重复不会增加独立样本量。','', '| 预算 | 更换输出目标 | 更换表示参考 | 更换输入替换 |','|---|---|---|---|']
for bu in [.1,.2,.3]:
 vv=[]
 for change in ['target_change','reference_change','substitution_change']:
  xx=[x for x in s['selection_overlap'] if x['modality']=='T' and x['method']=='coalition' and x['budget']==bu and x['change']==change];vv.append(sum(x['same_span_rate']*x['model_segment_pairs'] for x in xx)/sum(x['model_segment_pairs'] for x in xx))
 lines.append('| '+str(bu)+' | '+' | '.join(f'{v:.2%}' for v in vv)+' |')
lines+=['','## 输出变换与重新编码','']
for d in s['game_diagnostics']:
 if d['variant']=='main_only':lines.append(f"- main_only / {d['target']}：二阶项绝对值之和平均 {d['mean_pair_l1']:.8g}；三阶项最大绝对值 {d['max_triple_abs']:.8g}；共同积分网格下原方法与IG最大位置差 {d['max_common_grid_coalition_ig']:.8g}。")
for ref in ['zero','encoded']:
 xx=[x for x in cells if x['modality']=='T' and x['reference']==ref and x['target']=='intensity' and x['budget']==.2];gap=np.mean([x['coalition_mean_absolute_gap']['outside_reencoding'] for x in xx]);lines.append(f'- 20%预算、有界强度、{ref}参考：文本未改变位置的重新编码差平均绝对值 {gap:.8g}。各条件选区可不同，绝对值分量不能当作可相加的因果比例。')
lines+=['','## 数值与统计检查','',f"全部预期样本/方法/预算/重复记录覆盖通过。最大批量重放误差 {max(x['max_batch_error'] for x in statuses):.8g}；最大旧归因重放误差 {max(x['max_prior_attribution_replay_error'] for x in statuses):.8g}。18 个运行均满足预设积分与加和检查；版本和样本哈希与锁定记录一致。",'随机重复先在片段/种子内平均，再平均三个固定种子；按整视频bootstrap20000次。95%区间为逐配置区间，不是跨配置同时区间，也不包含重新训练种子的总体不确定性。MOSI仍只有10个视频簇。','', '本轮结果只能支持其实际呈现的结论。解释组件的一致性性质、优于随机选择、优于其他归因方法和算法原创性是不同命题；不能互相替代。无论门槛是否通过，均不得删除负结果或事后更换主要目标。']
(S/'RESULTS_中文.md').write_text('\n'.join(lines)+'\n');(S/'completion.json').write_text(json.dumps(dict(status='completed_and_analyzed',models=18,games=s['attribution_games'],records=s['records'],primary_text_gate_passed=sum(x['stability_gate_input'] for x in primary),primary_text_gate_total=len(primary),test_evaluated=False),indent=2));print('\n'.join(lines))
