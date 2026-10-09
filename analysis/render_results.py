"""Check saved evidence and render results; never fits or executes a classifier."""
import base64
import csv
import hashlib
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
FIG=HERE/'figures'; FIG.mkdir(exist_ok=True)
r=json.loads((OUT/'RESULTS.json').read_text())
models={m['name']:m for m in r['models']}
contrasts={(c['reference'],c['alternative']):c for c in r['contrasts']}
cases=json.loads((OUT/'WORKED_CASES.json').read_text())
run=json.loads((HERE/'EXECUTION.json').read_text())
variance=json.loads((OUT/'PCA_CONTEXT.json').read_text())['variance_fraction_unquantized_pca']
pred=np.load(OUT/'predictions.npz',allow_pickle=False)
frozen=json.loads((OUT/'FIT_FROZEN.json').read_text())
checks={}
checks['frozen_model_files_unchanged']=all(hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h for n,h in frozen['files'].items())
checks['five_configurations']=len(frozen['configurations'])==5
checks['unique_sorted_image_ids']=len(np.unique(pred['ids']))==10000 and bool(np.all(np.diff(pred['ids'])>0))
for name,m in models.items():
    p=pred['pred_'+name]; y=pred['labels']
    assert int(np.sum(p==y))==m['correct']
    cm=np.zeros((10,10),np.int64);np.add.at(cm,(y,p),1)
    assert np.array_equal(cm,np.array(m['confusion']))
for (ref,alt),c in contrasts.items():
    a=pred['pred_'+ref]==pred['labels'];b=pred['pred_'+alt]==pred['labels']
    assert int(np.sum(~a & b))==c['corrected']
    assert int(np.sum(a & ~b))==c['broken']
    assert c['both_correct']+c['both_wrong']+c['corrected']+c['broken']==10000
    assert abs(c['delta_pp']-(c['corrected']-c['broken'])/100)<1e-10
checks['accuracies_confusions_and_paired_counts_recomputed']=True
checks['anchor_arrays_bit_exact']=all(x['bit_exact'] for x in json.loads((OUT/'ANCHOR_REPRODUCTION.json').read_text()).values())
checks['arithmetic_checks_pass']=json.loads((OUT/'EVALUATION_CHECKS.json').read_text())['status']=='PASS'
checks['within_budget']=run['elapsed_seconds']<=run['cap_seconds']
checks['containers_cleaned']=all(s['container_absent'] for s in run['stages'])
checks['original_source_unchanged']=hashlib.sha256((HERE.parents[2]/'mnist_search/deliverables/professor_comparison_001/classifier.py').read_bytes()).hexdigest()==r['source_sha256']
checks['selected_case_sums_reconcile']=all(abs(sum(c['terms'].values())-c['margin'])<1e-5 for c in cases['cases'] if c['eligible_n'])
assert all(checks.values()),checks
(HERE/'AUDIT.json').write_text(json.dumps({'status':'PASS','kind':'self-check from saved raw predictions, independent calculation path; not independent reviewer',
                                          'checks':checks},indent=2)+'\n')

names={'q20':'20 coordinates · quadratic','q40':'40 coordinates · quadratic',
       'q60':'60 coordinates · quadratic','l40':'40 coordinates · linear',
       'f40':'40 coordinates · float32 projection/head','c2':'Frozen C2 · int8 projection/head',
       'c2_no_quadratic':'Frozen C2 · quadratic terms removed'}
with (HERE/'comparison.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['configuration','coordinates','head_features','correct','errors','accuracy_pct','raw_parameter_payload_bytes'])
    for key in ['q20','q40','q60','l40','f40','c2','c2_no_quadratic']:
        m=models[key];w.writerow([names[key],m['k'],m['head_features'],m['correct'],m['errors'],m['accuracy_pct'],m['raw_parameter_payload_bytes']])
with (HERE/'paired_effects.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['reference','alternative','delta_percentage_points','corrected','broken','both_correct','both_wrong','prediction_changes','descriptive_95_low','descriptive_95_high'])
    for (a,b),c in contrasts.items():
        w.writerow([a,b,c['delta_pp'],c['corrected'],c['broken'],c['both_correct'],c['both_wrong'],c['prediction_changes'],*c['descriptive_95_interval_pp']])

INK='#172f46';TEAL='#087e82';BLUE='#477aaa';GREY='#647483';RED='#b84a48';GOLD='#b17b24'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':INK,'text.color':INK,'axes.titleweight':'bold','axes.titlesize':13,
                     'savefig.facecolor':'white','figure.facecolor':'white'})
fig,axs=plt.subplots(1,3,figsize=(15,4.8),gridspec_kw={'width_ratios':[1.02,.95,1.5]})
ax=axs[0]
xs=np.array([20,40,60]);ys=[models['q'+str(k)]['accuracy_pct'] for k in xs]
ax.plot(xs,ys,'o-',color=TEAL,lw=2,ms=7)
for x,y in zip(xs,ys):ax.annotate(f'{y:.2f}%',(x,y),xytext=(0,12),textcoords='offset points',ha='center',weight='bold')
ax.set(xlim=(14,66),ylim=(95.3,98.55),xticks=xs,xlabel='PCA coordinates',ylabel='Development accuracy (%)',title='1. Retaining more information')
ax.grid(axis='y',alpha=.17)
ax.text(.03,.07,'Same quadratic recipe\nFloat32 heads',transform=ax.transAxes,color=GREY,fontsize=10)
ax=axs[1]
vals=[models[n]['accuracy_pct'] for n in ['l40','q40']]
ax.bar([0,1],vals,width=.56,color=[BLUE,TEAL])
for x,y in enumerate(vals):ax.text(x,y+2,f'{y:.2f}%',ha='center',weight='bold')
ax.set(ylim=(0,110),xticks=[0,1],xticklabels=['Linear','Quadratic'],ylabel='Development accuracy (%)',title='2. Using the same 40 numbers')
ax.text(.5,.075,'+8.47 percentage points',transform=ax.transAxes,ha='center',color=INK,weight='bold',fontsize=10,
        bbox={'facecolor':'white','edgecolor':'none','alpha':.96,'pad':4})
ax=axs[2]
keys=['f40','q40','c2'];sizes=[models[k]['raw_parameter_payload_bytes']/1024 for k in keys]
ax.barh([2,1,0],sizes,color=[BLUE,'#75a7be',TEAL],height=.55)
for y,key,size in zip([2,1,0],keys,sizes):
    ax.text(size+.8,y,f'{models[key]["raw_parameter_payload_bytes"]:,} B\n{models[key]["accuracy_pct"]:.2f}%',va='center',fontsize=10)
ax.set(yticks=[2,1,0],yticklabels=['Float projection\nFloat head','Int8 projection\nFloat head','Int8 projection\nInt8 head'],xlim=(0,88),xlabel='Raw parameter payload (KiB)',title='3. Encoding the parameters')
ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
fig.suptitle('Where C2’s compactness comes from',fontsize=19,weight='bold',x=.03,ha='left',y=.995)
fig.text(.03,.012,'10,000 adaptively reused development images. Five fixed fits. Payload excludes source/encoding overhead; frozen C2 source is 27,654 B.',fontsize=9.5,color=GREY)
fig.tight_layout(rect=[0,.065,1,.94],w_pad=2.0)
for suffix in ['png','svg']:fig.savefig(FIG/('compactness_results.'+suffix),dpi=180)
plt.close(fig)

imgs=np.load(OUT/'worked_case_images.npz',allow_pickle=False)
valid=[c for c in cases['cases'] if c['eligible_n']]
fig,axs=plt.subplots(len(valid),3,figsize=(13,11),gridspec_kw={'width_ratios':[1,1,3.7]})
titles={'ordinary_success':'Ordinary correct decision','quadratic_corrected':'Quadratic comparison corrects an error',
        'quadratic_harmed':'Quadratic comparison introduces an error','C2_error':'An error made by C2'}
for row,c in enumerate(valid):
    imageidx=c['panel_array_row']
    for col,data in enumerate([imgs['images'][imageidx],imgs['pooled'][imageidx]]):
        ax=axs[row,col];ax.imshow(data,cmap='gray_r',interpolation='nearest');ax.axis('off')
        if row==0:ax.set_title('Input' if col==0 else 'Deskewed / pooled',fontsize=11)
    axs[row,0].text(.5,-.25,f'Label {c["true_digit"]} · C2 {c["C2_digit"]}\nIndex {c["source_index"]}',transform=axs[row,0].transAxes,ha='center',fontsize=10)
    ax=axs[row,2]
    vals=[c['terms'][k] for k in ['bias','linear','squares','cross_products']]
    ax.barh([3,2,1,0],vals,color=[TEAL if v>=0 else RED for v in vals],height=.6)
    ax.axvline(0,color=GREY,lw=.8)
    for y,v in zip([3,2,1,0],vals):ax.text(v+(.014 if v>=0 else -.014),y,f'{v:+.3f}',va='center',ha='left' if v>=0 else 'right',fontsize=10)
    ax.set(yticks=[3,2,1,0],yticklabels=['Bias','Linear','Squares','Cross-products'],xlim=(-.43,.85))
    ax.set_title(titles[c['case']]+f'\nScore({c["C2_digit"]}) − score({c["competitor"]}) = {c["margin"]:+.3f}',loc='left',fontsize=12,pad=10)
    ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
    if row==len(valid)-1:ax.set_xlabel('Contribution to C2’s class-score difference')
fig.suptitle('Following real images into C2’s decision',fontsize=19,weight='bold',x=.03,y=.997,ha='left')
fig.text(.03,.012,'Cases follow the predeclared median-selection rule. Positive terms favour C2’s prediction, including when it is wrong. Scores are not probabilities.',fontsize=9.4,color=GREY)
fig.tight_layout(rect=[0,.035,1,.965],h_pad=3.1,w_pad=2)
for suffix in ['png','svg']:fig.savefig(FIG/('worked_decisions.'+suffix),dpi=165)
plt.close(fig)

def acc(n):return f'{models[n]["accuracy_pct"]:.2f}%'
def ci(a,b):
    lo,hi=contrasts[a,b]['descriptive_95_interval_pp'];return f'[{lo:+.2f}, {hi:+.2f}]'
rows=[]
for n,label in [('q20','20 coordinates, quadratic'),('q40','40 coordinates, quadratic'),('q60','60 coordinates, quadratic'),
                ('l40','40 coordinates, linear'),('f40','40 coordinates, float32 projection and head'),('c2','Frozen C2, int8 projection and head')]:
    m=models[n];rows.append(f'| {label} | {m["head_features"]:,} | {acc(n)} | {m["errors"]:,} | {m["raw_parameter_payload_bytes"]:,} |')
gain=contrasts['l40','q40'];larger=contrasts['q40','q60'];smaller=contrasts['q40','q20'];quant=contrasts['q40','c2']
saving=100*(1-models['c2']['raw_parameter_payload_bytes']/models['f40']['raw_parameter_payload_bytes'])
worked=next(c for c in valid if c['case']=='quadratic_corrected')
linpart=worked['terms']['bias']+worked['terms']['linear']
quadpart=worked['terms']['squares']+worked['terms']['cross_products']
text=f'''# Why the compact classifier works

The clearest result is that much of the useful information fits into forty numbers, but the way the classifier combines those numbers matters. A linear score on that representation reached **{acc('l40')}**. Adding squares and pairwise products reached **{acc('q40')}**, with the input representation held fixed. Keeping sixty coordinates reached **{acc('q60')}**. Encoding the projection and final head as integers gave the frozen C2 result of **{acc('c2')}** while reducing the raw parameter payload by **{saving:.1f}%** relative to the float32 version.

These are results on the same 10,000-image **development fold**, which the earlier search already used. They explain this fitted construction on that cohort; they are exploratory. The previously measured **98.09% final-test accuracy** is unchanged and was not rerun. The five fitted configurations and evaluation completed in **{run['elapsed_seconds']:.2f} seconds** within the 30-minute numerical-run ceiling, using one CPU, a 1 GiB memory limit and no paid model calls. Implementation, figure production and writing are outside that runtime.

## The small experiment

| Configuration | Head features | Development accuracy | Errors / 10,000 | Raw parameter bytes |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

The first four rows use the same integer-encoded PCA construction and float32 heads. This keeps head encoding out of the main dimension and interaction comparisons. The float32 comparison removes projection encoding and refits its head under the same selection policy. The final row is the original frozen C2. Raw payload counts the mean, projection, scaling values, head weights and bias in their declared storage formats; it excludes archive and source encoding overhead. C2's actual distributed source is **27,654 bytes**. Stored int8 coefficients are decoded to float32 for computation.

![The dimension, interaction and encoding comparisons](figures/compactness_results.png)

## What forty numbers preserve

After deskewing and averaging each 2-by-2 pixel block, the image has 196 values. PCA expresses it through shared directions of variation. On the 45,000 rows used to fit PCA, the first twenty directions account for **{100*variance['20']:.2f}%** of variance, forty for **{100*variance['40']:.2f}%**, and sixty for **{100*variance['60']:.2f}%**. These fractions describe the unquantized PCA fit; they are context, not a measure of classification information.

The forty numbers change with each image. Each is a weighted combination of the 196 pooled pixels, and the projection stores 196 × 40 learned coefficients. Those directions were learned without digit labels; labels then fitted the class-score weights. Deskewing and pooling were prescribed by the program. Separating those jobs prevents all of the accuracy from being attributed to what one fitted matrix learned.

The classification comparisons tell us more. Reducing forty coordinates to twenty broke **{smaller['broken']}** correct decisions and corrected **{smaller['corrected']}**, a net loss of **{smaller['broken']-smaller['corrected']} images** or **1.88 percentage points**. The descriptive paired interval is {ci('q40','q20')} points. The lost net accuracy occurs across every digit class, rather than in one isolated class.

Increasing forty coordinates to sixty corrected **{larger['corrected']}** cases and broke **{larger['broken']}**. The net gain was **10 images, or 0.10 points**, with a descriptive interval of {ci('q40','q60')} points. The head grows from 860 to 1,890 features and the raw payload with a float32 head roughly doubles. Forty therefore retains much of what this quadratic construction uses, with a small observed cost against sixty here. This does not establish that forty is optimal or that the two constructions are equivalent. Dimension changes also change the available products and selected regularization.

## Why combining the numbers matters

A linear score gives each coordinate a fixed weight and adds the results. A quadratic score can also make the contribution of one coordinate depend on another. For example, a term z2 × z4 changes sign when just one of the coordinates changes sign; a square responds to a coordinate's magnitude on either side of zero. These relationships let the boundary bend within the same forty-dimensional representation.

The program generates 40 × 41 / 2 = **820 squares and cross-products** from those forty values. Together with the forty linear values, they feed an 860-by-10 weight matrix. These 860 features are computed for the current image; they are not stored digit templates or 860 independently identified human concepts. All ten classes share the same forty-coordinate representation.

The controlled comparison supports the value of that added flexibility. The quadratic head corrected **{gain['corrected']}** linear-head errors and broke **{gain['broken']}** correct decisions: **{gain['corrected']-gain['broken']} additional correct images**, or **8.47 points**, with a descriptive interval of {ci('l40','q40')} points. The net improvement appears in all ten digit classes. The representation was unchanged and each head received the same five-choice selection allowance, so this is evidence for the benefit of the quadratic construction under that fitting policy.

Removing the quadratic terms from C2 while leaving its remaining weights unchanged reduced accuracy to **{acc('c2_no_quadratic')}**. That establishes strong reliance in the existing fitted program. The separately fitted linear head reaches {acc('l40')}, so the removal result must not be presented as what a linear model can learn. It changes a fitted predictor whose linear and quadratic weights were learned together.

## One decision, worked through

The selected corrected case is a labelled **{worked['true_digit']}**, at original image index **{worked['source_index']}**. The refitted linear model calls it **{worked['linear_model_digit']}**; C2 calls it **{worked['C2_digit']}**. Within C2's own {worked['C2_digit']}-versus-{worked['competitor']} comparison, the bias and linear terms sum to **{linpart:+.3f}**, favouring {worked['competitor']}. The squares and cross-products contribute **{quadpart:+.3f}**, reversing that comparison. Their total is **{worked['margin']:+.3f}**, favouring {worked['C2_digit']}.

One of the largest individual contributions is **z2 × z4: +0.282**. This is a measured relationship in the actual decision calculation. It does not, by itself, name either coordinate as a particular stroke detector. The useful answer at this stage is that combinations of learned image directions carry distinctions that the linear rule often misses.

![Four image-to-decision examples, including failures](figures/worked_decisions.png)

The panel includes an ordinary correct decision, a corrected case, a newly harmed case and a C2 error. The corrected and harmed examples are selected by the predetermined median-margin rule. Of the quadratic comparison's 868 corrected cases, 865 also agree with frozen C2; of its 21 harmed cases, 19 agree. The examples come from those eligible groups. They illustrate the full group counts rather than establish those counts. In the harmed example, the refitted linear model gets a 7 right while C2 calls it 3; this does not mean that deleting C2's quadratic terms would fix that image.

## What integer encoding costs

The float32-projection and integer-projection versions with refitted float32 heads produced **identical predicted digits on all 10,000 images**. Encoding the anchor's final head then changed **{quant['prediction_changes']} predictions**: **{quant['corrected']} errors were corrected**, **{quant['broken']} correct answers were broken**, and four wrong predictions changed to other wrong digits. The net difference was **one additional correct image**, with a descriptive interval of {ci('q40','c2')} points.

This supports a small observed decision cost for the chosen encoding. It does not show that integer encoding generally improves accuracy or loses no information. The projection comparison includes refitting; the head comparison perturbs the stored weights of one existing fit. The zero-width empirical bootstrap interval for the projection comparison is a consequence of observing no disagreements, not proof of population equivalence.

The raw payload falls from **66,744 bytes** with float32 projection and head to **43,384 bytes** with an integer projection, then to **17,624 bytes** with both encoded. The deployed file is larger because base64 and program text add overhead. Neither payload size nor source size measures process memory or the cost of finding the model.

## The explanation these results support

The compactness comes from three connected choices. The image is expressed through a shared low-dimensional representation; products make that representation useful for more flexible class boundaries; limited-precision storage preserves almost all of the observed decisions while reducing bytes. The evidence shows a substantial cost below forty coordinates, a modest observed gain above forty, and a large benefit from quadratic relationships at forty.

That combination explains how this classifier can be small and accurate without storing examples or a large separate model for each digit. It leaves open a finer semantic account of individual directions, whether a random projection would work similarly, and how much of the result depends on deskewing. Those questions were outside this fixed experiment. The interpretation is conditional on the same preprocessing, fitting data and ridge-selection grid. The anchor, linear and float32 fits selected the lower boundary of that grid, so these are comparisons within the stated recipe, not global optima.

## How the evidence was checked

C2's mean, projection, whitening values, decoded integer head and biases reproduced **bit for bit**. The identity, compensated coordinate-change and known-perturbation controls passed before comparative evaluation. Across all development images, the score reconstruction had a maximum absolute error of **2.9334 × 10^-6** (rounded upward), within its predeclared arithmetic bounds, and **no changed predicted digit**. Four cases were near ties under the conservative bound; they also retained the same decisions.

All five configurations were frozen before comparative evaluation. The accuracy tables, confusion matrices and corrected/broken counts were independently recalculated from saved predictions by the reporting script; this is a separate calculation path by the same author, not an independent reviewer. Each paired partition sums to 10,000. The intervals use 2,000 class-stratified paired bootstrap replicates with seed 20261009 and describe image-sampling variability conditional on the fitted models. They do not undo development-fold reuse or measure training-seed uncertainty.

The selected C2 and original comparison files are unchanged. No final-test images, QMNIST data or model-service calls were used. Both isolated containers were removed after completion.

## Reproducible evidence

- [Comparison table](comparison.csv) and [paired effects](paired_effects.csv).
- [Full numerical results](results/RESULTS.json), including classwise outcomes and changed class pairs.
- [Saved predictions and scores](results/predictions.npz).
- [Worked decisions](results/WORKED_CASES.json) and [fitted model manifest](results/FIT_FROZEN.json).
- [Anchor reproduction](results/ANCHOR_REPRODUCTION.json), [instrument checks](results/INSTRUMENT_CHECKS.json), [evaluation checks](results/EVALUATION_CHECKS.json) and [report audit](AUDIT.json).
- [Execution receipt](EXECUTION.json), [frozen workflow](WORKFLOW_FROZEN.md), [experiment code](code/experiment.py) and [launcher](run.py).

The launcher refuses to overwrite this completed run. A future reproduction needs a fresh destination with the pinned input hashes and container image; it should preserve this run as evidence.
'''
(HERE/'RESULTS.md').write_text(text)

# A small renderer for the report's own fixed Markdown subset, with figures embedded.
def inline(s):
    s=html.escape(s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
    return s
body=[];lines=text.splitlines();i=0
while i<len(lines):
    line=lines[i]
    if not line.strip():i+=1;continue
    if line.startswith('!['):
        match=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line)
        encoded=base64.b64encode((HERE/match[2]).read_bytes()).decode()
        body.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(match[1])}"></figure>');i+=1
    elif line.startswith('#'):
        n=len(line)-len(line.lstrip('#'));body.append(f'<h{n}>{inline(line[n:].strip())}</h{n}>');i+=1
    elif line.startswith('|'):
        table=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[-:]+',c) for c in cells):table.append(cells)
            i+=1
        body.append('<div class="table-wrap"><table>')
        for rownum,row in enumerate(table):
            tag='th' if rownum==0 else 'td';body.append('<tr>'+''.join(f'<{tag}>{inline(c)}</{tag}>' for c in row)+'</tr>')
        body.append('</table></div>')
    elif line.startswith('- '):
        body.append('<ul>')
        while i<len(lines) and lines[i].startswith('- '):body.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
        body.append('</ul>')
    else:
        para=[]
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','- ')):para.append(lines[i]);i+=1
        body.append('<p>'+inline(' '.join(para))+'</p>')
style='''body{margin:0;background:#edf2f4;color:#172f46;font-family:system-ui,-apple-system,sans-serif;line-height:1.65}main{max-width:1020px;margin:36px auto;background:white;padding:44px 54px;border-top:7px solid #087e82;box-shadow:0 3px 20px #172f4610}h1{font-size:36px;line-height:1.2;letter-spacing:-.6px;margin:0 0 24px}h2{font-size:22px;margin-top:38px;border-bottom:1px solid #dae4e9;padding-bottom:8px}p{max-width:91ch}p:first-of-type{font-size:20px;line-height:1.6;color:#21455e}a{color:#087e82}table{border-collapse:collapse;width:100%;font-size:14px}th{background:#172f46;color:white;text-align:left}td,th{padding:11px 12px;border-bottom:1px solid #dde5e9}tr:nth-child(even){background:#f1f6f7}td:not(:first-child){white-space:nowrap;text-align:right}figure{margin:28px -12px}img{display:block;width:100%;height:auto}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:1.3px;color:#647483;margin-bottom:12px}.table-wrap{overflow:auto}@media(max-width:760px){main{margin:0;padding:24px}h1{font-size:29px}figure{margin:20px -12px}}@media print{body{background:white}main{box-shadow:none;margin:0;padding:0;border:0}h2{break-after:avoid}figure,table{break-inside:avoid}}'''
page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Why C2’s compact representation works</title><style>'+style+'</style><main><div class="eyebrow">C2 dissection · Fixed five-configuration experiment · 9 October 2026</div>'+''.join(body)+'</main></html>'
(HERE/'RESULTS.html').write_text(page)
print(json.dumps({'audit':'PASS','report':str(HERE/'RESULTS.html'),'figures':2,'numerical_model_runs_added':0}))
