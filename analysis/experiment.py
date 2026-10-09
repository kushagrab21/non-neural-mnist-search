"""Fixed five-configuration C2 study. Fit and evaluation are separate invocations."""
import ast
import hashlib
import importlib.util
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path('/out')
SRC = Path('/src')
CONFIGS = [('q40', 40, True, True), ('q20', 20, True, True),
           ('q60', 60, True, True), ('l40', 40, False, True),
           ('f40', 40, True, False)]
MULTS = [0.1, 1., 10., 100., 1000.]
SOURCE_SHA = '4a718c35621e3c76a170a6447a765e87f39b4c53438e76ddb1a3134b23418a8d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, value):
    p = OUT / name
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def emit(event, **kw):
    print(json.dumps({'event': event, **kw}), flush=True)


def original():
    assert sha(SRC / 'classifier.py') == SOURCE_SHA
    spec = importlib.util.spec_from_file_location('frozen_c2', SRC / 'classifier.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def original_helpers():
    tree = ast.parse((SRC / 'original_build.py').read_text())
    keep = {'pca', 'quantize', 'ridge', 'featmap', 'gram'}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    assert {n.name for n in body} == keep
    ns = {'np': np, 'STAT_CHUNK': 5000, 'FEAT_CHUNK': 2500}
    exec(compile(ast.Module(body=body, type_ignores=[]), 'original_helpers', 'exec'), ns)
    return ns


def project(p, m):
    return ((p - m['mu']) @ m['P']) * m['S']


def features(z, quadratic):
    z = np.asarray(z, dtype=np.float32)
    k = z.shape[-1]
    if not quadratic:
        return z.copy()
    i, j = np.triu_indices(k)
    f = np.empty(z.shape[:-1] + (k + len(i),), dtype=np.float32)
    f[..., :k] = z
    np.multiply(z[..., i], z[..., j], out=f[..., k:])
    return f


def symmetric_heads(W, k):
    i, j = np.triu_indices(k)
    Q = np.zeros((10, k, k), dtype=np.float64)
    for c in range(10):
        Q[c, i, j] = W[k:, c]
        Q[c, j, i] = W[k:, c]
        Q[c] *= .5
        Q[c, np.arange(k), np.arange(k)] *= 2.
    return Q


def algebra_scores(z, W, B):
    z = z.astype(np.float64)
    k = len(z)
    i, j = np.triu_indices(k)
    terms = np.concatenate([z, z[i] * z[j]])[:, None] * W.astype(np.float64)
    return B.astype(np.float64) + terms.sum(axis=0), terms


def score_bounds(terms, B):
    # Conservative forward-error bound for product formation plus a naive dot
    # and bias addition, conditional on the same float32 coordinates z.
    n = 2 * len(terms) + 8
    eps = np.finfo(np.float32).eps
    gamma = n * eps / (1 - n * eps)
    return gamma * np.maximum(1., np.abs(terms).sum(axis=0) + np.abs(B))


def controls(c2, imgs):
    W, B = c2._W, c2._B
    Q = symmetric_heads(W, 40)
    maxima = dict(score_residual=0., bound_ratio=0., sign_residual=0., perturbation_residual=0.)
    disagreements = 0
    for image in imgs[:200]:
        pred = c2.predict(image)
        z = c2._Z.copy()
        raw = c2._F @ W + B
        f = features(z, True)
        assert np.array_equal(f, c2._F)
        assert np.array_equal(f @ W + B, raw)
        calc, terms = algebra_scores(z, W, B)
        err = np.abs(calc - raw)
        bound = score_bounds(terms, B)
        assert np.all(err <= bound)
        disagreements += int(pred != np.argmax(calc))
        maxima['score_residual'] = max(maxima['score_residual'], float(err.max()))
        maxima['bound_ratio'] = max(maxima['bound_ratio'], float((err / bound).max()))
        zd = z.astype(np.float64)
        by_Q = B.astype(np.float64) + zd @ W[:40].astype(np.float64) + np.einsum('i,cij,j->c', zd, Q, zd)
        scale = max(1., float(np.abs(terms).sum()))
        bound64 = 4096 * np.finfo(np.float64).eps * scale
        assert np.max(np.abs(by_Q - calc)) <= bound64
        sign = np.ones(40); sign[0] = -1
        zp = zd * sign
        ap = W[:40].astype(np.float64) * sign[:, None]
        qp = Q * sign[None, :, None] * sign[None, None, :]
        same = B.astype(np.float64) + zp @ ap + np.einsum('i,cij,j->c', zp, qp, zp)
        er = float(np.max(np.abs(same - by_Q)))
        maxima['sign_residual'] = max(maxima['sign_residual'], er)
        assert er <= bound64
        delta = .125
        moved = zd.copy(); moved[0] += delta
        changed = B.astype(np.float64) + moved @ W[:40].astype(np.float64) + np.einsum('i,cij,j->c', moved, Q, moved)
        known = delta * W[0].astype(np.float64) + 2 * delta * (Q[:, 0, :] @ zd) + delta**2 * Q[:, 0, 0]
        er = float(np.max(np.abs((changed - by_Q) - known)))
        maxima['perturbation_residual'] = max(maxima['perturbation_residual'], er)
        assert er <= bound64
    assert disagreements == 0
    result = {'status': 'PASS', 'n_fit_images': 200, 'identity_features_bit_exact': True,
              'decision_disagreements': disagreements, 'maxima': maxima,
              'float32_bound': 'gamma(2*features+8) * max(1,sum(abs(real terms))+abs(bias)); eps=float32.eps',
              'float64_control_bound': '4096*float64.eps*max(1,sum(abs(terms)))',
              'controls': ['identity', 'symmetric quadratic reconstruction', 'compensated coordinate sign change', 'known +0.125 coordinate perturbation'],
              'frozen_before_development_evaluation': True}
    dump('INSTRUMENT_CHECKS.json', result)
    emit('instrument_checks', **result)


def fit():
    start = time.monotonic()
    c2 = original()
    imgs = np.load('/fit/fit_images.npy', mmap_mode='r', allow_pickle=False)
    labels = np.asarray(np.load('/fit/fit_labels.npy', allow_pickle=False)).reshape(-1).astype(np.int64)
    assert imgs.shape == (50000, 28, 28) and labels.shape == (50000,)
    perm = np.random.default_rng(0).permutation(len(imgs))
    hold_idx, tr_idx = np.sort(perm[:5000]), np.sort(perm[5000:])
    helpers = original_helpers()
    helpers.update(n_tr=len(tr_idx), tr_idx=tr_idx, labels=labels)
    cache = OUT / 'fit_preprocessing.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        X, mean64, ev, vec = [z[k] for k in ['X', 'mean64', 'ev', 'vec']]
    else:
        X = np.empty((len(imgs), 196), dtype=np.float32)
        for n, im in enumerate(imgs):
            X[n] = c2._x(im)
        mean64, ev, vec = helpers['pca'](X)
        np.savez(cache, X=X, mean64=mean64, ev=ev, vec=vec)
    st = mean64, ev, vec
    variance = {str(k): float(ev[:k].sum() / ev.sum()) for k in [20, 40, 60]}
    dump('PCA_CONTEXT.json', {'fit_rows': 45000, 'variance_fraction_unquantized_pca': variance,
                              'classification_sufficiency_claim': False})
    emit('preprocessing_complete', seconds=time.monotonic()-start, variance=variance)
    for name, k, quad, pquant in CONFIGS:
        path = OUT / (name + '.npz')
        meta_path = OUT / (name + '.json')
        if path.exists() and meta_path.exists():
            emit('reuse_completed_fit', configuration=name)
            continue
        t0 = time.monotonic()
        # The quantized quadratic branch is the original feature-map code.
        if quad and pquant:
            fm = helpers['featmap'](X, st, k)
            p_store, p_scale = fm['Pq'], fm['Ps']
            Pd, mu, S = p_store.astype(np.float32) * p_scale, fm['mu'], fm['ws']
        else:
            mu = mean64.astype(np.float32)
            S = (1. / np.sqrt(ev[:k])).astype(np.float32)
            if pquant:
                p_store, p_scale = helpers['quantize'](vec[:, :k], 8)
                Pd = p_store.astype(np.float32) * p_scale
            else:
                p_store = vec[:, :k].astype(np.float32)
                p_scale = np.empty(0, dtype=np.float32)
                Pd = p_store
            d = 1 + k + (k * (k + 1) // 2 if quad else 0)
            def feats(idx):
                z = ((X[idx] - mu) @ Pd) * S
                f = np.empty((len(idx), d), dtype=np.float32)
                f[:, 0] = 1.
                f[:, 1:] = features(z, quad)
                return f
            fm = {'d': d, 'feats': feats}
        d = fm['d']
        fh = np.ascontiguousarray(fm['feats'](hold_idx)[:, 1:])
        g_tr, b_tr = helpers['gram'](fm, tr_idx)
        base = float(np.trace(g_tr)) / d / 1000.
        table, best = [], None
        for mult in MULTS:
            w = helpers['ridge'](g_tr, b_tr, mult * base)
            scores = fh @ w[1:].astype(np.float32) + w[0].astype(np.float32)
            acc = float(np.mean(np.argmax(scores, axis=1) == labels[hold_idx]))
            table.append({'mult': mult, 'lambda': mult*base, 'selection_accuracy': acc})
            if best is None or acc > best[1]:
                best = (mult, acc)
        mult, selection = best
        g_h, b_h = helpers['gram'](fm, hold_idx)
        g_all, b_all = g_tr + g_h, b_tr + b_h
        lam = mult * float(np.trace(g_all)) / d / 1000.
        w = helpers['ridge'](g_all, b_all, lam)
        W, B = w[1:].astype(np.float32), w[0].astype(np.float32)
        # Match C2: quantization receives original float64 fitted weights.
        Wq, Wscale = helpers['quantize'](w[1:], 8)
        Wdecoded = Wq.astype(np.float32) * Wscale
        meta = {'name': name, 'k': k, 'quadratic': quad, 'projection_quantized': pquant,
                'head_features': d-1, 'selection_table': table, 'selected_mult': mult,
                'grid_boundary_winner': mult in (MULTS[0], MULTS[-1]),
                'selection_accuracy': selection, 'final_lambda': lam,
                'PCA_fit_rows': 45000, 'selection_rows': 5000, 'final_head_rows': 50000,
                'primary_head': 'float32', 'fit_seconds': time.monotonic()-t0,
                'payload_bytes_float_head': int(mu.nbytes + S.nbytes + p_store.nbytes + p_scale.nbytes + W.nbytes + B.nbytes),
                'payload_bytes_int8_head': int(mu.nbytes + S.nbytes + p_store.nbytes + p_scale.nbytes + Wq.nbytes + Wscale.nbytes + B.nbytes)}
        if name == 'q40':
            comparisons = {key: {'bit_exact': bool(np.array_equal(a, b)), 'max_abs_difference': float(np.max(np.abs(a.astype(np.float64)-b)))}
                           for key, a, b in [('mean', mu, c2._MU), ('projection', Pd, c2._P), ('whitening', S, c2._S),
                                              ('encoded_head_decoded', Wdecoded, c2._W), ('bias', B, c2._B)]}
            dump('ANCHOR_REPRODUCTION.json', comparisons)
            assert all(x['bit_exact'] for x in comparisons.values()), 'Anchor mismatch; halt before further fits/evaluation'
            emit('anchor_bit_exact', arrays=list(comparisons))
            controls(c2, imgs)
        np.savez(path, mu=mu, S=S, Pstored=p_store, Pscale=p_scale,
                 W=W, B=B, Wq=Wq, Wscale=Wscale)
        meta['array_file_sha256'] = sha(path)
        dump(name + '.json', meta)
        emit('fit_complete', **meta)
        del g_tr, b_tr, g_h, b_h, g_all, b_all, w, fh
    frozen_names = [n+s for n, *_ in CONFIGS for s in ['.npz', '.json']]
    frozen_names += ['ANCHOR_REPRODUCTION.json', 'INSTRUMENT_CHECKS.json', 'PCA_CONTEXT.json']
    dump('FIT_FROZEN.json', {'files': {n: sha(OUT/n) for n in frozen_names},
                           'configurations': [n for n, *_ in CONFIGS], 'solves': 30,
                           'source_sha256': SOURCE_SHA, 'numpy': np.__version__, 'python': platform.python_version(),
                           'fit_stage_seconds': time.monotonic()-start,
                           'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           'development_data_mounted': False})
    emit('all_fits_frozen', seconds=time.monotonic()-start)


def load_model(name):
    z = dict(np.load(OUT/(name+'.npz'), allow_pickle=False))
    meta = json.loads((OUT/(name+'.json')).read_text())
    z['P'] = z['Pstored'].astype(np.float32) * z['Pscale'] if meta['projection_quantized'] else z['Pstored']
    z['meta'] = meta
    return z


def paired(a, b, truth):
    # a is reference; b is the alternative.
    ca, cb = a == truth, b == truth
    return {'n': len(truth), 'both_correct': int(np.sum(ca & cb)),
            'both_wrong': int(np.sum(~ca & ~cb)), 'corrected': int(np.sum(~ca & cb)),
            'broken': int(np.sum(ca & ~cb)), 'prediction_changes': int(np.sum(a != b)),
            'delta_pp': float(100 * (np.mean(cb) - np.mean(ca)))}


def evaluate():
    start = time.monotonic()
    frozen = json.loads((OUT/'FIT_FROZEN.json').read_text())
    assert all(sha(OUT/n) == h for n, h in frozen['files'].items())
    assert json.loads((OUT/'INSTRUMENT_CHECKS.json').read_text())['status'] == 'PASS'
    c2 = original()
    models = {name: load_model(name) for name, *_ in CONFIGS}
    images = np.load('/development/train_images.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load('/development/train_labels.npy', allow_pickle=False)
    folds = np.load('/development/folds.npy', allow_pickle=False)
    ids = np.flatnonzero(folds == 0)
    truth = labels[ids].astype(np.int64)
    assert len(ids) == 10000
    names = [n for n, *_ in CONFIGS] + ['c2', 'c2_no_quadratic']
    scores = {n: np.empty((len(ids), 10), np.float32) for n in names}
    P = np.empty((len(ids), 196), np.float32)
    Z = np.empty((len(ids), 40), np.float32)
    original_preds = np.empty(len(ids), np.int64)
    algebra_disagreements, near_ties = [], []
    maxerr = maxratio = 0.
    for row, idx in enumerate(ids):
        image = images[idx]
        original_preds[row] = c2.predict(image)
        p, z = c2._x(image), c2._Z.copy()
        P[row], Z[row] = p, z
        scores['c2'][row] = c2._F @ c2._W + c2._B
        scores['c2_no_quadratic'][row] = z @ c2._W[:40] + c2._B
        calc, terms = algebra_scores(z, c2._W, c2._B)
        err = np.abs(calc - scores['c2'][row]); bound = score_bounds(terms, c2._B)
        assert np.all(err <= bound)
        maxerr, maxratio = max(maxerr, float(err.max())), max(maxratio, float((err/bound).max()))
        if np.argmax(calc) != original_preds[row]:
            algebra_disagreements.append(int(idx))
        rank = np.argsort(scores['c2'][row])
        if scores['c2'][row, rank[-1]] - scores['c2'][row, rank[-2]] <= bound[rank[-1]]+bound[rank[-2]]:
            near_ties.append(int(idx))
        for name, m in models.items():
            zm = project(p, m)
            f = features(zm, m['meta']['quadratic'])
            scores[name][row] = f @ m['W'] + m['B']
        if (row+1) % 2500 == 0:
            emit('evaluation_progress', images=row+1)
    preds = {n: np.argmax(s, axis=1) for n, s in scores.items()}
    assert np.array_equal(preds['c2'], original_preds)
    assert not algebra_disagreements, 'Investigate arithmetic decision mismatch'
    assert int(np.sum(preds['c2'] == truth)) == 9783, 'Saved development result mismatch'
    check = {'status': 'PASS', 'n': len(ids), 'max_score_reconstruction_error': maxerr,
             'max_error_over_bound': maxratio, 'algebra_decision_disagreements': algebra_disagreements,
             'near_ties_by_conservative_bound': near_ties, 'original_predict_matches': True,
             'saved_C2_development_correct_count_matches': True}
    dump('EVALUATION_CHECKS.json', check)
    np.savez(OUT/'predictions.npz', ids=ids, labels=truth,
             **{'pred_'+n: preds[n] for n in names}, **{'scores_'+n: scores[n] for n in names})
    rows = []
    for name in names:
        correct = int(np.sum(preds[name] == truth))
        if name in models:
            meta = models[name]['meta']
            payload, k, dim = meta['payload_bytes_float_head'], meta['k'], meta['head_features']
        else:
            payload, k, dim = (17624 if name == 'c2' else None), 40, (860 if name == 'c2' else 40)
        cm = np.zeros((10, 10), np.int64)
        np.add.at(cm, (truth, preds[name]), 1)
        row = {'name': name, 'correct': correct, 'errors': len(ids)-correct, 'accuracy_pct': 100*correct/len(ids),
               'k': k, 'head_features': dim, 'raw_parameter_payload_bytes': payload,
               'classwise': [{'digit': c, 'n': int(np.sum(truth==c)), 'correct': int(cm[c,c]),
                              'accuracy_pct': float(100*cm[c,c]/np.sum(truth==c))} for c in range(10)],
               'confusion': cm.tolist()}
        rows.append(row)
    contrast_specs = [('q40', 'q20'), ('q40', 'q60'), ('l40', 'q40'), ('f40', 'q40'),
                      ('q40', 'c2'), ('f40', 'c2'), ('c2', 'c2_no_quadratic')]
    rng = np.random.default_rng(20261009)
    contrasts = []
    for ref, alt in contrast_specs:
        result = paired(preds[ref], preds[alt], truth)
        bootstrap = np.zeros(2000)
        classes = []
        for c in range(10):
            idx = truth == c
            part = paired(preds[ref][idx], preds[alt][idx], truth[idx])
            part['digit'] = c; classes.append(part)
            # Exactly the paired image bootstrap for this accuracy statistic,
            # aggregated into delta-correctness categories within each class.
            counts = np.array([part['broken'], part['n']-part['broken']-part['corrected'], part['corrected']])
            samples = rng.multinomial(part['n'], counts / part['n'], size=2000)
            bootstrap += samples[:, 2] - samples[:, 0]
        ci = np.quantile(bootstrap * 100/len(ids), [.025, .975]).tolist()
        assert sum(result[k] for k in ['both_correct','both_wrong','corrected','broken']) == len(ids)
        assert abs(result['delta_pp'] - 100*(result['corrected']-result['broken'])/len(ids)) < 1e-10
        disagreements = []
        for t in range(10):
            for a in range(10):
                for b in range(10):
                    n = int(np.sum((truth==t) & (preds[ref]==a) & (preds[alt]==b)))
                    if n and a != b:
                        disagreements.append({'true':t,'reference_prediction':a,'alternative_prediction':b,'n':n})
        result.update(reference=ref, alternative=alt, descriptive_95_interval_pp=ci, classwise=classes,
                      changed_class_pairs=disagreements)
        contrasts.append(result)
    dump('RESULTS.json', {'cohort': '10,000 fold-0 development images; adaptively reused, exploratory',
                         'source_sha256': SOURCE_SHA, 'models': rows, 'contrasts': contrasts,
                         'bootstrap': {'replicates':2000,'seed':20261009,'method':'class-stratified multinomial counts of paired correctness differences; equivalent to resampling paired image indices for accuracy',
                                       'interpretation':'descriptive, conditional on fitted models; no fresh confirmatory or training-seed uncertainty claim'},
                         'evaluation_seconds': time.monotonic()-start,
                         'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    # Teaching examples selected only after reporting every group count.
    cp = preds['c2']; good = cp == truth
    true_scores = scores['c2'][np.arange(len(ids)), truth]
    other = scores['c2'].copy(); other[np.arange(len(ids)), truth] = -np.inf
    correct_margin = true_scores - other.max(axis=1)
    pred_true_margin = scores['c2'][np.arange(len(ids)), cp] - true_scores
    fixed = (preds['l40'] != truth) & (preds['q40'] == truth)
    broken = (preds['l40'] == truth) & (preds['q40'] != truth)
    agrees = cp == preds['q40']
    change_margin = scores['c2'][np.arange(len(ids)), cp] - scores['c2'][np.arange(len(ids)), preds['l40']]
    groups = [('ordinary_success',good,correct_margin),('quadratic_corrected',fixed & agrees,change_margin),
              ('quadratic_harmed',broken & agrees,change_margin),('C2_error',~good,pred_true_margin)]
    cases, case_images, case_pooled = [], [], []
    for label, mask, margin in groups:
        members = np.flatnonzero(mask)
        if not len(members):
            cases.append({'case':label,'eligible_n':0}); continue
        order = np.lexsort((ids[members], margin[members]))
        row = int(members[order[(len(order)-1)//2]])
        a = int(cp[row])
        if label.startswith('quadratic_'):
            b = int(preds['l40'][row])
        elif label == 'C2_error':
            b = int(truth[row])
        else:
            s = scores['c2'][row].copy(); s[a] = -np.inf; b = int(np.argmax(s))
        z = Z[row].astype(np.float64)
        I,J = np.triu_indices(40)
        dw = c2._W[:,a].astype(np.float64) - c2._W[:,b].astype(np.float64)
        linear = z*dw[:40]; products=z[I]*z[J]*dw[40:]
        bias=float(c2._B[a])-float(c2._B[b])
        totals={'bias':bias,'linear':float(linear.sum()),'squares':float(products[I==J].sum()),'cross_products':float(products[I!=J].sum())}
        total=sum(totals.values()); actual=float(scores['c2'][row,a])-float(scores['c2'][row,b])
        records=[{'term':f'z{i+1}','value':float(x)} for i,x in enumerate(linear)]
        records += [{'term':f'z{i+1}*z{j+1}','value':float(x)} for i,j,x in zip(I,J,products)]
        records.sort(key=lambda x:abs(x['value']),reverse=True)
        cases.append({'case':label,'eligible_n':len(members),'source_index':int(ids[row]),'evaluation_row':row,
                      'true_digit':int(truth[row]),'C2_digit':a,'competitor':b,'linear_model_digit':int(preds['l40'][row]),
                      'margin':actual,'reconstructed_margin':total,'residual':total-actual,'terms':totals,
                      'largest_terms':records[:8],'remaining_terms_sum':float(sum(x['value'] for x in records[8:])),
                      'panel_array_row':len(case_images)})
        case_images.append(images[ids[row]].copy());case_pooled.append(P[row].reshape(14,14))
    dump('WORKED_CASES.json', {'selection':'lower median of specified relevant C2 margin, original index breaks ties',
                              'quadratic_fixed_total':int(fixed.sum()),'quadratic_harmed_total':int(broken.sum()),
                              'fixed_excluded_C2_disagreement':int((fixed & ~agrees).sum()),
                              'harmed_excluded_C2_disagreement':int((broken & ~agrees).sum()),'cases':cases})
    np.savez(OUT/'worked_case_images.npz', images=np.array(case_images), pooled=np.array(case_pooled))
    # Frozen removal diagnostics retain both label transitions and true-class margins.
    removed=scores['c2_no_quadratic']; rother=removed.copy();rother[np.arange(len(ids)),truth]=-np.inf
    removed_margin=removed[np.arange(len(ids)),truth]-rother.max(axis=1)
    np.savez(OUT/'reliance_margins.npz', ids=ids, full_true_margin=correct_margin, removed_true_margin=removed_margin)
    dump('RELIANCE.json', {'margin_definition':'true digit score minus greatest other-digit score, within frozen C2',
                           'full_margin_quantiles':np.quantile(correct_margin,[0,.1,.5,.9,1]).tolist(),
                           'removed_margin_quantiles':np.quantile(removed_margin,[0,.1,.5,.9,1]).tolist(),
                           'no_label_change_but_margin_change':int(np.sum((cp==preds['c2_no_quadratic']) & (correct_margin!=removed_margin)))})
    emit('evaluation_complete', seconds=time.monotonic()-start,
         accuracies={r['name']:r['accuracy_pct'] for r in rows}, checks=check)


if __name__ == '__main__':
    try:
        {'fit':fit,'evaluate':evaluate}[sys.argv[1]]()
    except Exception as exc:
        dump('FAILURE_'+sys.argv[1]+'.json', {'type':type(exc).__name__,'message':str(exc)})
        raise
