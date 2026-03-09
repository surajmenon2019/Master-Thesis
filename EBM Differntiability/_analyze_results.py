import numpy as np
import os

_dir = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(_dir, 'exp1_results.npy'), allow_pickle=True).item()
print('Keys:', list(data.keys()))
print()

for key in sorted(data.keys()):
    d = data[key]
    print(f'=== {key} ===')
    print(f'  regime: {d.get("regime", "?")}')
    print(f'  display_name: {d.get("display_name", "?")}')
    print(f'  horizon: {d.get("horizon", "?")}')
    print(f'  eval_steps (len): {len(d.get("eval_steps", []))}')
    print(f'  eval_rewards (len): {len(d.get("eval_rewards", []))}')
    if d.get('eval_rewards'):
        r = d['eval_rewards']
        print(f'  eval_rewards first 5: {r[:5]}')
        print(f'  eval_rewards last 5: {r[-5:]}')
        print(f'  eval_rewards min/max/mean: {min(r):.1f} / {max(r):.1f} / {np.mean(r):.1f}')
        print(f'  final 5 avg: {np.mean(r[-5:]):.1f}')
    print(f'  grad_norms (len): {len(d.get("grad_norms", []))}')
    if d.get('grad_norms'):
        gn = d['grad_norms']
        print(f'  grad_norms min/max/mean: {min(gn):.4f} / {max(gn):.4f} / {np.mean(gn):.4f}')
        print(f'  grad_norms last 10 avg: {np.mean(gn[-10:]):.4f}')
    diag = d.get('diagnostics', {})
    if diag.get('grad_variance'):
        gv = diag['grad_variance']
        print(f'  grad_variance entries: {len(gv)}')
        variances = [g['variance'] for g in gv]
        print(f'  grad_variance min/max/mean: {min(variances):.6f} / {max(variances):.6f} / {np.mean(variances):.6f}')
    if diag.get('cumulative_spectral_radius'):
        csr = diag['cumulative_spectral_radius']
        print(f'  spectral_radius entries: {len(csr)}')
        vals = [s['value'] for s in csr]
        print(f'  cumulative_spectral_radius min/max/mean: {min(vals):.4f} / {max(vals):.4f} / {np.mean(vals):.4f}')
    if diag.get('effective_rank'):
        er = diag['effective_rank']
        print(f'  effective_rank entries: {len(er)}')
        ranks = [e['effective_rank'] for e in er]
        print(f'  effective_rank min/max/mean: {min(ranks)} / {max(ranks)} / {np.mean(ranks):.1f}')
    print()

# Summary table
print('\n' + '='*80)
print('SUMMARY TABLE')
print('='*80)
print(f'{"Key":<30} {"Final5 Reward":>14} {"Grad Norm (last10)":>20} {"Grad Var (mean)":>18} {"Spectral Rad":>14} {"Eff Rank":>10}')
print('-'*110)
for key in sorted(data.keys()):
    d = data[key]
    r = d.get('eval_rewards', [])
    gn = d.get('grad_norms', [])
    diag = d.get('diagnostics', {})
    
    final_r = f'{np.mean(r[-5:]):.1f}' if len(r) >= 5 else 'N/A'
    gn_last = f'{np.mean(gn[-10:]):.4f}' if len(gn) >= 10 else 'N/A'
    
    gv = diag.get('grad_variance', [])
    gv_mean = f'{np.mean([g["variance"] for g in gv]):.6f}' if gv else 'N/A'
    
    csr = diag.get('cumulative_spectral_radius', [])
    csr_mean = f'{np.mean([s["value"] for s in csr]):.4f}' if csr else 'N/A'
    
    er = diag.get('effective_rank', [])
    er_mean = f'{np.mean([e["effective_rank"] for e in er]):.1f}' if er else 'N/A'
    
    print(f'{key:<30} {final_r:>14} {gn_last:>20} {gv_mean:>18} {csr_mean:>14} {er_mean:>10}')
