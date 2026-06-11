import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ============================================================
# VISUALISASI LENGKAP — 1.000 REPLIKASI SIMULASI MONTE CARLO
# PT Soluteknika Informatika
# ============================================================

MASTER_SEED = 42
N           = 10_000
N_REP       = 1_000

baseline_components = {
    'Hardware'     : 80_000_000,
    'Software'     : 60_000_000,
    'Tenaga Kerja' : 75_000_000,
    'Infrastruktur': 18_300_000,
    'Lain-lain'    : 10_000_000,
}
BASELINE       = sum(baseline_components.values())
CONSTRAINT_MAX = BASELINE * 1.20
threshold_S1   = BASELINE * 1.10

# ── Load atau regenerate data replikasi ───────────────────────
try:
    df_rep = pd.read_csv('replikasi_results.csv')
    print(f"✓ Data dimuat dari replikasi_results.csv ({len(df_rep)} replikasi)")
except FileNotFoundError:
    print("Data tidak ditemukan, menjalankan ulang simulasi...")
    def simulate_one_rep(scenario, rng):
        hw   = rng.triangular(0.90, 1.00, 1.25, N)
        sw   = rng.triangular(0.95, 1.00, 1.30, N)
        tk   = rng.normal(1.05, 0.10, N)
        inf_ = rng.uniform(0.95, 1.20, N)
        misc = rng.triangular(0.90, 1.00, 1.50, N)
        if scenario == 'S2':
            hw = rng.triangular(0.95, 1.00, 1.10, N)
            sw = rng.triangular(0.97, 1.00, 1.10, N)
        total = (baseline_components['Hardware']      * hw   +
                 baseline_components['Software']      * sw   +
                 baseline_components['Tenaga Kerja']  * tk   +
                 baseline_components['Infrastruktur'] * inf_ +
                 baseline_components['Lain-lain']     * misc)
        return np.clip(total, 0, None)

    rng_master = np.random.default_rng(MASTER_SEED)
    seeds = rng_master.integers(0, 1_000_000, size=N_REP)
    rep_records = []
    for i, seed in enumerate(seeds):
        c0 = simulate_one_rep('S0', np.random.default_rng(seed))
        c2 = simulate_one_rep('S2', np.random.default_rng(seed + 500_000))
        p95_rep = np.percentile(c0, 95)
        rep_records.append({
            'rep': i+1,
            'S0_mean': np.mean(c0), 'S0_p90': np.percentile(c0,90),
            'S0_p95': p95_rep,      'S0_std': np.std(c0),
            'S0_pover': np.mean(c0 > BASELINE)*100,
            'S0_pover_S1': np.mean(c0 > threshold_S1)*100,
            'S2_mean': np.mean(c2), 'S2_p95': np.percentile(c2,95),
            'S2_std': np.std(c2),   'S2_pover': np.mean(c2 > BASELINE)*100,
            'S3_threshold': p95_rep,
            'S3_cadangan': (p95_rep - BASELINE)/BASELINE*100,
            'S3_pover': np.mean(c0 > p95_rep)*100,
        })
    df_rep = pd.DataFrame(rep_records)
    df_rep.to_csv('replikasi_results.csv', index=False)
    print(f"✓ Simulasi selesai & disimpan.")

def ci95(s): return np.percentile(s, 2.5), np.percentile(s, 97.5)

# ── Ambil data final run untuk histogram distribusi ───────────
np.random.seed(MASTER_SEED)
def simulate_final(scenario):
    hw   = np.random.triangular(0.90, 1.00, 1.25, N)
    sw   = np.random.triangular(0.95, 1.00, 1.30, N)
    tk   = np.random.normal(1.05, 0.10, N)
    inf_ = np.random.uniform(0.95, 1.20, N)
    misc = np.random.triangular(0.90, 1.00, 1.50, N)
    if scenario == 'S2':
        hw = np.random.triangular(0.95, 1.00, 1.10, N)
        sw = np.random.triangular(0.97, 1.00, 1.10, N)
    return np.clip(
        baseline_components['Hardware']*hw + baseline_components['Software']*sw +
        baseline_components['Tenaga Kerja']*tk + baseline_components['Infrastruktur']*inf_ +
        baseline_components['Lain-lain']*misc, 0, None)

cost_S0 = simulate_final('S0')
cost_S2 = simulate_final('S2')
thr_S3  = np.percentile(cost_S0, 95)

# ============================================================
# GAMBAR 1 — Dashboard Utama (3×3)
# ============================================================
print("\nMembuat Gambar 1: Dashboard Utama...")
fig1 = plt.figure(figsize=(22, 16))
fig1.patch.set_facecolor('#1a1a2e')
gs = GridSpec(3, 3, figure=fig1, hspace=0.5, wspace=0.38)

title_kw  = dict(fontsize=11, fontweight='bold', color='white', pad=10)
label_kw  = dict(fontsize=10, color='#cccccc')
tick_kw   = dict(colors='#aaaaaa', labelsize=8)
grid_kw   = dict(alpha=0.2, linestyle='--', color='#555555')
rupiah_M  = mticker.FuncFormatter(lambda x, _: f'Rp {x/1e6:.1f}M')

def style_ax(ax, title):
    ax.set_facecolor('#16213e')
    ax.set_title(title, **title_kw)
    ax.tick_params(axis='both', **tick_kw)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

# ── [0,0] Histogram distribusi biaya S0 vs S2 ────────────────
ax = fig1.add_subplot(gs[0, :2])
style_ax(ax, '① Distribusi Total Biaya Proyek — S0 vs S2  (Representasi 1 Run)')
ax.hist(cost_S0/1e6, bins=70, alpha=0.6, color='#3498db', edgecolor='none',
        density=True, label='S0 – Baseline')
ax.hist(cost_S2/1e6, bins=70, alpha=0.6, color='#2ecc71', edgecolor='none',
        density=True, label='S2 – Optimistic Vendor')
for val, col, ls, lbl in [
    (BASELINE/1e6,       '#aaaaaa', '-',  f'Baseline  Rp {BASELINE/1e6:.1f}M'),
    (threshold_S1/1e6,   '#f39c12', '--', f'S1 +10%   Rp {threshold_S1/1e6:.1f}M'),
    (thr_S3/1e6,         '#e74c3c', ':',  f'S3 P95    Rp {thr_S3/1e6:.1f}M'),
    (CONSTRAINT_MAX/1e6, 'white',   '-.', f'Max 120%  Rp {CONSTRAINT_MAX/1e6:.1f}M'),
]:
    ax.axvline(val, color=col, linestyle=ls, linewidth=1.8, label=lbl)
ax.set_xlabel('Total Biaya (Juta Rp)', **label_kw)
ax.set_ylabel('Densitas', **label_kw)
ax.legend(fontsize=8, ncol=2, framealpha=0.2, labelcolor='white')
ax.grid(**grid_kw)
ax.xaxis.set_major_formatter(rupiah_M)

# ── [0,2] Pie komposisi baseline ─────────────────────────────
ax = fig1.add_subplot(gs[0, 2])
ax.set_facecolor('#16213e')
ax.set_title('② Komposisi\nBiaya Baseline', **title_kw)
pie_colors = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']
wedges, texts, autotexts = ax.pie(
    baseline_components.values(), labels=baseline_components.keys(),
    autopct='%1.1f%%', colors=pie_colors, startangle=130,
    pctdistance=0.78, wedgeprops=dict(edgecolor='#1a1a2e', linewidth=2))
for t in texts: t.set_color('white'); t.set_fontsize(8)
for at in autotexts: at.set_fontweight('bold'); at.set_fontsize(8); at.set_color('white')
ax.text(0, -1.45, f'Total: Rp {BASELINE/1e6:.1f}M',
        ha='center', color='#aaaaaa', fontsize=9)

# ── [1,0:2] CDF semua skenario ────────────────────────────────
ax = fig1.add_subplot(gs[1, :2])
style_ax(ax, '③ Cumulative Distribution Function (CDF) — Semua Skenario')
scenario_plot = [
    ('S0 – Baseline',         cost_S0, '#e74c3c'),
    ('S2 – Optimistic Vendor', cost_S2, '#2ecc71'),
]
for lbl, arr, col in scenario_plot:
    sx  = np.sort(arr)
    cdf = np.arange(1, N+1) / N * 100
    ax.plot(sx/1e6, cdf, color=col, linewidth=2.2, label=lbl)
for pct, col in [(90,'#f39c12'),(95,'#e74c3c')]:
    ax.axhline(pct, color=col, linestyle=':', linewidth=1, alpha=0.6)
    ax.text(cost_S0.min()/1e6, pct+0.8, f'P{pct}', color=col, fontsize=8)
for val, col, ls, lbl in [
    (BASELINE/1e6,       '#aaaaaa', '-',  'Baseline'),
    (threshold_S1/1e6,   '#f39c12', '--', 'S1 +10%'),
    (thr_S3/1e6,         '#e74c3c', ':',  'S3 P95'),
    (CONSTRAINT_MAX/1e6, 'white',   '-.', 'Max 120%'),
]:
    ax.axvline(val, color=col, linestyle=ls, linewidth=1.5, label=lbl)
ax.set_xlabel('Total Biaya (Juta Rp)', **label_kw)
ax.set_ylabel('Persentil Kumulatif (%)', **label_kw)
ax.set_ylim(0, 100)
ax.legend(fontsize=8, ncol=3, framealpha=0.2, labelcolor='white')
ax.grid(**grid_kw)
ax.xaxis.set_major_formatter(rupiah_M)

# ── [1,2] Bar P(overrun) grand mean ──────────────────────────
ax = fig1.add_subplot(gs[1, 2])
style_ax(ax, '④ P(Overrun)\nGrand Mean 1.000 Rep')
slabels = ['S0', 'S1', 'S2', 'S3']
pmeans  = [df_rep['S0_pover'].mean(), df_rep['S0_pover_S1'].mean(),
           df_rep['S2_pover'].mean(), df_rep['S3_pover'].mean()]
pstds   = [df_rep['S0_pover'].std(),  df_rep['S0_pover_S1'].std(),
           df_rep['S2_pover'].std(),  df_rep['S3_pover'].std()]
bcolors = ['#e74c3c','#f39c12','#2ecc71','#3498db']
bars = ax.bar(slabels, pmeans, yerr=pstds, capsize=5, color=bcolors,
              alpha=0.85, edgecolor='#1a1a2e', linewidth=1.5,
              error_kw=dict(elinewidth=1.5, ecolor='white'))
ax.axhline(5, color='#ff6b6b', linestyle='--', linewidth=2, label='Target 5%')
for bar, v, s in zip(bars, pmeans, pstds):
    ax.text(bar.get_x()+bar.get_width()/2, v+s+0.5, f'{v:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='white')
ax.set_ylabel('P(Overrun) %', **label_kw)
ax.set_ylim(0, max(pmeans)*1.35)
ax.legend(fontsize=9, framealpha=0.2, labelcolor='white')
ax.grid(axis='y', **grid_kw)

# ── [2,0] Konvergensi running mean ────────────────────────────
ax = fig1.add_subplot(gs[2, 0])
style_ax(ax, '⑤ Konvergensi Running Mean\nE[Cost] S0')
rm = df_rep['S0_mean'].cumsum() / (np.arange(N_REP)+1)
ax.plot(df_rep['rep'], rm/1e6, color='#3498db', linewidth=1.5)
ax.fill_between(df_rep['rep'], (rm-rm.std())/1e6, (rm+rm.std())/1e6,
                alpha=0.15, color='#3498db')
ax.axhline(df_rep['S0_mean'].mean()/1e6, color='#e74c3c', linestyle='--',
           linewidth=1.5, label=f"Konvergen: Rp {df_rep['S0_mean'].mean()/1e6:.3f}M")
ax.set_xlabel('Replikasi ke-', **label_kw)
ax.set_ylabel('Running Mean (Juta Rp)', **label_kw)
ax.legend(fontsize=8, framealpha=0.2, labelcolor='white')
ax.grid(**grid_kw)
ax.yaxis.set_major_formatter(rupiah_M)

# ── [2,1] Distribusi E[Cost] 1000 replikasi ──────────────────
ax = fig1.add_subplot(gs[2, 1])
style_ax(ax, '⑥ Distribusi E[Cost] S0\n1.000 Replikasi')
ax.hist(df_rep['S0_mean']/1e6, bins=40, color='#3498db', edgecolor='none', alpha=0.8)
lo, hi = ci95(df_rep['S0_mean'])
ax.axvline(df_rep['S0_mean'].mean()/1e6, color='white', linewidth=2,
           label=f"Mean: Rp {df_rep['S0_mean'].mean()/1e6:.2f}M")
ax.axvline(lo/1e6, color='#f39c12', linestyle='--', linewidth=1.5,
           label=f"95% CI: [{lo/1e6:.2f}–{hi/1e6:.2f}M]")
ax.axvline(hi/1e6, color='#f39c12', linestyle='--', linewidth=1.5)
ax.set_xlabel('E[Cost] (Juta Rp)', **label_kw)
ax.set_ylabel('Frekuensi', **label_kw)
ax.legend(fontsize=8, framealpha=0.2, labelcolor='white')
ax.grid(**grid_kw)
ax.xaxis.set_major_formatter(rupiah_M)

# ── [2,2] Distribusi P95 1000 replikasi ──────────────────────
ax = fig1.add_subplot(gs[2, 2])
style_ax(ax, '⑦ Distribusi P95 S0\n1.000 Replikasi')
ax.hist(df_rep['S0_p95']/1e6, bins=40, color='#e74c3c', edgecolor='none', alpha=0.8)
lo95, hi95 = ci95(df_rep['S0_p95'])
ax.axvline(df_rep['S0_p95'].mean()/1e6, color='white', linewidth=2,
           label=f"Mean P95: Rp {df_rep['S0_p95'].mean()/1e6:.2f}M")
ax.axvline(lo95/1e6, color='#f39c12', linestyle='--', linewidth=1.5,
           label=f"95% CI: [{lo95/1e6:.2f}–{hi95/1e6:.2f}M]")
ax.axvline(hi95/1e6, color='#f39c12', linestyle='--', linewidth=1.5)
ax.set_xlabel('P95 Biaya (Juta Rp)', **label_kw)
ax.set_ylabel('Frekuensi', **label_kw)
ax.legend(fontsize=8, framealpha=0.2, labelcolor='white')
ax.grid(**grid_kw)
ax.xaxis.set_major_formatter(rupiah_M)

# Judul utama
fig1.suptitle(
    'SIMULASI MONTE CARLO — ANALISIS COST OVERRUN PROYEK IT\n'
    'PT Soluteknika Informatika  |  1.000 Replikasi × 10.000 Iterasi  |  Total: 10.000.000 Sampel',
    fontsize=14, fontweight='bold', color='white', y=0.995)

plt.savefig('gambar1_dashboard.png', dpi=300, bbox_inches='tight',
            facecolor=fig1.get_facecolor())
print("✓ Gambar 1 disimpan: gambar1_dashboard.png")
plt.close()

# ============================================================
# GAMBAR 2 — Analisis Perbandingan Skenario (2×2)
# ============================================================
print("Membuat Gambar 2: Analisis Skenario...")
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.patch.set_facecolor('#f8f9fa')
fig2.suptitle('Perbandingan Skenario — Cost Overrun Proyek IT\n'
              'PT Soluteknika Informatika  |  1.000 Replikasi',
              fontsize=14, fontweight='bold', color='#2c3e50')

# [0,0] Box plot E[Cost] per skenario ─────────────────────────
ax = axes2[0, 0]
ax.set_facecolor('#fdfdfd')
data_box = [df_rep['S0_mean']/1e6, df_rep['S0_mean']/1e6,
            df_rep['S2_mean']/1e6, df_rep['S0_mean']/1e6]
lbl_box  = ['S0\nBaseline','S1\n+10%','S2\nOpt.Vendor','S3\nP95']
bp = ax.boxplot(data_box, tick_labels=lbl_box, patch_artist=True,
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
for patch, col in zip(bp['boxes'], ['#e74c3c','#f39c12','#2ecc71','#3498db']):
    patch.set_facecolor(col); patch.set_alpha(0.75)
ax.axhline(BASELINE/1e6, color='gray', linestyle='--', linewidth=1.5,
           label=f'Baseline Rp {BASELINE/1e6:.1f}M')
ax.set_ylabel('E[Cost] (Juta Rp)', fontsize=11)
ax.set_title('① Box Plot E[Cost] per Skenario\n(1.000 Replikasi)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.yaxis.set_major_formatter(rupiah_M)

# [0,1] Scatter E[Cost] S0 vs S2 ─────────────────────────────
ax = axes2[0, 1]
ax.set_facecolor('#fdfdfd')
sc = ax.scatter(df_rep['S0_mean']/1e6, df_rep['S2_mean']/1e6,
                alpha=0.3, s=15, c=df_rep['S0_pover'], cmap='RdYlGn_r')
plt.colorbar(sc, ax=ax, label='P(Overrun) S0 (%)')
mn = min(df_rep[['S0_mean','S2_mean']].min())/1e6
mx = max(df_rep[['S0_mean','S2_mean']].max())/1e6
ax.plot([mn,mx],[mn,mx],'k--', linewidth=1.5, label='S0 = S2 (garis paritas)')
ax.set_xlabel('E[Cost] S0 (Juta Rp)', fontsize=11)
ax.set_ylabel('E[Cost] S2 (Juta Rp)', fontsize=11)
ax.set_title('② Scatter E[Cost]: S0 vs S2\n(warna = P(overrun) S0)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle='--')
ax.xaxis.set_major_formatter(rupiah_M); ax.yaxis.set_major_formatter(rupiah_M)

# [1,0] Violin plot P(overrun) per skenario ───────────────────
ax = axes2[1, 0]
ax.set_facecolor('#fdfdfd')
vdata = [df_rep['S0_pover'], df_rep['S0_pover_S1'],
         df_rep['S2_pover'], df_rep['S3_pover']]
vp = ax.violinplot(vdata, positions=[1,2,3,4], showmedians=True, showextrema=True)
for body, col in zip(vp['bodies'], ['#e74c3c','#f39c12','#2ecc71','#3498db']):
    body.set_facecolor(col); body.set_alpha(0.6)
vp['cmedians'].set_color('white'); vp['cmedians'].set_linewidth(2)
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(['S0\nBaseline','S1\n+10%','S2\nOpt.Vendor','S3\nP95'])
ax.axhline(5, color='red', linestyle='--', linewidth=2, label='Target ≤ 5%')
ax.set_ylabel('P(Overrun) %', fontsize=11)
ax.set_title('③ Violin Plot P(Overrun)\n(Variabilitas 1.000 Replikasi)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3, linestyle='--')

# [1,1] Running std (stabilitas simulasi) ─────────────────────
ax = axes2[1, 1]
ax.set_facecolor('#fdfdfd')
for col_name, label, color in [
    ('S0_mean',     'E[Cost] S0',    '#3498db'),
    ('S0_p95',      'P95 S0',        '#e74c3c'),
]:
    series = df_rep[col_name]
    running_std = [series.iloc[:i+1].std() if i > 0 else 0 for i in range(len(series))]
    ax.plot(df_rep['rep'], np.array(running_std)/1e6, label=label, linewidth=1.8, color=color)
ax.set_xlabel('Replikasi ke-', fontsize=11)
ax.set_ylabel('Running Std Dev (Juta Rp)', fontsize=11)
ax.set_title('④ Konvergensi Std Dev\n(Stabilitas Simulasi)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle='--')
ax.yaxis.set_major_formatter(rupiah_M)

plt.tight_layout()
plt.savefig('gambar2_perbandingan_skenario.png', dpi=300, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
print("✓ Gambar 2 disimpan: gambar2_perbandingan_skenario.png")
plt.close()

# ============================================================
# GAMBAR 3 — Ringkasan Eksekutif (1 halaman)
# ============================================================
print("Membuat Gambar 3: Ringkasan Eksekutif...")
fig3 = plt.figure(figsize=(16, 10))
fig3.patch.set_facecolor('#0d1117')
gs3 = GridSpec(2, 4, figure=fig3, hspace=0.55, wspace=0.45)

def style_dark(ax, title):
    ax.set_facecolor('#161b22')
    ax.set_title(title, fontsize=10, fontweight='bold', color='white', pad=8)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

# KPI cards di baris atas (4 metric boxes)
kpi_data = [
    ('E[Cost] Grand Mean',    df_rep['S0_mean'].mean(),   'Rp',  '#58a6ff'),
    ('P95 Grand Mean',        df_rep['S0_p95'].mean(),    'Rp',  '#f85149'),
    ('P(Overrun) S0',         df_rep['S0_pover'].mean(),  '%',   '#ffa657'),
    ('P(Overrun) S3',         df_rep['S3_pover'].mean(),  '%',   '#3fb950'),
]
for idx, (title, val, unit, col) in enumerate(kpi_data):
    ax = fig3.add_subplot(gs3[0, idx])
    ax.set_facecolor(col + '22')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(2)
    if unit == 'Rp':
        val_str = f'Rp {val/1e6:.2f}M'
    else:
        val_str = f'{val:.2f}%'
    col_map = ['S0_mean','S0_p95','S0_pover','S3_pover']
    lo, hi = ci95(df_rep[col_map[idx]])
    ax.text(0.5, 0.62, val_str, ha='center', va='center',
            fontsize=16, fontweight='bold', color=col, transform=ax.transAxes)
    ax.text(0.5, 0.30, title, ha='center', va='center',
            fontsize=9, color='#8b949e', transform=ax.transAxes)
    ax.text(0.5, 0.12, f'N=1.000 rep', ha='center', va='center',
            fontsize=7, color='#555555', transform=ax.transAxes)

# Histogram distribusi final di baris bawah
ax_h = fig3.add_subplot(gs3[1, :2])
style_dark(ax_h, '① Distribusi Biaya (S0 vs S2) — Final Run')
ax_h.hist(cost_S0/1e6, bins=60, alpha=0.6, color='#58a6ff', edgecolor='none',
          density=True, label='S0')
ax_h.hist(cost_S2/1e6, bins=60, alpha=0.6, color='#3fb950', edgecolor='none',
          density=True, label='S2')
for val, col, ls in [(BASELINE/1e6,'#aaaaaa','-'),(threshold_S1/1e6,'#ffa657','--'),
                     (thr_S3/1e6,'#f85149',':'),(CONSTRAINT_MAX/1e6,'white','-.')]:
    ax_h.axvline(val, color=col, linestyle=ls, linewidth=1.5)
ax_h.set_xlabel('Biaya (Juta Rp)', color='#8b949e', fontsize=9)
ax_h.set_ylabel('Densitas', color='#8b949e', fontsize=9)
ax_h.legend(fontsize=9, framealpha=0.1, labelcolor='white')
ax_h.grid(alpha=0.15, linestyle='--', color='#333333')
ax_h.xaxis.set_major_formatter(rupiah_M)

# Konvergensi di baris bawah kanan
ax_k = fig3.add_subplot(gs3[1, 2:])
style_dark(ax_k, '② Konvergensi & 95% CI — E[Cost] S0')
rm = df_rep['S0_mean'].cumsum() / (np.arange(N_REP)+1)
ax_k.plot(df_rep['rep'], rm/1e6, color='#58a6ff', linewidth=1.5, label='Running Mean')
ax_k.fill_between(df_rep['rep'],
                  (rm - 2*df_rep['S0_mean'].std()/np.sqrt(df_rep['rep']))/1e6,
                  (rm + 2*df_rep['S0_mean'].std()/np.sqrt(df_rep['rep']))/1e6,
                  alpha=0.2, color='#58a6ff', label='95% CI')
ax_k.axhline(df_rep['S0_mean'].mean()/1e6, color='#f85149', linestyle='--',
             linewidth=1.5, label=f"Konvergen: {df_rep['S0_mean'].mean()/1e6:.3f}M")
ax_k.set_xlabel('Replikasi ke-', color='#8b949e', fontsize=9)
ax_k.set_ylabel('Running Mean E[Cost] (Juta Rp)', color='#8b949e', fontsize=9)
ax_k.legend(fontsize=9, framealpha=0.1, labelcolor='white')
ax_k.grid(alpha=0.15, linestyle='--', color='#333333')
ax_k.yaxis.set_major_formatter(rupiah_M)

fig3.suptitle(
    'RINGKASAN EKSEKUTIF — SIMULASI MONTE CARLO COST OVERRUN\n'
    'PT Soluteknika Informatika  |  1.000 Replikasi × 10.000 Iterasi',
    fontsize=13, fontweight='bold', color='white', y=1.01)

plt.savefig('gambar3_ringkasan_eksekutif.png', dpi=300, bbox_inches='tight',
            facecolor=fig3.get_facecolor())
print("✓ Gambar 3 disimpan: gambar3_ringkasan_eksekutif.png")
plt.close()

print("\n" + "="*55)
print("  SELESAI — 3 gambar berhasil dibuat:")
print("  1. gambar1_dashboard.png        (7 panel, dark theme)")
print("  2. gambar2_perbandingan_skenario.png (4 panel)")
print("  3. gambar3_ringkasan_eksekutif.png  (KPI cards + chart)")
print("="*55)
