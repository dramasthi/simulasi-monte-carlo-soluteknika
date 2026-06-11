import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# SIMULASI MONTE CARLO - COST OVERRUN PROYEK IT
# PT Soluteknika Informatika
# ============================================================

np.random.seed(42)
N = 10_000  # Jumlah iterasi

# ── Komponen Biaya Baseline (Deterministik) ──────────────────
# Nilai dalam Rupiah
baseline_components = {
    'Hardware'       : 80_000_000,
    'Software'       : 60_000_000,
    'Tenaga Kerja'   : 75_000_000,
    'Infrastruktur'  : 18_300_000,
    'Lain-lain'      : 10_000_000,
}
BASELINE = sum(baseline_components.values())   # Rp 243,300,000
CONSTRAINT_MAX = BASELINE * 1.20               # 120% dari baseline

print("=" * 70)
print("  SIMULASI MONTE CARLO — COST OVERRUN PROYEK IT")
print("  PT Soluteknika Informatika")
print("=" * 70)
print(f"  Baseline Deterministik    : Rp {BASELINE:>15,.0f}")
print(f"  Constraint Maksimal (120%): Rp {CONSTRAINT_MAX:>15,.0f}")
print(f"  Jumlah Iterasi            : {N:,}")
print()

# ── Fungsi Simulasi Biaya ─────────────────────────────────────
def simulate_cost(scenario='S0'):
    """
    Simulasi total biaya proyek berdasarkan skenario.
    Setiap komponen bisa mengalami overrun sesuai distribusinya.
    """
    # Hardware: Triangular(0.9, 1.0, 1.25) × baseline
    hw_factor = np.random.triangular(0.90, 1.00, 1.25, N)
    # Software: Triangular(0.95, 1.0, 1.30) × baseline
    sw_factor = np.random.triangular(0.95, 1.00, 1.30, N)
    # Tenaga Kerja: Normal(mean=1.05, std=0.10)
    tk_factor = np.random.normal(1.05, 0.10, N)
    # Infrastruktur: Uniform(0.95, 1.20)
    inf_factor = np.random.uniform(0.95, 1.20, N)
    # Lain-lain: Triangular(0.9, 1.0, 1.5)
    misc_factor = np.random.triangular(0.90, 1.00, 1.50, N)

    # S2: Vendor dikunci → kurangi variansi hardware & software
    if scenario == 'S2':
        hw_factor  = np.random.triangular(0.95, 1.00, 1.10, N)
        sw_factor  = np.random.triangular(0.97, 1.00, 1.10, N)

    total = (
        baseline_components['Hardware']      * hw_factor  +
        baseline_components['Software']      * sw_factor  +
        baseline_components['Tenaga Kerja']  * tk_factor  +
        baseline_components['Infrastruktur'] * inf_factor +
        baseline_components['Lain-lain']     * misc_factor
    )
    # Clip agar tidak negatif
    return np.clip(total, 0, None)

# ── Jalankan Simulasi ─────────────────────────────────────────
cost_S0 = simulate_cost('S0')   # Baseline tanpa cadangan
cost_S2 = simulate_cost('S2')   # Optimistic (vendor dikunci)

# ── Threshold per Skenario ────────────────────────────────────
threshold_S0 = BASELINE                        # S0: tidak ada cadangan
threshold_S1 = BASELINE * 1.10               # S1: cadangan 10%
p95_S0       = np.percentile(cost_S0, 95)    # S3: threshold = P95 aktual
threshold_S3 = p95_S0

scenarios = {
    'S0 (Baseline)':          (cost_S0, threshold_S0, 0.0),
    'S1 (Standard 10%)':      (cost_S0, threshold_S1, 10.0),
    'S2 (Optimistic Vendor)': (cost_S2, threshold_S0, 0.0),
    'S3 (Data-Driven P95)':   (cost_S0, threshold_S3,
                                (threshold_S3 - BASELINE) / BASELINE * 100),
}

# ── Hitung KPI tiap Skenario ──────────────────────────────────
results = []
for name, (cost_arr, threshold, cadangan_pct) in scenarios.items():
    p_overrun = np.mean(cost_arr > threshold)
    e_cost    = np.mean(cost_arr)
    p90_val   = np.percentile(cost_arr, 90)
    p95_val   = np.percentile(cost_arr, 95)
    results.append({
        'Skenario'   : name,
        'Threshold'  : threshold,
        'Cadangan'   : cadangan_pct,
        'P(overrun)' : p_overrun * 100,
        'E[Cost]'    : e_cost,
        'P90'        : p90_val,
        'P95'        : p95_val,
    })

df = pd.DataFrame(results)

# ── Tampilkan Tabel Ringkasan ─────────────────────────────────
print(f"{'Skenario':<30} {'Threshold':>15}  {'Cadangan':>8}  {'P(overrun)':>10}  {'E[Cost]':>15}  {'P95':>15}")
print("-" * 100)
for _, row in df.iterrows():
    print(f"{row['Skenario']:<30} "
          f"Rp {row['Threshold']:>13,.0f}  "
          f"{row['Cadangan']:>7.1f}%  "
          f"{row['P(overrun)']:>9.2f}%  "
          f"Rp {row['E[Cost]']:>13,.0f}  "
          f"Rp {row['P95']:>13,.0f}")

# ── Kesimpulan ────────────────────────────────────────────────
best = df.loc[df['P(overrun)'].idxmin()]
s3   = df[df['Skenario'].str.startswith('S3')].iloc[0]

print()
print("=" * 70)
print("  KESIMPULAN & REKOMENDASI")
print("-" * 70)
print(f"  Skenario terbaik : {best['Skenario']}")
print(f"  Dana cadangan    : {s3['Cadangan']:.1f}% dari baseline → Rp {s3['Threshold']:,.0f}")
print(f"  P(overrun)       : {s3['P(overrun)']:.2f}% (target <5%) {'✅' if s3['P(overrun)'] <= 5 else '⚠️'}")
print(f"  Constraint       : {s3['Cadangan']:.1f}% ≤ 20% {'✅' if s3['Cadangan'] <= 20 else '⚠️'}")
print(f"  Expected Cost    : Rp {s3['E[Cost]']:,.0f}")
print(f"  P90              : Rp {s3['P90']:,.0f}")
print(f"  P95 (Threshold)  : Rp {s3['P95']:,.0f}")
print()
print("  Catatan: Skenario S2 (lock vendor) juga mengurangi variansi biaya,")
print("  disarankan dikombinasikan dengan S3 sebagai strategi mitigasi optimal.")
print("=" * 70)

# ── Visualisasi ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Simulasi Monte Carlo – Cost Overrun Proyek IT\nPT Soluteknika Informatika',
             fontsize=13, fontweight='bold')

rupiah_fmt = mticker.FuncFormatter(lambda x, _: f'Rp {x/1e6:.0f}M')

# --- Plot 1: Histogram S0 vs S2 ---
ax1 = axes[0]
ax1.hist(cost_S0 / 1e6, bins=60, alpha=0.6, color='steelblue',
         edgecolor='white', label='S0 (Baseline)', density=True)
ax1.hist(cost_S2 / 1e6, bins=60, alpha=0.6, color='green',
         edgecolor='white', label='S2 (Optimistic Vendor)', density=True)

# Garis threshold
for label, val, col, ls in [
    ('Baseline',       BASELINE / 1e6,      'gray',       '-'),
    ('S1 (+10%)',       threshold_S1 / 1e6,  'orange',     '--'),
    ('S3 P95',         threshold_S3 / 1e6,  'darkred',    ':'),
    ('Max (120%)',      CONSTRAINT_MAX/1e6,  'black',      '-.'),
]:
    ax1.axvline(val, color=col, linestyle=ls, linewidth=1.8, label=f'{label} = Rp {val:.1f}M')

ax1.set_xlabel('Total Biaya Proyek (Juta Rp)', fontsize=11)
ax1.set_ylabel('Densitas', fontsize=11)
ax1.set_title('Distribusi Biaya: S0 vs S2', fontsize=12)
ax1.legend(fontsize=8)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# --- Plot 2: Bar chart P(overrun) per skenario ---
ax2 = axes[1]
colors_bar = ['#e74c3c', '#f39c12', '#27ae60', '#2980b9']
bars = ax2.bar(df['Skenario'], df['P(overrun)'],
               color=colors_bar, alpha=0.85, edgecolor='black', linewidth=1)

# Garis target 5%
ax2.axhline(5, color='red', linestyle='--', linewidth=2, label='Target P(overrun) = 5%')

# Nilai di atas bar
for bar, val in zip(bars, df['P(overrun)']):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Probabilitas Overrun (%)', fontsize=11)
ax2.set_title('P(Overrun) per Skenario', fontsize=12)
ax2.set_ylim(0, max(df['P(overrun)']) * 1.25)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig('simulasiuas_hasil.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Grafik disimpan ke: simulasiuas_hasil.png")
