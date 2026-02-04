import numpy as np, matplotlib.pyplot as plt

def plot_ts(x, xhat, resid, overload_ma, alerts, gts, out_path):
    t = np.arange(len(x))
    plt.figure(figsize=(10,6))
    plt.plot(t, x, label='x(t)')
    plt.plot(t, xhat, label='x̂(t)')
    plt.plot(t, resid, label='resid', alpha=0.7)
    for g in gts:
        plt.axvline(g, color='g', linestyle='--', alpha=0.6, label='gt' if g==gts[0] else None)
    for a in alerts:
        plt.axvline(a, color='r', linestyle=':', alpha=0.7, label='alert' if a==alerts[0] else None)
    ax2 = plt.twinx()
    ax2.plot(t, overload_ma, label='overload_ma', alpha=0.5)
    ax2.set_ylabel('overload_ma')
    plt.title('Time series + reconstruction + alerts')
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines+lines2, labels+labels2, loc='upper right')
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_features(overload, flip_ma, out_path):
    t = np.arange(len(overload))
    plt.figure(figsize=(10,4))
    plt.plot(t, overload, label='overload (0/1)', alpha=0.6)
    plt.plot(t, flip_ma, label='flip_rate (MA)', alpha=0.8)
    plt.legend(); plt.title('Bit-plane features')
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_hist(run_lengths, overload, out_path):
    import numpy as np
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    if len(run_lengths)>0:
        mx = min(100, max(run_lengths))
        bins = np.arange(1, mx+2)
        plt.hist(run_lengths, bins=bins, edgecolor='k')
    plt.xlabel('run length'); plt.ylabel('count'); plt.title('Run-length histogram')
    plt.subplot(1,2,2)
    plt.hist(overload, bins=20, edgecolor='k')
    plt.xlabel('overload_ma'); plt.title('Overload histogram')
    plt.tight_layout(); plt.savefig(out_path); plt.close()
