import yaml, numpy as np
from pathlib import Path
from jinja2 import Template
from pipeline.noise import synth_series, inject_anomalies
from pipeline.dm_codec import dm_encode, flip_rate, run_lengths
from pipeline.stream_detector import ewma, online_threshold, detect_on_overload
from pipeline.viz import plot_ts, plot_features, plot_hist

def metrics_from_alerts(alerts, gts, tol=30):
    # match each gt to earliest alert within [t0, t0+tol]
    gts = sorted(gts)
    alerts = sorted(alerts)
    used = set(); tp=0; delays=[]
    for g in gts:
        cand = [a for a in alerts if a>=g and a<=g+tol and a not in used]
        if cand:
            tp += 1; used.add(cand[0]); delays.append(cand[0]-g)
    fp = len([a for a in alerts if all(not (g<=a<=g+tol) for g in gts)])
    fn = len(gts) - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec / (prec+rec + 1e-9)
    avg_delay = float(np.mean(delays)) if delays else float('nan')
    return prec, rec, f1, avg_delay

def run(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    n = cfg["data"]["length"]
    base = synth_series(n,
                        trend=cfg["data"]["pattern"].get("trend"),
                        season=cfg["data"]["pattern"].get("season"),
                        noise_sigma=cfg["data"]["pattern"].get("noise_sigma",0.1))
    x, gts = inject_anomalies(base, cfg.get("anomalies",[]))
    # DM encode
    s, xhat, delta, overload, resid = dm_encode(x,
        delta0=cfg["dm"]["delta0"],
        gamma=cfg["dm"]["gamma"],
        theta_k=cfg["dm"]["theta_k"])
    overload_ma = ewma(overload, alpha=cfg["detect"]["ewma_alpha"])
    flip_ma = flip_rate(s, win=cfg["detect"]["window"])
    thr = online_threshold(overload_ma, z_k=cfg["detect"]["z_k"], warm=200)
    alerts = detect_on_overload(overload_ma, thr, cooldown=cfg["detect"]["cooldown"])

    # bitrate（bit/sample），MVP 近似为 oversample × 1
    bitrate = cfg["dm"]["oversample"] * 1.0

    # metrics
    prec, rec, f1, avg_delay = metrics_from_alerts(alerts, gts, tol=30)

    # figures
    out_dir = Path("data/figs"); out_dir.mkdir(parents=True, exist_ok=True)
    ts_path = str(out_dir/"ts.png"); feat_path = str(out_dir/"feat.png"); hist_path = str(out_dir/"hist.png")
    plot_ts(x, xhat, resid, overload_ma, alerts, gts, ts_path)
    plot_features(overload_ma, flip_ma, feat_path)
    rls = run_lengths(s)
    import numpy as np
    plot_hist(rls, overload_ma, hist_path)

    # report
    html_t = Template(open("pipeline/report_dmguard.html","r",encoding="utf-8").read())
    html = html_t.render(
        title="DeltaMod-Guard 报告",
        scenario_name=cfg["name"],
        data_desc=f"synthetic n={n}",
        delta0=cfg["dm"]["delta0"], gamma=cfg["dm"]["gamma"], theta_k=cfg["dm"]["theta_k"], oversample=cfg["dm"]["oversample"],
        f1=round(f1,4), prec=round(prec,4), rec=round(rec,4),
        avg_delay="%.1f"%avg_delay if avg_delay==avg_delay else "NA",
        bitrate=round(bitrate,3),
        ts_path=ts_path, feat_path=feat_path, hist_path=hist_path,
        conclusion_1=f"在当前配置下，F1={round(f1,4)}，平均检测延迟约 {('%.1f'%avg_delay) if avg_delay==avg_delay else 'NA'} 步；可通过增大 γ 或减少 θₖ 提高召回。",
        conclusion_2=f"码率约 {round(bitrate,3)} bit/样本；若带宽受限可考虑 oversample<1（下采样+反混叠）或调整 Δ₀ 以减小重构误差。"
    )
    Path(cfg["report"]["out_html"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["report"]["out_html"]).write_text(html, encoding="utf-8")
    print("Report ->", cfg["report"]["out_html"])

if __name__ == "__main__":
    run("configs/scenario.example.yaml")
