# -*- coding: utf-8 -*-
"""
Qidashan_XGB_Kriging_Realtime_v22_full.py
Author: zhiyuan Wu
Date: 2025-12-05

在 v21 的基础上，增加 5 类 XGBoost 相关诊断图：
  1) 校准曲线（XGBoost & Fusion）
  2) XGBoost 特征重要性图
  3) XGBoost 学习曲线
  4) XGBoost 部分依赖图（PDP，按重要性自动选取若干特征）
  5) XGBoost 的 SHAP 解释图（summary + 单特征依赖）

保持原有数据流程与 Excel 列名完全一致：
  - 钻孔坐标1.xlsx  （包含 “工程号”、“开孔坐标E”、“开孔坐标N”）
  - 实验数据1.xlsx  （包含 “工程号”、“从”、“至”、“TFe”）
  - 钻孔定位1.xlsx  （结构不变，暂不直接使用）

Stage 概览：
  - Stage 1: 读取→清洗→SMOGN→WB2→PCA→标准化→切分
  - BO-Kriging：仅 spherical/exponential/gaussian
  - 分块 Kriging（OK3D/OK 兼容）
  - XGBoost 直接回归
  - Hybrid：局部残差（KMeans 分簇约束）
  - Fusion：Ridge 投影 + Meta-XGB Stacking
  - 可视化：
      · 原有：模型对比、散点、残差分布、3D 钻孔、ΔTFe、hexbin、Fusion 校准、空间切片
      · 新增：校准曲线（多模型）、重要性、学习曲线、PDP、SHAP
"""

import os, sys, time, math, threading, warnings
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings("ignore")

# sklearn
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.inspection import PartialDependenceDisplay
from sklearn.calibration import calibration_curve

# xgboost
import xgboost as xgb

# smogn（可选）
try:
    from smogn import smoter
    SMOGN_AVAILABLE = True
except Exception:
    SMOGN_AVAILABLE = False

# shap（可选）
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# skopt（BO）
from skopt import gp_minimize
from skopt.space import Real, Categorical

# ------------------- 全局配置与路径 -------------------
SEED = 42
np.random.seed(SEED)

BASE_OUTDIR = "outputs_chapter5_dynamic_v22"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = os.path.join(BASE_OUTDIR, f"run_{RUN_ID}")
os.makedirs(OUTDIR, exist_ok=True)
LOG_PATH = os.path.join(OUTDIR, "log_runtime.txt")

FAST_THRESHOLD = 5000
BLOCKS = (5, 5, 3)
DEFAULT_VARIOGRAM_MODEL = "spherical"
REFRESH_INTERVAL = 0.6
FUSION_KFOLD = 5
LOCAL_CLUSTER_K = 5
PROJECTION_ITERS = 3

# ------------------- 日志与工具 -------------------
def log_msg(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def progress_bar(step,total,elapsed,eta,rmse=None,prefix="[Stage]"):
    pct = step/total if total>0 else 0
    filled = int(30*pct)
    bar = "▓"*filled + "░"*(30-filled)
    t1 = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    t2 = time.strftime("%H:%M:%S", time.gmtime(eta))
    rm = f" | RMSE={rmse:.3f}" if rmse is not None else ""
    sys.stdout.write(f"\r{prefix} {bar} {pct*100:5.1f}% | Elapsed {t1} | ETA {t2}{rm}")
    sys.stdout.flush()

class Timer:
    def __init__(self, total):
        self.start = time.time()
        self.total = total
        self.current = 0
    def update(self, step):
        self.current = step
        elapsed = time.time() - self.start
        eta = elapsed / max(step,1) * max(self.total-step,0)
        return elapsed, eta

class SystemMonitor(threading.Thread):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared
        self.running = True
        self.cpu_hist, self.mem_hist = [], []
        self.start_time = time.time()
    def run(self):
        while self.running:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            self.cpu_hist.append(cpu); self.mem_hist.append(mem)
            st  = self.shared.get("stage","Init")
            p   = self.shared.get("progress",0)
            t   = self.shared.get("total",1)
            eta = self.shared.get("eta",0)
            rm  = self.shared.get("rmse",0)
            pct = (p/t*100) if t else 0
            eta_s = time.strftime("%H:%M:%S", time.gmtime(eta))
            sys.stdout.write(
                f"\r[Monitor] {st:<12}| {pct:5.1f}% | RMSE={rm:6.3f} | CPU={cpu:5.1f}% | Mem={mem:5.1f}% | ETA={eta_s}  "
            )
            sys.stdout.flush()
            time.sleep(1)
    def stop(self):
        self.running = False
        avg_cpu = np.mean(self.cpu_hist) if self.cpu_hist else 0
        avg_mem = np.mean(self.mem_hist) if self.mem_hist else 0
        peak_mem = np.max(self.mem_hist) if self.mem_hist else 0
        log_msg(f"[Monitor] AvgCPU={avg_cpu:.1f}%, AvgMem={avg_mem:.1f}%, PeakMem={peak_mem:.1f}%")
        log_msg("[Monitor] Stopped successfully.")

# ------------------- 数据读取与样点构建 -------------------
def read_data():
    log_msg("Reading Excel files: 钻孔坐标1.xlsx, 实验数据1.xlsx, 钻孔定位1.xlsx …")
    loc = pd.read_excel("钻孔坐标1.xlsx")   # 应包含: 工程号, 开孔坐标E, 开孔坐标N
    grd = pd.read_excel("实验数据1.xlsx")   # 应包含: 工程号, 从, 至, TFe
    dev = pd.read_excel("钻孔定位1.xlsx")   # 保留结构兼容
    return loc, grd, dev

def build_samples(loc, grd, step=2.0):
    rows=[]
    grd_by = {k: v.sort_values("从").reset_index(drop=True)
              for k,v in grd.groupby("工程号")}
    for bh, collar in tqdm(loc.set_index("工程号").iterrows(),
                           total=len(loc),
                           desc="[Building Samples]"):
        g = grd_by.get(bh)
        if g is None:
            continue
        for _, r in g.iterrows():
            d0, d1, tfe = float(r["从"]), float(r["至"]), float(r["TFe"])
            if d1 <= d0:
                continue
            n = max(1, int((d1-d0)/step))
            for z in np.linspace(d0, d1, n):
                rows.append(
                    (collar["开孔坐标E"], collar["开孔坐标N"], -z, tfe, bh)
                )
    df = pd.DataFrame(rows, columns=["x","y","depth","TFe","工程号"])
    log_msg(f"Generated {len(df)} samples.")
    return df.dropna().reset_index(drop=True)

# ------------------- 清洗/特征/平衡/标准化 -------------------
def remove_outliers(df, cols, z_th=4.0):
    keep = pd.Series([True]*len(df), index=df.index)
    for c in cols:
        vals = df[c].astype(float)
        z = (vals - vals.mean())/(vals.std()+1e-9)
        keep &= (np.abs(z) <= z_th)
        q1,q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3-q1
        low, high = q1-1.5*iqr, q3+1.5*iqr
        keep &= vals.between(low, high)
    removed = (~keep).sum()
    if removed>0:
        log_msg(f"[Clean] Removed outliers: {removed}")
    return df[keep].reset_index(drop=True)

def feature_engineering(df):
    df = df.copy()
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2)
    df["xy"] = df["x"]*df["y"]
    df["depth2"] = df["depth"]**2
    df["r2"] = df["r"]**2
    dmin,dmax = df["depth"].min(), df["depth"].max()
    df["depth_norm"] = (df["depth"]-dmin)/max(dmax-dmin,1e-9)
    return df

def smogn_preprocessing(df, target="TFe"):
    if SMOGN_AVAILABLE:
        smogn_df = smoter(df, target)
        log_msg("SMOGN preprocessing done. Synthetic samples generated.")
        return smogn_df.reset_index(drop=True)
    # fallback：对上下 2% 极值段做少量过采样
    k = 0.02
    ql, qh = df[target].quantile(k), df[target].quantile(1-k)
    ext = pd.concat([df[df[target]<=ql], df[df[target]>=qh]], axis=0)
    if len(ext) > 0:
        aug = ext.sample(min(len(ext), 1000), replace=True, random_state=SEED)
        df2 = pd.concat([df, aug], ignore_index=True)
        log_msg("SMOGN not available. Applied simple extreme oversampling (fallback).")
        return df2.reset_index(drop=True)
    log_msg("SMOGN not available and no extremes found. Skipped.")
    return df.reset_index(drop=True)

def wb2_positioning(df, target="TFe", k=0.02, jitter=0.01):
    df = df.copy()
    q_low, q_high = df[target].quantile(k), df[target].quantile(1-k)
    low_ext = df[df[target] <= q_low]
    high_ext = df[df[target] >= q_high]
    s = df[target].std()
    def jitter_df(part):
        if part.empty:
            return part
        J = part.copy()
        noise = np.random.normal(0, max(s*jitter,1e-6), size=len(J))
        J[target] = J[target] + noise
        return J
    aug = pd.concat([jitter_df(low_ext), jitter_df(high_ext)], ignore_index=True)
    out = pd.concat([df, aug], ignore_index=True)
    log_msg(f"[WB2] Added {len(aug)} jittered extreme samples (k={k}).")
    return out.reset_index(drop=True)

def apply_pca_safe(df, n_components=5, exclude=("TFe","工程号")):
    use_cols = [c for c in df.columns
                if (c not in exclude) and (df[c].dtype.kind in "fc")]
    n_features = len(use_cols)
    if n_features == 0:
        return df, None
    nc = max(1, min(n_components, n_features))
    pca = PCA(n_components=nc, random_state=SEED)
    Z = pca.fit_transform(df[use_cols].values)
    pca_df = pd.DataFrame(
        Z,
        columns=[f"PC{i+1}" for i in range(nc)],
        index=df.index
    )
    df = pd.concat([df, pca_df], axis=1)
    log_msg(f"[PCA] Added {nc} PCs (from {n_features} numeric features).")
    return df, pca

def scale_features(df, feature_cols):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    sc_df = pd.DataFrame(
        X,
        columns=[f"{c}_sc" for c in feature_cols],
        index=df.index
    )
    log_msg(f"[Scale] Standardized {len(feature_cols)} features.")
    return pd.concat([df, sc_df], axis=1), scaler

# ------------------- Kriging -------------------
def partition_blocks(df, nblocks=BLOCKS):
    x = np.linspace(df["x"].min(), df["x"].max(), nblocks[0]+1)
    y = np.linspace(df["y"].min(), df["y"].max(), nblocks[1]+1)
    z = np.linspace(df["depth"].min(), df["depth"].max(), nblocks[2]+1)
    blocks=[]
    for i in range(nblocks[0]):
        for j in range(nblocks[1]):
            for k in range(nblocks[2]):
                cond=(df["x"].between(x[i],x[i+1]) &
                      df["y"].between(y[j],y[j+1]) &
                      df["depth"].between(z[k],z[k+1]))
                blk=df[cond]
                if len(blk)>0:
                    blocks.append(blk)
    log_msg(f"Partitioned into {len(blocks)} blocks (non-empty).")
    return blocks

def run_ok3d(train, grid, model="spherical", variogram_parameters=None):
    try:
        from pykrige.ok3d import OrdinaryKriging3D
        ok=OrdinaryKriging3D(
            train["x"],train["y"],train["depth"],train["TFe"],
            variogram_model=model, variogram_parameters=variogram_parameters,
            enable_plotting=False, verbose=False
        )
        z,_=ok.execute("points",grid["x"],grid["y"],grid["depth"])
        return np.array(z)
    except Exception:
        from pykrige.ok import OrdinaryKriging
        ok=OrdinaryKriging(
            train["x"],train["y"],train["TFe"],
            variogram_model=model, variogram_parameters=variogram_parameters,
            enable_plotting=False, verbose=False
        )
        z,_=ok.execute("points",grid["x"],grid["y"])
        return np.array(z)

def evaluate_rmse(true, pred):
    return math.sqrt(mean_squared_error(true, pred))

def bayes_optimize_kriging(train_df, n_calls=30, fast_mode=True, random_state=SEED):
    models = ["spherical","exponential","gaussian"]
    space = [
        Categorical(models, name="model"),
        Real(10.0, 500.0, name="range"),
        Real(0.1, 10.0, name="sill"),
        Real(0.0, 5.0,  name="nugget"),
    ]
    sub = train_df.sample(
        n=min(3000,len(train_df)),
        random_state=random_state
    ) if fast_mode else train_df
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    def objective(params):
        model, rng, sill, nug = params
        vp = {"range":rng, "sill":sill, "nugget":nug}
        rmses=[]
        for tr_idx, val_idx in kf.split(sub):
            tr=sub.iloc[tr_idx]; va=sub.iloc[val_idx]
            try:
                pred = run_ok3d(tr, va, model=model, variogram_parameters=vp)
                rmse = evaluate_rmse(va["TFe"].values, pred)
            except Exception:
                rmse = 1e6
            rmses.append(rmse)
        return float(np.mean(rmses))

    res = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=random_state,
        n_initial_points=8,
        acq_func="EI"
    )
    all_scores = res.func_vals
    stats = {
        "best_rmse": float(np.min(all_scores)),
        "worst_rmse": float(np.max(all_scores)),
        "mean_rmse": float(np.mean(all_scores)),
        "median_rmse": float(np.median(all_scores)),
        "std_rmse": float(np.std(all_scores)),
        "evals_total": int(len(all_scores)),
        "evals_to_best": int(np.argmin(all_scores))+1,
        "best_params": {
            "model": res.x[0],
            "range": res.x[1],
            "sill":  res.x[2],
            "nugget":res.x[3]
        }
    }
    log_msg(f"[BO-Kriging] Stats: {stats}")
    pd.DataFrame({"rmse":all_scores}).to_csv(
        os.path.join(OUTDIR,f"bo_kriging_trials_{RUN_ID}.csv"),
        index=False
    )
    with open(os.path.join(OUTDIR, f"bo_kriging_best_{RUN_ID}.txt"),
              "w", encoding="utf-8") as f:
        f.write(str(stats))
    return stats

def kriging_realtime(train_df, test_df, shared, model, variogram_params, fast_mode=True):
    blocks = partition_blocks(test_df, BLOCKS)
    timer=Timer(len(blocks)); rmse_hist=[]; preds=[]
    plt.ion()
    fig,ax1=plt.subplots(figsize=(8,5)); ax2=ax1.twinx()
    for i,blk in enumerate(blocks):
        elapsed,eta=timer.update(i+1)
        shared.update(stage="Kriging",progress=i+1,total=len(blocks),eta=eta)
        subset=train_df.sample(
            n=min(3000,len(train_df))
        ) if fast_mode else train_df
        try:
            pred = run_ok3d(subset, blk, model=model, variogram_parameters=variogram_params)
            blk = blk.copy(); blk["Kriging_pred"]=pred; preds.append(blk)
        except Exception as e:
            log_msg(f"[ERROR] Kriging block {i+1}: {e}")
            continue
        rmse=evaluate_rmse(blk["TFe"].values, blk["Kriging_pred"].values)
        rmse_hist.append(rmse); shared.update(rmse=rmse)
        progress_bar(i+1,len(blocks),elapsed,eta,rmse,prefix="[Kriging]")
        ax1.clear(); ax2.clear()
        ax1.plot(rmse_hist, lw=1.6, label="RMSE")
        ax2.plot(rmse_hist, lw=1.0, color="tab:red")
        ax1.set_xlabel("Block"); ax1.set_ylabel("RMSE"); ax2.set_ylabel("Track")
        ax1.set_title(f"[Kriging] {i+1}/{len(blocks)} | RMSE={rmse:.3f}")
        plt.pause(REFRESH_INTERVAL)
    plt.ioff()
    fig.savefig(os.path.join(OUTDIR,f"fig_kriging_progress_{RUN_ID}.png"), dpi=400)
    plt.close(fig)
    result=pd.concat(preds, ignore_index=True)
    log_msg(f"Kriging interpolation finished. {len(result)} predictions.")
    return result

# ------------------- Stage 1 总流程 -------------------
def stage1_prepare(shared):
    log_msg("Stage 1: Data reading & preprocessing (v22)")
    loc,grd,dev = read_data()
    df = build_samples(loc,grd)
    df = remove_outliers(df, cols=["x","y","depth","TFe"])
    df = feature_engineering(df)
    df = smogn_preprocessing(df, target="TFe")
    df = wb2_positioning(df, target="TFe", k=0.02, jitter=0.01)
    df, _ = apply_pca_safe(df, n_components=5, exclude=("TFe","工程号"))

    base_feats=["x","y","depth","r","xy","depth2","r2","depth_norm"]
    pca_feats=[c for c in df.columns if c.startswith("PC")]
    df, scaler = scale_features(df, base_feats+pca_feats)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=SEED)
    fast = len(df) > FAST_THRESHOLD
    log_msg(f"FAST_MODE={fast}")

    shared.update(stage="BO-Kriging", progress=0, total=1)
    bo_stats = bayes_optimize_kriging(train_df, n_calls=50, fast_mode=fast, random_state=SEED)
    model = bo_stats["best_params"]["model"]
    variogram_params = {
        "range":  bo_stats["best_params"]["range"],
        "sill":   bo_stats["best_params"]["sill"],
        "nugget": bo_stats["best_params"]["nugget"]
    }

    preds = kriging_realtime(train_df, test_df, shared,
                             model=model,
                             variogram_params=variogram_params,
                             fast_mode=fast)

    if "Kriging_pred" not in train_df.columns:
        train_df = train_df.copy()
        train_df["Kriging_pred"] = train_df["TFe"].rolling(
            window=3,
            min_periods=1
        ).mean()
        log_msg("Filled train_df['Kriging_pred'] by rolling mean (proxy).")

    ml_feats_all = base_feats + pca_feats
    return train_df, preds, bo_stats, ml_feats_all

# ------------------- XGB & Hybrid（带局部空间约束） -------------------
def xgb_direct_predict(train_df, test_df, features_sc, shared):
    Xtr = train_df[[f"{c}_sc" for c in features_sc if f"{c}_sc" in train_df.columns]]
    Xte = test_df[[f"{c}_sc" for c in features_sc if f"{c}_sc" in test_df.columns]]
    ytr = train_df["TFe"].values

    param_grid = {
        "n_estimators":[200,300,400],
        "max_depth":[6,8,10],
        "learning_rate":[0.05,0.1,0.2],
        "subsample":[0.7,0.9],
        "colsample_bytree":[0.8,1.0]
    }
    combos=[(n,d,lr,s,cs)
            for n in param_grid["n_estimators"]
            for d in param_grid["max_depth"]
            for lr in param_grid["learning_rate"]
            for s in param_grid["subsample"]
            for cs in param_grid["colsample_bytree"]]
    rmse_vals=[]; timer=Timer(len(combos)); best_rm=1e9; best_model=None
    plt.ion(); fig,ax=plt.subplots(figsize=(8,5))
    for i,(n,d,lr,s,cs) in enumerate(combos):
        elapsed,eta=timer.update(i+1)
        shared.update(stage="XGBoost",progress=i+1,total=len(combos),eta=eta)
        model=xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n,
            max_depth=d,
            learning_rate=lr,
            subsample=s,
            colsample_bytree=cs,
            reg_lambda=1.0,
            reg_alpha=0.2,
            random_state=SEED
        )
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        rm = np.sqrt(np.mean((test_df["TFe"].values - preds)**2))
        rmse_vals.append(rm)
        if rm < best_rm:
            best_rm=rm; best_model=model
        progress_bar(i+1,len(combos),elapsed,eta,rm,prefix="[XGBoost]")
        ax.clear()
        ax.plot(rmse_vals, lw=1.5)
        ax.set_xlabel("Combo")
        ax.set_ylabel("RMSE")
        plt.pause(0.05)
    plt.ioff()
    fig.savefig(os.path.join(OUTDIR,f"fig_xgb_search_{RUN_ID}.png"),dpi=400)
    plt.close(fig)
    te_pred = best_model.predict(Xte)
    return te_pred, best_model

def hybrid_residual_with_local(train_df, test_df, features_sc):
    Xtr_full = train_df[[f"{c}_sc" for c in features_sc if f"{c}_sc" in train_df.columns]]
    ytr_res  = (train_df["TFe"] - train_df["Kriging_pred"]).values

    coords_tr = train_df[["x","y","depth"]].values
    kmeans = KMeans(n_clusters=LOCAL_CLUSTER_K, random_state=SEED, n_init=10)
    clu_tr = kmeans.fit_predict(coords_tr)

    models = {}
    for c in range(LOCAL_CLUSTER_K):
        idx = np.where(clu_tr==c)[0]
        if len(idx)<50:
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=SEED
            )
            model.fit(Xtr_full, ytr_res)
            models[c]=model
            continue
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED
        )
        model.fit(Xtr_full.iloc[idx], ytr_res[idx])
        models[c]=model

    coords_te = test_df[["x","y","depth"]].values
    clu_te = kmeans.predict(coords_te)
    Xte_full = test_df[[f"{c}_sc" for c in features_sc if f"{c}_sc" in test_df.columns]]
    res_pred = np.zeros(len(test_df))
    for c in range(LOCAL_CLUSTER_K):
        idx = np.where(clu_te==c)[0]
        if len(idx)==0:
            continue
        res_pred[idx] = models[c].predict(Xte_full.iloc[idx])
    hybrid_pred = test_df["Kriging_pred"].values + res_pred
    return hybrid_pred, kmeans, models

# ------------------- Fusion（增强：投影 + Stacking） -------------------
def _project_simplex_nonneg(w):
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        return np.array([1.0/len(w)]*len(w))
    return w / s

def fusion_enhanced(test_df, use_meta_xgb=True, kfold=FUSION_KFOLD):
    y = test_df["TFe"].values
    P = np.vstack([
        test_df["Kriging_pred"].values,
        test_df["XGBoost_pred"].values,
        test_df["Hybrid_pred"].values
    ]).T

    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(P, y)
    w = ridge.coef_.astype(float)
    b = float(ridge.intercept_)
    for _ in range(PROJECTION_ITERS):
        w = _project_simplex_nonneg(w)
    fusion_pred_ridge = P.dot(w) + b
    weights = {"Kriging": w[0], "XGBoost": w[1], "Hybrid": w[2], "Intercept": b}
    log_msg(f"[Fusion] Ridge weights (projected): {weights}")

    if not use_meta_xgb:
        return fusion_pred_ridge, None, weights

    kf = KFold(n_splits=kfold, shuffle=True, random_state=SEED)
    oof_meta_X = np.zeros((len(test_df), 4))
    oof_meta_y = y.copy()
    for fold,(tr_idx, va_idx) in enumerate(kf.split(P), start=1):
        ridge_f = Ridge(alpha=1.0, random_state=SEED)
        ridge_f.fit(P[tr_idx], y[tr_idx])
        wf = _project_simplex_nonneg(ridge_f.coef_.astype(float))
        bf = float(ridge_f.intercept_)
        ridge_va = P[va_idx].dot(wf) + bf
        oof_meta_X[va_idx,0] = ridge_va
        oof_meta_X[va_idx,1:] = P[va_idx]

    meta = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=SEED
    )
    meta.fit(oof_meta_X, oof_meta_y)

    meta_X_full = np.column_stack([fusion_pred_ridge, P])
    fusion_pred_meta = meta.predict(meta_X_full)
    log_msg("[Fusion] Meta-XGB stacked fusion trained.")
    return fusion_pred_ridge, fusion_pred_meta, weights

# ------------------- 评估与基础可视化 -------------------
def evaluate_all(y_true,y_pred,name="Model"):
    mae=float(mean_absolute_error(y_true,y_pred))
    rmse=float(np.sqrt(mean_squared_error(y_true,y_pred)))
    r2=float(r2_score(y_true,y_pred))
    evs=float(explained_variance_score(y_true,y_pred))
    mape=float(np.mean(np.abs((y_true - y_pred)/np.maximum(np.abs(y_true),1e-9)))*100)
    bias=float(np.mean(y_pred - y_true))
    log_msg(f"[{name}] MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}, EVS={evs:.3f}, MAPE={mape:.2f}%, Bias={bias:.3f}")
    return {"MAE":mae,"RMSE":rmse,"R²":r2,"EVS":evs,"MAPE":mape,"Bias":bias}

def plot_model_comparison_bars(metrics, save_path):
    labels = list(metrics.keys())
    rmse = [metrics[m]["RMSE"] for m in labels]
    r2   = [metrics[m]["R²"] for m in labels]
    fig,ax1 = plt.subplots(figsize=(8.2,5.2))
    ax2=ax1.twinx()
    bars=ax1.bar(labels, rmse, alpha=0.72, label="RMSE")
    ax2.plot(labels, r2, "o-", lw=2, color="tab:red", label="R²")
    ax1.set_ylabel("RMSE")
    ax2.set_ylabel("R²")
    ax1.grid(alpha=0.25)
    for b,val in zip(bars, rmse):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for x,val in zip(labels, r2):
        ax2.text(x, val+0.002, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9, color="tab:red")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=400)
    plt.close(fig)

def plot_scatter_with_r2(df, save_path, include_fusion=True):
    plt.figure(figsize=(6.8,6.8))
    plt.scatter(df["TFe"],df["Kriging_pred"],s=10,alpha=0.55,
                label=f"Kriging (R²={r2_score(df['TFe'],df['Kriging_pred']):.3f})")
    plt.scatter(df["TFe"],df["XGBoost_pred"],s=10,alpha=0.55,
                label=f"XGBoost (R²={r2_score(df['TFe'],df['XGBoost_pred']):.3f})")
    plt.scatter(df["TFe"],df["Hybrid_pred"],s=10,alpha=0.70,
                label=f"Hybrid (R²={r2_score(df['TFe'],df['Hybrid_pred']):.3f})")
    if include_fusion and "Fusion_pred" in df.columns:
        plt.scatter(df["TFe"],df["Fusion_pred"],s=12,alpha=0.8,
                    label=f"Fusion (R²={r2_score(df['TFe'],df['Fusion_pred']):.3f})")
        lims=[min(df["TFe"].min(),
                  df[["Kriging_pred","XGBoost_pred","Hybrid_pred","Fusion_pred"]].min().min())-1,
              max(df["TFe"].max(),
                  df[["Kriging_pred","XGBoost_pred","Hybrid_pred","Fusion_pred"]].max().max())+1]
    else:
        lims=[min(df["TFe"].min(),
                  df[["Kriging_pred","XGBoost_pred","Hybrid_pred"]].min().min())-1,
              max(df["TFe"].max(),
                  df[["Kriging_pred","XGBoost_pred","Hybrid_pred"]].max().max())+1]
    plt.plot(lims,lims,"k--",lw=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Actual TFe (%)")
    plt.ylabel("Predicted TFe (%)")
    plt.legend(frameon=False,loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

def plot_residual_hist_compare(df, save_path):
    res_k = df["Kriging_pred"] - df["TFe"]
    res_x = df["XGBoost_pred"] - df["TFe"]
    res_h = df["Hybrid_pred"]   - df["TFe"]
    plt.figure(figsize=(8.8,5.3)); bins=40
    plt.hist(res_k, bins=bins, alpha=0.35,
             label=f"Kriging (μ={res_k.mean():.2f}, σ={res_k.std():.2f})")
    plt.hist(res_x, bins=bins, alpha=0.35,
             label=f"XGBoost (μ={res_x.mean():.2f}, σ={res_x.std():.2f})")
    plt.hist(res_h, bins=bins, alpha=0.50,
             label=f"Hybrid (μ={res_h.mean():.2f}, σ={res_h.std():.2f})")
    if "Fusion_pred" in df.columns:
        res_f = df["Fusion_pred"] - df["TFe"]
        plt.hist(res_f, bins=bins, alpha=0.5,
                 label=f"Fusion (μ={res_f.mean():.2f}, σ={res_f.std():.2f})")
    plt.axvline(0,color="k",lw=1)
    plt.xlabel("Residual (Pred - Actual)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution Comparison")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

def plot_3d_drillhole(train_df, test_df, save_path):
    fig = plt.figure(figsize=(8.6,6.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(train_df["x"], train_df["y"], train_df["depth"],
               c="tab:blue", s=6, alpha=0.45, label="Train")
    ax.scatter(test_df["x"],  test_df["y"],  test_df["depth"],
               c="tab:red",  s=10, alpha=0.75, label="Test")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.legend(loc="best")
    ax.invert_zaxis()
    plt.tight_layout()
    fig.savefig(save_path, dpi=400)
    plt.close(fig)

def plot_delta_tfe(df, save_path_prefix):
    if "Fusion_pred" not in df.columns:
        return
    delta = df["Fusion_pred"] - df["XGBoost_pred"]
    plt.figure(figsize=(7.2,4.2))
    plt.hist(delta, bins=40, alpha=0.8, edgecolor="black")
    plt.xlabel("ΔTFe = Fusion – XGB")
    plt.ylabel("Count")
    plt.title("ΔTFe Distribution")
    plt.tight_layout()
    plt.savefig(save_path_prefix + "_hist.png", dpi=400)
    plt.close()

    plt.figure(figsize=(6.6,5.6))
    norm = TwoSlopeNorm(vcenter=0.0)
    sc = plt.scatter(df["x"], df["y"], c=delta,
                     s=8, alpha=0.8, cmap="coolwarm", norm=norm)
    plt.colorbar(sc, label="ΔTFe")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Spatial ΔTFe (Fusion – XGB)")
    plt.tight_layout()
    plt.savefig(save_path_prefix + "_map.png", dpi=400)
    plt.close()

def plot_error_hexbin(df, col_pred, save_path):
    err = df[col_pred] - df["TFe"]
    plt.figure(figsize=(6.6,5.6))
    hb = plt.hexbin(df["x"], df["y"], C=err,
                    gridsize=35, cmap="coolwarm",
                    reduce_C_function=np.mean)
    plt.colorbar(hb, label="Mean Error")
    plt.title(f"Spatial Mean Error (model={col_pred})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

def plot_calibration_single(df, col_pred, save_path, bins=15):
    df2 = df[[col_pred,"TFe"]].copy()
    df2["bin"] = pd.qcut(df2[col_pred], q=bins, duplicates="drop")
    grp = df2.groupby("bin").agg(pred_mean=(col_pred,"mean"),
                                 true_mean=("TFe","mean"))
    plt.figure(figsize=(5.8,5.2))
    plt.plot(grp["pred_mean"], grp["true_mean"], "o-", lw=1.8)
    lim = [min(grp.min().min(), df["TFe"].min())-1,
           max(grp.max().max(), df["TFe"].max())+1]
    plt.plot(lim, lim, "k--", lw=1)
    plt.xlabel("Predicted mean")
    plt.ylabel("True mean")
    plt.title(f"Calibration ({col_pred})")
    plt.xlim(lim); plt.ylim(lim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

def plot_depth_slice(df, col_pred, save_path, slices=4):
    zs = np.linspace(df["depth"].min(), df["depth"].max(), slices+2)[1:-1]
    fig, axes = plt.subplots(1, slices, figsize=(4*slices, 4), sharex=False, sharey=False)
    if slices==1:
        axes=[axes]
    for ax, zc in zip(axes, zs):
        band = df[np.abs(df["depth"]-zc) <= (df["depth"].std()*0.1 + 1e-6)]
        sc = ax.scatter(band["x"], band["y"],
                        c=band[col_pred]-band["TFe"],
                        s=7, cmap="coolwarm", vmin=-10, vmax=10)
        ax.set_title(f"Depth≈{zc:.1f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    fig.colorbar(sc, ax=axes, shrink=0.85, label="Pred-True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close(fig)

# ------------- 新增五类图：校准曲线 / 重要性 / 学习曲线 / PDP / SHAP -------------
def plot_calibration_multi(df, cols_pred, save_path, bins=15):
    """
    多模型回归“校准曲线”：将预测值分位数分箱，绘制 (pred_mean, true_mean)。
    cols_pred 示例：["XGBoost_pred", "Fusion_pred"]
    """
    plt.figure(figsize=(6.2,5.4))
    for col in cols_pred:
        df2 = df[[col,"TFe"]].copy()
        df2["bin"] = pd.qcut(df2[col], q=bins, duplicates="drop")
        grp = df2.groupby("bin").agg(pred_mean=(col,"mean"),
                                     true_mean=("TFe","mean"))
        plt.plot(grp["pred_mean"], grp["true_mean"], "o-", lw=1.4, label=col)
    lim = [min(df["TFe"].min(), df[cols_pred].min().min())-1,
           max(df["TFe"].max(), df[cols_pred].max().max())+1]
    plt.plot(lim, lim, "k--", lw=1)
    plt.xlabel("Predicted mean grade")
    plt.ylabel("Observed mean grade")
    plt.xlim(lim); plt.ylim(lim)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

def plot_xgb_feature_importance(xgb_model, feature_names, save_path, top_n=20):
    importance = xgb_model.feature_importances_
    idx = np.argsort(importance)[::-1]
    idx = idx[:min(top_n, len(idx))]
    sorted_imp = importance[idx]
    sorted_names = [feature_names[i] for i in idx]

    plt.figure(figsize=(7.0, 0.35*len(idx)+1.5))
    y_pos = np.arange(len(idx))
    plt.barh(y_pos, sorted_imp, alpha=0.8)
    plt.yticks(y_pos, sorted_names)
    plt.xlabel("Feature importance (gain-based)")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

def plot_xgb_learning_curve(xgb_model, X_train, y_train, save_path):
    """
    使用 sklearn.learning_curve 对 XGBoost 进行学习曲线分析。
    """
    train_sizes, train_scores, val_scores = learning_curve(
        xgb_model,
        X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        shuffle=True,
        random_state=SEED
    )
    train_rmse = -np.mean(train_scores, axis=1)
    val_rmse   = -np.mean(val_scores, axis=1)

    plt.figure(figsize=(6.4,4.8))
    plt.plot(train_sizes, train_rmse, "o-", lw=1.8, label="Training RMSE")
    plt.plot(train_sizes, val_rmse,   "s--", lw=1.8, label="Validation RMSE")
    plt.xlabel("Number of training samples")
    plt.ylabel("RMSE")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

def plot_xgb_pdp(xgb_model, X, feature_names, save_path, top_n=4):
    """
    按特征重要性自动选取若干特征绘制 PDP。
    """
    importance = xgb_model.feature_importances_
    idx = np.argsort(importance)[::-1]
    idx = idx[:min(top_n, len(idx))]

    fig, ax = plt.subplots(
        nrows=2, ncols=int(np.ceil(len(idx)/2)),
        figsize=(4*int(np.ceil(len(idx)/2)), 6.0)
    )
    ax = np.array(ax).reshape(-1)
    for i,(fid, axis) in enumerate(zip(idx, ax)):
        PartialDependenceDisplay.from_estimator(
            xgb_model,
            X,
            [fid],
            feature_names=feature_names,
            ax=axis
        )
        axis.set_title(feature_names[fid])
    for j in range(i+1, len(ax)):
        ax[j].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

def plot_xgb_shap(xgb_model, X, feature_names, out_prefix):
    """
    SHAP summary plot + 单特征依赖图（对若干最重要特征）。
    若环境中未安装 shap，则跳过并写日志。
    """
    if not SHAP_AVAILABLE:
        log_msg("[SHAP] shap package not available, skip SHAP plots.")
        return

    # 在脚本环境下不强制 initjs，避免 AssertionError
    try:
        import shap
        shap.initjs()
    except Exception as e:
        log_msg(f"[SHAP] initjs failed or not needed in this environment: {e}")

    # 随机抽样以控制计算量
    n_sample = min(4000, X.shape[0])
    idx = np.random.choice(X.shape[0], size=n_sample, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # summary plot
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(out_prefix + "_shap_summary.png", dpi=400, bbox_inches="tight")
    plt.close()

    # 对前若干重要特征做 dependence plot
    importance = xgb_model.feature_importances_
    order = np.argsort(importance)[::-1]
    top_m = min(4, len(order))
    for i in range(top_m):
        fid = order[i]
        plt.figure()
        shap.dependence_plot(
            fid,
            shap_values, X_sample,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(
            out_prefix + f"_shap_dep_{feature_names[fid]}.png",
            dpi=400, bbox_inches="tight"
        )
        plt.close()

# ------------------- 主流程 -------------------
def main():
    shared = {"stage":"Init","progress":0,"total":1,"eta":0,"rmse":0.0}
    monitor = SystemMonitor(shared); monitor.start()
    start_time = time.time()
    try:
        # Stage 1: 数据 & Kriging
        train_df, test_df, bo_stats, ml_feats_all = stage1_prepare(shared)

        # 保存 Kriging 结果
        test_df.to_csv(
            os.path.join(OUTDIR, f"kriging_predictions_{RUN_ID}.csv"),
            index=False
        )

        # XGB
        xgb_pred, xgb_model = xgb_direct_predict(train_df, test_df, ml_feats_all, shared)
        test_df["XGBoost_pred"] = xgb_pred

        # Hybrid
        hybrid_pred, kmeans, res_models = hybrid_residual_with_local(
            train_df, test_df, ml_feats_all
        )
        test_df["Hybrid_pred"] = hybrid_pred

        # Fusion
        fusion_ridge, fusion_meta, weights = fusion_enhanced(
            test_df, use_meta_xgb=True, kfold=FUSION_KFOLD
        )
        test_df["Fusion_pred"] = fusion_meta if fusion_meta is not None else fusion_ridge

        # 评估
        metrics_kriging = evaluate_all(
            test_df["TFe"].values, test_df["Kriging_pred"].values,
            "Baseline Kriging"
        )
        metrics_xgb     = evaluate_all(
            test_df["TFe"].values, test_df["XGBoost_pred"].values,
            "Pure XGBoost"
        )
        metrics_hybrid  = evaluate_all(
            test_df["TFe"].values, test_df["Hybrid_pred"].values,
            "Hybrid XGB+Kriging"
        )
        metrics_fusion  = evaluate_all(
            test_df["TFe"].values, test_df["Fusion_pred"].values,
            "Fusion (Ridge+MetaXGB)"
        )

        # 基础可视化
        shared.update(stage="Visualization", progress=0, total=1)
        plot_model_comparison_bars(
            {"Kriging":metrics_kriging,
             "XGBoost":metrics_xgb,
             "Hybrid":metrics_hybrid,
             "Fusion":metrics_fusion},
            os.path.join(OUTDIR, f"fig_model_compare_{RUN_ID}.png")
        )
        plot_scatter_with_r2(
            test_df,
            os.path.join(OUTDIR, f"fig_scatter_compare_{RUN_ID}.png")
        )
        plot_residual_hist_compare(
            test_df,
            os.path.join(OUTDIR, f"fig_residual_compare_{RUN_ID}.png")
        )
        plot_3d_drillhole(
            train_df, test_df,
            os.path.join(OUTDIR, f"fig_3d_train_test_{RUN_ID}.png")
        )
        plot_delta_tfe(
            test_df,
            os.path.join(OUTDIR, f"fig_delta_tfe_{RUN_ID}")
        )
        plot_error_hexbin(
            test_df, "Fusion_pred",
            os.path.join(OUTDIR, f"fig_hexbin_fusion_{RUN_ID}.png")
        )
        plot_calibration_single(
            test_df, "Fusion_pred",
            os.path.join(OUTDIR, f"fig_calibration_fusion_{RUN_ID}.png")
        )
        plot_depth_slice(
            test_df, "Fusion_pred",
            os.path.join(OUTDIR, f"fig_depth_slice_fusion_{RUN_ID}.png"),
            slices=4
        )

        # ---------- 新增 5 类 XGBoost 诊断图 ----------
        # 统一特征矩阵（使用 _sc 特征）
        feat_sc_names = [f"{c}_sc" for c in ml_feats_all
                         if f"{c}_sc" in train_df.columns]
        X_train = train_df[feat_sc_names].values
        y_train = train_df["TFe"].values
        X_test  = test_df[feat_sc_names].values
        y_test  = test_df["TFe"].values  # 如需使用

        # 1) 多模型校准曲线（XGBoost & Fusion）
        plot_calibration_multi(
            test_df,
            cols_pred=["XGBoost_pred","Fusion_pred"],
            save_path=os.path.join(OUTDIR, f"fig_calibration_multi_{RUN_ID}.png"),
            bins=15
        )

        # 2) XGBoost 特征重要性
        plot_xgb_feature_importance(
            xgb_model,
            feature_names=feat_sc_names,
            save_path=os.path.join(OUTDIR, f"fig_xgb_feature_importance_{RUN_ID}.png"),
            top_n=20
        )

        # 3) XGBoost 学习曲线（使用同一模型配置重新拟合）
        # 为避免重复调参，这里用当前 best_model 的参数重新构造一个同构模型
        # 假设前面已经有 xgb_model = XGBRegressor(...) 并且 fit 过

        # 1. 取出当前 XGBoost 的参数
        xgb_params = xgb_model.get_xgb_params()

        # 2. 删掉其中的 objective，避免重复传参
        xgb_params.pop("objective", None)

        # 3. 用“干净”的参数重新构造一个用于学习曲线的模型
        xgb_for_lc = xgb.XGBRegressor(
            **xgb_params,
            objective="reg:squarederror",  # 这里保留一处即可
        )
        plot_xgb_learning_curve(
            xgb_for_lc,
            X_train, y_train,
            save_path=os.path.join(OUTDIR, f"fig_xgb_learning_curve_{RUN_ID}.png")
        )

        # 4) PDP（按重要性自动选取）
        plot_xgb_pdp(
            xgb_model,
            X_test,
            feature_names=feat_sc_names,
            save_path=os.path.join(OUTDIR, f"fig_xgb_pdp_{RUN_ID}.png"),
            top_n=4
        )

        # 5) SHAP（summary + dependence）
        plot_xgb_shap(
            xgb_model,
            X_test,
            feature_names=feat_sc_names,
            out_prefix=os.path.join(OUTDIR, f"fig_xgb")
        )

        # 汇总导出
        summary = pd.DataFrame({
            "Model":["Kriging","XGBoost","Hybrid","Fusion"],
            "MAE":[metrics_kriging["MAE"], metrics_xgb["MAE"],
                   metrics_hybrid["MAE"], metrics_fusion["MAE"]],
            "RMSE":[metrics_kriging["RMSE"], metrics_xgb["RMSE"],
                    metrics_hybrid["RMSE"], metrics_fusion["RMSE"]],
            "R²":[metrics_kriging["R²"], metrics_xgb["R²"],
                  metrics_hybrid["R²"], metrics_fusion["R²"]],
            "EVS":[metrics_kriging["EVS"], metrics_xgb["EVS"],
                   metrics_hybrid["EVS"], metrics_fusion["EVS"]],
            "MAPE(%)":[metrics_kriging["MAPE"], metrics_xgb["MAPE"],
                       metrics_hybrid["MAPE"], metrics_fusion["MAPE"]],
            "Bias":[metrics_kriging["Bias"], metrics_xgb["Bias"],
                    metrics_hybrid["Bias"], metrics_fusion["Bias"]],
        })
        summary.to_excel(
            os.path.join(OUTDIR, f"summary_metrics_{RUN_ID}.xlsx"),
            index=False
        )

        with open(os.path.join(OUTDIR, f"fusion_weights_{RUN_ID}.txt"),
                  "w", encoding="utf-8") as f:
            f.write(str(weights))

        total_time = time.time() - start_time
        log_msg(f"✅ All stages completed. Total time: {total_time/60:.2f} min")
        log_msg(f"[SUCCESS] Outputs saved in {OUTDIR}")

    except Exception as e:
        log_msg(f"[FATAL ERROR] {e}")
        raise
    finally:
        monitor.stop()

if __name__ == "__main__":
    log_msg("========== New Run Started (v22) ==========")
    log_msg(f"Outputs → {OUTDIR}")
    main()
