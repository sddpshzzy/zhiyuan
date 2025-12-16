# -*- coding: utf-8 -*-
"""
Qidashan_Vis_TrainTest_Sections_Final_v2.py

基于 Qidashan_XGB_Kriging_Realtime_v21_full.py 中的相同数据与预处理逻辑，
仅做可视化（train/test 剖面 + 深度切片），不影响原始建模脚本。

本版包含：
  1) 训练集/测试集构建（与 v21 逻辑一致）
  2) PCA、标准化、SMOGN fallback、WB2 等全部增强模块
  3) 深度切片三类核心图（train/test、TFe classes、TFe 连续色标）
  4) 3D 钻孔位置图
  5) 勘探线纵剖图、层位切片图
  6) 新增：所有数据输出为 CSV（train/test/all）
"""

import os, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 全局配置
# ------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

LOC_FILE = "钻孔坐标1.xlsx"
GRD_FILE = "实验数据1.xlsx"
DEV_FILE = "钻孔定位1.xlsx"

DEM_FILE = None

BASE_OUTDIR = "outputs_chapter5_dynamic_vis"
os.makedirs(BASE_OUTDIR, exist_ok=True)

# ------------------------------------------------------------
# 数据读取
# ------------------------------------------------------------
def read_data():
    print(f"[Read] {LOC_FILE}, {GRD_FILE}, {DEV_FILE} ...")
    loc = pd.read_excel(LOC_FILE)
    grd = pd.read_excel(GRD_FILE)
    dev = pd.read_excel(DEV_FILE)
    return loc, grd, dev

# ------------------------------------------------------------
# 构建采样点（与 v21 一致）
# ------------------------------------------------------------
def build_samples(loc, grd, step=2.0):
    rows = []
    grd_by = {k: v.sort_values("从").reset_index(drop=True)
              for k, v in grd.groupby("工程号")}
    loc_idx = loc.set_index("工程号")

    for bh in loc_idx.index:
        collar = loc_idx.loc[bh]
        g = grd_by.get(bh)
        if g is None:
            continue
        for _, r in g.iterrows():
            try:
                d0, d1, tfe = float(r["从"]), float(r["至"]), float(r["TFe"])
            except Exception:
                continue
            if not np.isfinite(d0) or not np.isfinite(d1) or not np.isfinite(tfe):
                continue
            if d1 <= d0:
                continue
            n = max(1, int((d1 - d0) / step))
            depths = np.linspace(d0, d1, n)
            for z in depths:
                rows.append(
                    (collar["开孔坐标E"], collar["开孔坐标N"], -z, tfe, bh)
                )
    df = pd.DataFrame(rows, columns=["x", "y", "depth", "TFe", "工程号"])
    print(f"[Build] Generated {len(df)} samples.")
    return df.dropna().reset_index(drop=True), loc

# ------------------------------------------------------------
# 异常值清洗
# ------------------------------------------------------------
def remove_outliers(df, cols, z_th=4.0):
    keep = pd.Series(True, index=df.index)
    for c in cols:
        vals = df[c].astype(float)
        z = (vals - vals.mean()) / (vals.std() + 1e-9)
        keep &= (np.abs(z) <= z_th)
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        keep &= vals.between(low, high)
    removed = (~keep).sum()
    if removed > 0:
        print(f"[Clean] Removed outliers: {removed}")
    return df[keep].reset_index(drop=True)

# ------------------------------------------------------------
# 特征工程（v21 保持一致）
# ------------------------------------------------------------
def feature_engineering(df):
    df = df.copy()
    df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    df["xy"] = df["x"] * df["y"]
    df["depth2"] = df["depth"] ** 2
    df["r2"] = df["r"] ** 2
    dmin, dmax = df["depth"].min(), df["depth"].max()
    df["depth_norm"] = (df["depth"] - dmin) / max(dmax - dmin, 1e-9)
    return df

# ------------------------------------------------------------
# SMOGN fallback
# ------------------------------------------------------------
def smogn_preprocessing(df, target="TFe"):
    k = 0.02
    ql, qh = df[target].quantile(k), df[target].quantile(1 - k)
    ext = pd.concat([df[df[target] <= ql],
                     df[df[target] >= qh]], axis=0)

    if len(ext) > 0:
        aug = ext.sample(min(len(ext), 1000),
                         replace=True,
                         random_state=SEED)
        print("[SMOGN] Extreme oversampling applied.")
        return pd.concat([df, aug], ignore_index=True)
    print("[SMOGN] No extremes found.")
    return df

# ------------------------------------------------------------
# WB2 jitter extreme samples
# ------------------------------------------------------------
def wb2_positioning(df, target="TFe", k=0.02, jitter=0.01):
    df = df.copy()
    q_low, q_high = df[target].quantile(k), df[target].quantile(1 - k)
    low_ext, high_ext = df[df[target] <= q_low], df[df[target] >= q_high]
    s = df[target].std()

    def jitter_df(part):
        if part.empty:
            return part
        J = part.copy()
        noise = np.random.normal(0, max(s * jitter, 1e-6), size=len(J))
        J[target] = J[target] + noise
        return J

    aug = pd.concat([jitter_df(low_ext),
                     jitter_df(high_ext)],
                    ignore_index=True)
    print(f"[WB2] Added {len(aug)} jittered samples.")
    return pd.concat([df, aug], ignore_index=True)

# ------------------------------------------------------------
# PCA 安全添加
# ------------------------------------------------------------
def apply_pca_safe(df, n_components=5, exclude=("TFe", "工程号")):
    use_cols = [c for c in df.columns
                if (c not in exclude) and df[c].dtype.kind in "fc"]
    n_features = len(use_cols)
    if n_features == 0:
        return df, None
    nc = max(1, min(n_components, n_features))
    pca = PCA(n_components=nc, random_state=SEED)
    Z = pca.fit_transform(df[use_cols].values)
    pca_df = pd.DataFrame(Z,
                          columns=[f"PC{i+1}" for i in range(nc)],
                          index=df.index)
    print(f"[PCA] {nc} PCs added")
    return pd.concat([df, pca_df], axis=1), pca

# ------------------------------------------------------------
# 标准化
# ------------------------------------------------------------
def scale_features(df, cols):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols].values)
    sc_df = pd.DataFrame(X,
                         columns=[f"{c}_sc" for c in cols],
                         index=df.index)
    print(f"[Scale] {len(cols)} features scaled.")
    return pd.concat([df, sc_df], axis=1), scaler

# ------------------------------------------------------------
# Train/Test 预处理入口
# ------------------------------------------------------------
def prepare_train_test_for_vis():
    loc, grd, dev = read_data()
    df, loc_idx = build_samples(loc, grd)
    df = remove_outliers(df, ["x", "y", "depth", "TFe"])
    df = feature_engineering(df)
    df = smogn_preprocessing(df)
    df = wb2_positioning(df)

    df, _ = apply_pca_safe(df, n_components=5)

    base_feats = ["x", "y", "depth", "r", "xy", "depth2", "r2", "depth_norm"]
    pca_feats = [c for c in df.columns if c.startswith("PC")]
    df, scaler = scale_features(df, base_feats + pca_feats)

    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=SEED
    )
    print(f"[Split] Train={len(train_df)}, Test={len(test_df)}")

    # merge collar extra columns
    for c in loc.columns:
        if c not in ["工程号", "开孔坐标E", "开孔坐标N"]:
            info = loc[["工程号", c]].drop_duplicates()
            train_df = train_df.merge(info, on="工程号", how="left")
            test_df = test_df.merge(info, on="工程号", how="left")

    return train_df, test_df, loc

# ------------------------------------------------------------
# DEM
# ------------------------------------------------------------
def load_dem_surface():
    if DEM_FILE is None or not os.path.exists(DEM_FILE):
        print("[DEM] Skip")
        return None
    dem = pd.read_csv(DEM_FILE)
    return dem if {"x","y","z"}.issubset(dem.columns) else None

def clip_by_dem(df, dem):
    if dem is None:
        return df
    # 简易剪裁
    keep = df["depth"] <= 0
    return df[keep].reset_index(drop=True)

# ------------------------------------------------------------
# TFe 分类
# ------------------------------------------------------------
def classify_tfe(t):
    if t < 30:
        return "<30"
    elif t < 40:
        return "30-40"
    elif t < 50:
        return "40-50"
    else:
        return ">=50"

# ------------------------------------------------------------
# 3D 分布图
# ------------------------------------------------------------
def plot_3d_train_test(train_df, test_df, outdir):
    from mpl_toolkits.mplot3d import Axes3D
    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure(figsize=(8.6, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(train_df["x"], train_df["y"], train_df["depth"],
               c="tab:blue", s=6, alpha=0.5, label="Train")
    ax.scatter(test_df["x"], test_df["y"], test_df["depth"],
               c="tab:red", s=10, alpha=0.7, label="Test")
    ax.invert_zaxis()
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Depth")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "3d_train_test_points.png"), dpi=400)
    plt.close()
    print("[Plot] 3D ok.")

# ------------------------------------------------------------
# 深度切片三类图
# ------------------------------------------------------------
def plot_depth_slices_every_20m(train_df, test_df, outdir,
                                interval=80, band=10):
    os.makedirs(outdir, exist_ok=True)

    zmin = float(min(train_df["depth"].min(), test_df["depth"].min()))
    zmax = float(max(train_df["depth"].max(), test_df["depth"].max()))
    levels = np.arange(zmin, zmax+interval, interval)

    for z0 in levels:
        sel_tr = train_df[np.abs(train_df["depth"] - z0) <= band]
        sel_te = test_df[np.abs(test_df["depth"] - z0) <= band]
        if len(sel_tr)+len(sel_te) == 0:
            continue

        # 1) train/test 散点
        plt.figure(figsize=(7.2,6))
        if len(sel_tr):
            plt.scatter(sel_tr["x"], sel_tr["y"],
                        c="#1f77b4", s=15, alpha=0.7,
                        label=f"Train (n={len(sel_tr)})")
        if len(sel_te):
            plt.scatter(sel_te["x"], sel_te["y"],
                        c="#d62728", s=15, alpha=0.7,
                        label=f"Test (n={len(sel_te)})")
        plt.title(f"Plan View @ depth≈{z0:.1f} m (±{band})")
        plt.xlabel("X"); plt.ylabel("Y"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,
                    f"slice_{int(z0)}m_sets.png"), dpi=400)
        plt.close()

        # 2) TFe class
        merged = pd.concat([sel_tr, sel_te], ignore_index=True)
        merged["TFe_class"] = merged["TFe"].apply(classify_tfe)
        colors = {"<30":"#3288bd","30-40":"#66c2a5",
                  "40-50":"#fee08b",">=50":"#d53e4f"}

        plt.figure(figsize=(7.2,6))
        for cls in ["<30","30-40","40-50",">=50"]:
            sub = merged[merged["TFe_class"]==cls]
            if len(sub):
                plt.scatter(sub["x"], sub["y"],
                            c=colors[cls], s=20, alpha=0.8,
                            label=f"{cls} (n={len(sub)})")
        plt.title(f"TFe Classes @ {z0:.1f} m")
        plt.xlabel("X"); plt.ylabel("Y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,
                    f"slice_{int(z0)}m_TFe_classes.png"),
                    dpi=400)
        plt.close()

        # 3) 连续色标（train / test）
        fig, ax = plt.subplots(1,2, figsize=(12.5,5.8), sharex=True, sharey=True)
        if len(sel_tr):
            sc1 = ax[0].scatter(sel_tr["x"], sel_tr["y"],
                                c=sel_tr["TFe"], cmap="viridis",
                                s=20, alpha=0.85)
            fig.colorbar(sc1, ax=ax[0], shrink=0.85)
        ax[0].set_title(f"Train @ {z0:.1f} m"); ax[0].set_xlabel("X"); ax[0].set_ylabel("Y")

        if len(sel_te):
            sc2 = ax[1].scatter(sel_te["x"], sel_te["y"],
                                c=sel_te["TFe"], cmap="plasma",
                                s=20, alpha=0.85)
            fig.colorbar(sc2, ax=ax[1], shrink=0.85)
        ax[1].set_title(f"Test @ {z0:.1f} m"); ax[1].set_xlabel("X")

        plt.tight_layout()
        plt.savefig(os.path.join(outdir,
                    f"slice_{int(z0)}m_TFe.png"),
                    dpi=400)
        plt.close()

# ------------------------------------------------------------
# 勘探线剖面（自动识别 line 字段）
# ------------------------------------------------------------
def find_line_column(df):
    for c in df.columns:
        name = str(c)
        if "勘探线" in name or "线号" in name:
            return c
        if "line" in name.lower():
            return c
    return None

def plot_sections_by_line(train_df, test_df, loc_df, outdir):
    os.makedirs(outdir, exist_ok=True)
    line_col = find_line_column(loc_df)
    if line_col is None:
        print("[Line] No line column detected.")
        return

    m = loc_df[["工程号", line_col]].drop_duplicates()
    train_df = train_df.merge(m, on="工程号", how="left")
    test_df = test_df.merge(m, on="工程号", how="left")

    all_df = pd.concat([train_df.assign(_set="Train"),
                        test_df.assign(_set="Test")],
                       ignore_index=True)

    for lid, sub in all_df.groupby(line_col):
        if pd.isna(lid) or len(sub)<10:
            continue
        sub = sub.copy()
        var_x, var_y = sub["x"].var(), sub["y"].var()
        if var_x < var_y:
            sub["hcoord"] = sub["y"]; hlabel="Y"
        else:
            sub["hcoord"] = sub["x"]; hlabel="X"

        sub["hcoord_rel"] = sub["hcoord"] - sub["hcoord"].min()

        fig, ax = plt.subplots(figsize=(8.5,5.8))
        sc = ax.scatter(sub["hcoord_rel"], sub["depth"],
                        c=sub["TFe"], cmap="viridis",
                        s=25, alpha=0.85)
        ax.invert_yaxis()
        fig.colorbar(sc, ax=ax, label="TFe (%)")
        ax.set_xlabel(hlabel+" (relative)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"Section Line: {lid}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,
                    f"section_line_{lid}.png"), dpi=400)
        plt.close()

# ------------------------------------------------------------
# 层位切片
# ------------------------------------------------------------
def find_layer_column(df):
    for c in df.columns:
        name = str(c)
        if "层位" in name or "岩性" in name or "层号" in name:
            return c
        if "lith" in name.lower() or "layer" in name.lower():
            return c
    return None

def plot_layer_maps(train_df, test_df, outdir):
    os.makedirs(outdir, exist_ok=True)
    all_df = pd.concat([train_df.assign(_set="Train"),
                        test_df.assign(_set="Test")],
                       ignore_index=True)

    layer_col = find_layer_column(all_df)
    if layer_col is None:
        print("[Layer] No layer column detected.")
        return

    for lid, sub in all_df.groupby(layer_col):
        if pd.isna(lid) or len(sub)<10:
            continue

        plt.figure(figsize=(7.2,6))
        sc = plt.scatter(sub["x"], sub["y"],
                         c=sub["TFe"], cmap="viridis",
                         s=20, alpha=0.85)
        plt.colorbar(sc, label="TFe (%)")
        plt.xlabel("X"); plt.ylabel("Y")
        plt.title(f"Layer: {layer_col}={lid}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,
                    f"layer_{lid}.png"), dpi=400)
        plt.close()

# ------------------------------------------------------------
# main —— 这里包含 CSV 导出
# ------------------------------------------------------------
def main():

    # 1. preprocess (v21 逻辑)
    train_df, test_df, loc_df = prepare_train_test_for_vis()

    # 2. DEM 裁剪
    dem = load_dem_surface()
    if dem is not None:
        train_df = clip_by_dem(train_df, dem)
        test_df = clip_by_dem(test_df, dem)

    # ---------------------------- CSV 输出 ----------------------------
    csv_dir = os.path.join(BASE_OUTDIR, "csv_exports")
    os.makedirs(csv_dir, exist_ok=True)

    all_df = pd.concat([train_df.assign(_set="Train"),
                        test_df.assign(_set="Test")],
                       ignore_index=True)

    all_df.to_csv(os.path.join(csv_dir,
                               "all_samples_after_preprocessing.csv"),
                  index=False, encoding="utf-8-sig")

    train_df.to_csv(os.path.join(csv_dir, "train_samples_full.csv"),
                    index=False, encoding="utf-8-sig")

    test_df.to_csv(os.path.join(csv_dir, "test_samples_full.csv"),
                   index=False, encoding="utf-8-sig")

    print(f"[CSV] Saved to: {csv_dir}")

    # 3. 3D 图
    plot_3d_train_test(train_df, test_df,
                       os.path.join(BASE_OUTDIR, "3d_overview"))

    # 4. 深度切片（80m）
    plot_depth_slices_every_20m(train_df, test_df,
                                os.path.join(BASE_OUTDIR, "depth_slices_every80m"),
                                interval=80.0, band=10.0)

    # 5. 勘探线剖面
    plot_sections_by_line(train_df, test_df, loc_df,
                          os.path.join(BASE_OUTDIR, "sections_by_line"))

    # 6. 层位切片（可选）
    plot_layer_maps(train_df, test_df,
                    os.path.join(BASE_OUTDIR, "layer_maps"))

    print("\n[Done] All figures and CSV saved.\n")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
