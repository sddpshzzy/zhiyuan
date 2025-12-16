# -*- coding: utf-8 -*-
"""
Qidashan_Vis_TrainTest_Sections_v1.py
基于 Qidashan_XGB_Kriging_Realtime_v21_full.py 中的相同数据与预处理逻辑，
仅做可视化（train/test 剖面 + 深度切片），不影响原始建模脚本。

功能：
  1) 3D 钻孔 train/test 分布图
  2) 深度每隔 20m 出一张切片图（多套）：
        - Train/Test 分色平面
        - Train/Test 按 TFe 连续色标
        - Train+Test TFe 等值线/填色等值线
        - Train+Test 按 TFe 分级着色
  3) 按勘探线纵剖图（若 collar 中存在“勘探线/线号/line”等字段）
  4) 若有 “层位” 字段，按地层输出 X-Y 面图
  5) 若提供 DEM_surface.csv(x,y,z)，用 DEM 裁剪（可选）
"""

import os, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ------------------- 全局配置（与 v21 保持一致的关键参数） -------------------
SEED = 42
np.random.seed(SEED)

# 原始 Excel 文件名（保持和 v21 一致）
LOC_FILE = "钻孔坐标1.xlsx"     # collar
GRD_FILE = "实验数据1.xlsx"     # 实验数据
DEV_FILE = "钻孔定位1.xlsx"     # 此脚本不直接用 dev，但保留以兼容结构

# 可选 DEM 文件（若无则设为 None）
DEM_FILE = None  # "DEM_surface.csv"  # CSV: x, y, z

# 输出目录
BASE_OUTDIR = "outputs_chapter5_dynamic_vis"
os.makedirs(BASE_OUTDIR, exist_ok=True)

# ------------------- 数据读入与预处理（尽量贴近 v21 的 Stage1 前半段） -------------------
def read_data():
    print(f"[Read] {LOC_FILE}, {GRD_FILE}, {DEV_FILE} ...")
    loc = pd.read_excel(LOC_FILE)
    grd = pd.read_excel(GRD_FILE)
    dev = pd.read_excel(DEV_FILE)  # 暂未使用
    return loc, grd, dev

def build_samples(loc, grd, step=2.0):
    """
    和 v21 中 build_samples 一致逻辑：
      - 按实验数据“从-至”区间在钻孔上等步长采样
      - 这里没有用定向/侧斜，直接用 E/N，depth 用负值
    """
    rows=[]
    grd_by = {k: v.sort_values("从").reset_index(drop=True)
              for k,v in grd.groupby("工程号")}
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
                rows.append((collar["开孔坐标E"], collar["开孔坐标N"], -z, tfe, bh))
    df = pd.DataFrame(rows, columns=["x","y","depth","TFe","工程号"])
    print(f"[Build] Generated {len(df)} samples.")
    return df.dropna().reset_index(drop=True), loc

def remove_outliers(df, cols, z_th=4.0):
    keep = pd.Series(True, index=df.index)
    for c in cols:
        vals = df[c].astype(float)
        z = (vals - vals.mean()) / (vals.std() + 1e-9)
        keep &= (np.abs(z) <= z_th)
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
        keep &= vals.between(low, high)
    removed = (~keep).sum()
    if removed > 0:
        print(f"[Clean] Removed outliers: {removed}")
    return df[keep].reset_index(drop=True)

def feature_engineering(df):
    df = df.copy()
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2)
    df["xy"] = df["x"] * df["y"]
    df["depth2"] = df["depth"]**2
    df["r2"] = df["r"]**2
    dmin, dmax = df["depth"].min(), df["depth"].max()
    df["depth_norm"] = (df["depth"] - dmin) / max(dmax - dmin, 1e-9)
    return df

def smogn_preprocessing(df, target="TFe"):
    """
    这里沿用 v21 的 fallback 简化版：
      - 若安装了 smogn，可自行替换为 smoter
      - 此处只做极值简单过采样，保证逻辑接近
    """
    k = 0.02
    ql, qh = df[target].quantile(k), df[target].quantile(1-k)
    ext = pd.concat([df[df[target]<=ql], df[df[target]>=qh]], axis=0)
    if len(ext) > 0:
        aug = ext.sample(min(len(ext), 1000), replace=True, random_state=SEED)
        df2 = pd.concat([df, aug], ignore_index=True)
        print("[SMOGN-fallback] Extreme oversampling applied.")
        return df2.reset_index(drop=True)
    print("[SMOGN-fallback] No extremes found. Skipped.")
    return df.reset_index(drop=True)

def wb2_positioning(df, target="TFe", k=0.02, jitter=0.01):
    df = df.copy()
    q_low, q_high = df[target].quantile(k), df[target].quantile(1-k)
    low_ext = df[df[target] <= q_low]
    high_ext = df[df[target] >= q_high]
    s = df[target].std()
    def jitter_df(part):
        if part.empty: return part
        J = part.copy()
        noise = np.random.normal(0, max(s*jitter, 1e-6), size=len(J))
        J[target] = J[target] + noise
        return J
    aug = pd.concat([jitter_df(low_ext), jitter_df(high_ext)], ignore_index=True)
    out = pd.concat([df, aug], ignore_index=True)
    print(f"[WB2] Added {len(aug)} jittered extreme samples.")
    return out.reset_index(drop=True)

def apply_pca_safe(df, n_components=5, exclude=("TFe","工程号")):
    use_cols = [c for c in df.columns if (c not in exclude) and (df[c].dtype.kind in "fc")]
    n_features = len(use_cols)
    if n_features == 0:
        return df, None
    nc = max(1, min(n_components, n_features))
    pca = PCA(n_components=nc, random_state=SEED)
    Z = pca.fit_transform(df[use_cols].values)
    pca_df = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(nc)], index=df.index)
    df = pd.concat([df, pca_df], axis=1)
    print(f"[PCA] Added {nc} PCs (from {n_features} numeric features).")
    return df, pca

def scale_features(df, feature_cols):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    sc_df = pd.DataFrame(X, columns=[f"{c}_sc" for c in feature_cols], index=df.index)
    print(f"[Scale] Standardized {len(feature_cols)} features.")
    return pd.concat([df, sc_df], axis=1), scaler

def prepare_train_test_for_vis():
    loc, grd, dev = read_data()
    df, loc_idx = build_samples(loc, grd)
    df = remove_outliers(df, cols=["x","y","depth","TFe"])
    df = feature_engineering(df)
    df = smogn_preprocessing(df, target="TFe")
    df = wb2_positioning(df, target="TFe", k=0.02, jitter=0.01)
    df, _ = apply_pca_safe(df, n_components=5, exclude=("TFe","工程号"))

    base_feats = ["x","y","depth","r","xy","depth2","r2","depth_norm"]
    pca_feats = [c for c in df.columns if c.startswith("PC")]
    df, scaler = scale_features(df, base_feats + pca_feats)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=SEED)
    print(f"[Split] Train={len(train_df)}, Test={len(test_df)}")

    # 把勘探线号、层位等 collar 信息 merge 进 train/test（方便剖面）
    for extra_col in loc.columns:
        if extra_col not in ["工程号","开孔坐标E","开孔坐标N"]:
            # 广播 collar 信息
            m = loc[["工程号", extra_col]].drop_duplicates()
            train_df = train_df.merge(m, on="工程号", how="left")
            test_df  = test_df.merge(m, on="工程号", how="left")

    return train_df, test_df, loc

# ------------------- DEM 裁剪（可选） -------------------
def load_dem_surface():
    if DEM_FILE is None or (not os.path.exists(DEM_FILE)):
        print("[DEM] No DEM file provided, skip DEM clipping.")
        return None
    dem = pd.read_csv(DEM_FILE)
    if not {"x","y","z"}.issubset(dem.columns):
        print("[DEM] DEM_surface.csv must contain columns x,y,z. Skip.")
        return None
    print(f"[DEM] Loaded DEM_surface with {len(dem)} points.")
    return dem

def clip_by_dem(df, dem):
    """简单最近邻 DEM 剪裁：保留 depth 在地表下方的点"""
    if dem is None:
        return df
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(dem[["x","y"]].values)
    dist, idx = nn.kneighbors(df[["x","y"]].values)
    surface_z = dem["z"].values[idx[:,0]]
    # depth 为负值，折算到同一系：假定地表 z → depth=0，下方为负
    # 若 df["depth"] 小于 surface_depth(近似 0)，说明在地表下；否则裁掉
    keep = df["depth"].values <= 0.0  # 简化处理
    kept = keep.sum()
    print(f"[DEM] Simple clipping: kept {kept}/{len(df)} points (based on depth<=0).")
    return df[keep].reset_index(drop=True)

# ------------------- TFe 分级 -------------------
def classify_tfe(t):
    """
    可按需要改级别，这里简单示例：
        <30, 30-40, 40-50, >=50
    """
    if t < 30: return "<30"
    elif t < 40: return "30-40"
    elif t < 50: return "40-50"
    else: return ">=50"

# ------------------- 3D 钻孔 Train/Test 分布 -------------------
def plot_3d_train_test(train_df, test_df, outdir):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure(figsize=(8.6, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(train_df["x"], train_df["y"], train_df["depth"],
               c="tab:blue", s=6, alpha=0.45, label="Train")
    ax.scatter(test_df["x"], test_df["y"], test_df["depth"],
               c="tab:red", s=10, alpha=0.75, label="Test")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.invert_zaxis()  # 深部向下
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "3d_train_test_points.png"), dpi=400)
    plt.close(fig)
    print("[Plot] 3D train/test distribution saved.")

# ------------------- 深度每 20m 切片（多套图） -------------------
def plot_depth_slices_every_20m(train_df, test_df, outdir,
                                interval=20.0, band=10.0):
    os.makedirs(outdir, exist_ok=True)

    z_min = float(min(train_df["depth"].min(), test_df["depth"].min()))
    z_max = float(max(train_df["depth"].max(), test_df["depth"].max()))
    # 确保按“向下为负”的系统，方便阅读，可从较浅到较深排序
    # 若 z_min 为负值较浅，可用 np.arange(z_min, z_max, interval)
    # 考虑 depth 为负数，z_min < z_max，例：-260 ~ -20
    # 用 sorted 深度列表更加保险：
    depth_levels = np.arange(z_min, z_max + interval, interval)
    depth_levels = sorted(depth_levels, key=lambda x: x)  # 数值递增

    print(f"[Slice] Depth range {z_min:.1f} → {z_max:.1f} 每隔 {interval}m, band=±{band}m, 共 {len(depth_levels)} 个切片")

    for z0 in depth_levels:
        # 选取该切片带的点
        sel_tr = train_df[np.abs(train_df["depth"] - z0) <= band]
        sel_te = test_df[np.abs(test_df["depth"] - z0) <= band]
        if len(sel_tr) + len(sel_te) == 0:
            continue

        # 1) Train/Test 分色平面图
        plt.figure(figsize=(7.2, 6.0))
        if len(sel_tr):
            plt.scatter(sel_tr["x"], sel_tr["y"], s=12, alpha=0.6,
                        c="#1f77b4", label=f"Train (n={len(sel_tr)})")
        if len(sel_te):
            plt.scatter(sel_te["x"], sel_te["y"], s=14, alpha=0.7,
                        c="#d62728", label=f"Test (n={len(sel_te)})")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"Plan View @ depth≈{z0:.1f} m (±{band} m)")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"slice_{int(z0)}m_sets.png"), dpi=400)
        plt.close()

        # 2) Train/Test 按 TFe 连续色标
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharex=True, sharey=True)
        if len(sel_tr):
            sc1 = axes[0].scatter(sel_tr["x"], sel_tr["y"], c=sel_tr["TFe"],
                                  cmap="viridis", alpha=0.85, s=18)
            fig.colorbar(sc1, ax=axes[0], shrink=0.85, label="TFe (%)")
        axes[0].set_title(f"Train @ depth≈{z0:.1f} m")
        axes[0].set_xlabel("X (m)")
        axes[0].set_ylabel("Y (m)")

        if len(sel_te):
            sc2 = axes[1].scatter(sel_te["x"], sel_te["y"], c=sel_te["TFe"],
                                  cmap="plasma", alpha=0.85, s=18)
            fig.colorbar(sc2, ax=axes[1], shrink=0.85, label="TFe (%)")
        axes[1].set_title(f"Test @ depth≈{z0:.1f} m")
        axes[1].set_xlabel("X (m)")

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"slice_{int(z0)}m_TFe.png"), dpi=400)
        plt.close(fig)

        # 3) Train+Test TFe 等值线（tricontourf）
        merged = pd.concat([sel_tr, sel_te], ignore_index=True)
        if len(merged) >= 10:  # 太少就不画等值线
            fig2, ax2 = plt.subplots(figsize=(7.2, 6.0))
            tcf = ax2.tricontourf(merged["x"], merged["y"], merged["TFe"],
                                  levels=12, cmap="viridis")
            cs = ax2.tricontour(merged["x"], merged["y"], merged["TFe"],
                                levels=12, colors="k", linewidths=0.5, alpha=0.6)
            ax2.clabel(cs, inline=True, fontsize=7)
            fig2.colorbar(tcf, ax=ax2, label="TFe (%)")
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            ax2.set_title(f"TFe Contours @ depth≈{z0:.1f} m (Train+Test)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"slice_{int(z0)}m_contour_TFe.png"), dpi=400)
            plt.close(fig2)

        # 4) 按 TFe 分级着色（离散 colormap）
        merged = merged.copy()
        merged["TFe_class"] = merged["TFe"].apply(classify_tfe)
        class_order = ["<30","30-40","40-50",">=50"]
        colors = {"<30":"#3288bd", "30-40":"#66c2a5", "40-50":"#fee08b", ">=50":"#d53e4f"}

        plt.figure(figsize=(7.2, 6.0))
        for cls in class_order:
            sub = merged[merged["TFe_class"] == cls]
            if len(sub) == 0:
                continue
            plt.scatter(sub["x"], sub["y"], s=20, alpha=0.8,
                        c=colors.get(cls, "gray"), label=f"{cls} (n={len(sub)})")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"TFe Classes @ depth≈{z0:.1f} m (Train+Test)")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"slice_{int(z0)}m_TFe_classes.png"), dpi=400)
        plt.close()

# ------------------- 按勘探线剖面图 -------------------
def find_line_column(loc_df):
    """
    尝试自动找勘探线字段：
      - 含“勘探线” 或 “线号”
      - 或包含 'line'/'Line' 的列
    """
    candidates = []
    for c in loc_df.columns:
        cname = str(c)
        if ("勘探线" in cname) or ("线号" in cname):
            candidates.append(c)
        elif "line" in cname.lower():
            candidates.append(c)
    if candidates:
        print(f"[Line] 使用勘探线字段：{candidates[0]}")
        return candidates[0]
    print("[Line] 未找到明显勘探线字段，将不能按勘探线剖面，仅可改写脚本手动指定。")
    return None

def plot_sections_by_line(train_df, test_df, loc_df, outdir):
    os.makedirs(outdir, exist_ok=True)
    line_col = find_line_column(loc_df)
    if line_col is None:
        return

    # 把勘探线字段从 collar merge 进样点（如果前面 prepare_train_test_for_vis 已经 merge，则不必）
    m_line = loc_df[["工程号", line_col]].drop_duplicates()
    train_df = train_df.merge(m_line, on="工程号", how="left")
    test_df  = test_df.merge(m_line, on="工程号", how="left")

    all_df = pd.concat([
        train_df.assign(_set="Train"),
        test_df.assign(_set="Test")
    ], ignore_index=True)

    for line_id, sub in all_df.groupby(line_col):
        if pd.isna(line_id):
            continue
        sub = sub.copy()
        if len(sub) < 10:
            continue

        # 判断该勘探线大致走向：看 x/y 方差
        var_x = sub["x"].var()
        var_y = sub["y"].var()
        if var_x < var_y:
            # 认为勘探线沿 y 方向（x 近似固定），剖面横轴用 y
            sub["hcoord"] = sub["y"]
            hlabel = "Along-line (Y, m)"
        else:
            sub["hcoord"] = sub["x"]
            hlabel = "Along-line (X, m)"

        # 归一化/平移，使横轴从 0 开始便于阅读
        sub["hcoord_rel"] = sub["hcoord"] - sub["hcoord"].min()

        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        # Train/Test 分图层 + TFe 着色
        for sname, sdata in sub.groupby("_set"):
            sc = ax.scatter(sdata["hcoord_rel"], sdata["depth"],
                            c=sdata["TFe"], cmap="viridis", s=25,
                            alpha=0.85, label=f"{sname} (n={len(sdata)})")
        ax.invert_yaxis()
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("TFe (%)")
        ax.set_xlabel(hlabel + " (relative)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"Section along line {line_id}")
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"section_line_{line_id}.png"), dpi=400)
        plt.close(fig)

        print(f"[Section] Line {line_id} saved.")

# ------------------- 按地层切片（若有“层位”等字段） -------------------
def find_layer_column(df):
    for c in df.columns:
        cname = str(c)
        if ("层位" in cname) or ("岩性" in cname) or ("层号" in cname):
            print(f"[Layer] 使用层位字段：{c}")
            return c
        if "lith" in cname.lower() or "layer" in cname.lower():
            print(f"[Layer] 使用层位字段：{c}")
            return c
    print("[Layer] 未找到层位字段，跳过地层切片。")
    return None

def plot_layer_maps(train_df, test_df, outdir):
    os.makedirs(outdir, exist_ok=True)
    all_df = pd.concat([
        train_df.assign(_set="Train"),
        test_df.assign(_set="Test")
    ], ignore_index=True)

    layer_col = find_layer_column(all_df)
    if layer_col is None:
        return

    for layer_id, sub in all_df.groupby(layer_col):
        if pd.isna(layer_id):
            continue
        if len(sub) < 10:
            continue

        plt.figure(figsize=(7.2, 6.0))
        sc = plt.scatter(sub["x"], sub["y"], c=sub["TFe"], cmap="viridis",
                         s=20, alpha=0.85)
        plt.colorbar(sc, label="TFe (%)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"Layer map: {layer_col}={layer_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"layer_{layer_id}.png"), dpi=400)
        plt.close()
        print(f"[Layer] Map for {layer_id} saved.")

# ------------------- main -------------------
def main():
    # 1. 构建与 v21 相同逻辑的 train/test 样本
    train_df, test_df, loc_df = prepare_train_test_for_vis()

    # 2. DEM 裁剪（可选）
    dem = load_dem_surface()
    if dem is not None:
        train_df = clip_by_dem(train_df, dem)
        test_df  = clip_by_dem(test_df, dem)

    # 3. 3D 钻孔 Train/Test
    out_3d = os.path.join(BASE_OUTDIR, "3d_overview")
    plot_3d_train_test(train_df, test_df, out_3d)

    # 4. 深度每 20 m 切片（带等值线 + TFe 分级）
    out_slices = os.path.join(BASE_OUTDIR, "depth_slices_every20m")
    plot_depth_slices_every_20m(train_df, test_df, out_slices,
                                interval=20.0, band=10.0)

    # 5. 按勘探线剖面
    out_sections = os.path.join(BASE_OUTDIR, "sections_by_line")
    plot_sections_by_line(train_df, test_df, loc_df, out_sections)

    # 6. 按地层面图（如果有层位信息）
    out_layers = os.path.join(BASE_OUTDIR, "layer_maps")
    plot_layer_maps(train_df, test_df, out_layers)

    print(f"\n[Done] All visual outputs saved under: {BASE_OUTDIR}\n")

if __name__ == "__main__":
    main()
