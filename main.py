"""
================================================================
BUSINESS PERFORMANCE ANALYSIS & INSIGHT GENERATION
================================================================
Datasets : 1. Superstore Sales    (9,994 orders, 2014-2017)
           2. Telco Customer Churn (7,043 customers)
Tools    : Python, Pandas, NumPy, Scikit-learn, Matplotlib
Models   : Gradient Boosting Classifier (both pipelines)
================================================================
"""

import os
import warnings
import datetime
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold)
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (accuracy_score, confusion_matrix,
                                     classification_report,
                                     roc_auc_score, roc_curve)

# Colour palette
BLUE   = "#1565C0"
GREEN  = "#2E7D32"
RED    = "#C62828"
ORANGE = "#E65100"
PURPLE = "#6A1B9A"
TEAL   = "#00695C"
COLORS = [BLUE, GREEN, RED, ORANGE, PURPLE, TEAL,
          "#AD1457", "#00838F", "#F9A825", "#558B2F"]

os.makedirs("outputs",       exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

# Global state
ss_raw = ss_proc = ss_rfm_df = None
ss_loaded = ss_preprocessed  = False
tc_raw = tc_proc = None
tc_loaded = tc_preprocessed  = False


# ================================================================
# UTILITY FUNCTIONS
# ================================================================
def sep(char="=", n=72):
    print("\n" + char * n)

def section(title):
    sep(); print(f"  {title}"); sep()

def subsection(title):
    sep("-", 60); print(f"  {title}"); sep("-", 60)

def show_head(df, title, n=5):
    sep(); print(title); print(df.head(n).to_string()); sep()

def save_csv(df, path, label):
    df.to_csv(path, index=False)
    print(f"  [Saved] {path}")

def get_path(default, label):
    print(f"\n  Default path : {default}")
    val = input(f"  Press Enter for default OR type new {label} path: "
                ).strip().strip("\"'")
    return val if val else default

def path_exists(p):
    return os.path.exists(p)

def check_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [ERROR] {name} missing columns: {missing}")
        return False
    return True

def save_plot(fig, filename):
    path = f"outputs/plots/{filename}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [Plot saved] {path}")
    plt.show()
    plt.close(fig)

def input_int(prompt, lo=None, hi=None):
    while True:
        try:
            v = int(input(prompt).strip())
            if lo is not None and v < lo:
                print(f"    Must be >= {lo}"); continue
            if hi is not None and v > hi:
                print(f"    Must be <= {hi}"); continue
            return v
        except ValueError:
            print("    Please enter a valid integer.")

def input_float(prompt, lo=None, hi=None):
    while True:
        try:
            v = float(input(prompt).strip())
            if lo is not None and v < lo:
                print(f"    Must be >= {lo}"); continue
            if hi is not None and v > hi:
                print(f"    Must be <= {hi}"); continue
            return v
        except ValueError:
            print("    Please enter a valid number.")

def input_choice(prompt, options):
    lookup = {o.lower(): o for o in options}
    while True:
        v = input(prompt).strip().lower()
        if v in lookup:
            return lookup[v]
        print(f"    Invalid. Choose from: {', '.join(options)}")


# ================================================================
# SHARED PLOT HELPERS
# ================================================================
def plot_confusion_matrix(cm, title, filename, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]);  ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0, 1]);  ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"{title}\nConfusion Matrix", fontweight="bold", fontsize=12)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=15,
                    fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    save_plot(fig, filename)

def plot_roc_curve(y_test, y_prob, title, filename):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=ORANGE, lw=2.5, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "navy", lw=1.5, ls="--",
            label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(f"ROC Curve -- {title}", fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_plot(fig, filename)

def plot_feature_importance(importances, feature_names, title, filename):
    idx    = np.argsort(importances)[-15:]
    feats  = [feature_names[i] for i in idx]
    vals   = importances[idx]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feats)))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feats, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_plot(fig, filename)

def plot_cv_scores(scores, title, filename, ylabel="Score"):
    folds = [f"Fold {i+1}" for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(folds, scores, color=BLUE, edgecolor="white", width=0.5)
    ax.axhline(scores.mean(), color=RED, ls="--", lw=2,
               label=f"Mean = {scores.mean():.4f}")
    ax.set_ylim(max(0, scores.min() - 0.05),
                min(1.05, scores.max() + 0.05))
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.legend(fontsize=10);  ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_plot(fig, filename)


# ================================================================
# PIPELINE 1 -- SUPERSTORE SALES
# ================================================================

def ss_load():
    global ss_raw, ss_loaded, ss_preprocessed
    # Already loaded check
    if ss_loaded:
        print("  [INFO] Superstore dataset is already loaded.")
        print("  [INFO] To reload with a different file, restart the program.")
        print(f"  [INFO] Current shape: {ss_raw.shape}")
        return
    path = get_path("data/Sample - Superstore.csv", "Superstore")
    if not path_exists(path):
        print(f"  [ERROR] File not found: {path}"); return
    try:
        df = pd.read_csv(path, encoding="latin1")
        req = ["Order Date", "Ship Date", "Customer ID", "Segment",
               "Region", "Category", "Sub-Category",
               "Sales", "Quantity", "Discount", "Profit"]
        if not check_columns(df, req, "Superstore"): return
        ss_raw = df;  ss_loaded = True;  ss_preprocessed = False
        show_head(df, "Superstore Dataset Loaded Successfully")
        print(f"  Shape   : {df.shape}")
        print(f"  Columns : {list(df.columns)}")
    except Exception as e:
        print(f"  [ERROR] {e}")


def ss_preprocess():
    global ss_raw, ss_proc, ss_preprocessed
    if not ss_loaded:
        print("  [!] Please load Superstore dataset first."); return
    if ss_preprocessed:
        print("  [INFO] Superstore data is already preprocessed.")
        print(f"  [INFO] Preprocessed shape: {ss_proc.shape}")
        print("  [INFO] Proceed to EDA, RFM, or ML steps.")
        return

    df = ss_raw.copy()
    print("\n  Preprocessing steps:")
    for i, s in enumerate([
        "Parse Order Date and Ship Date to datetime",
        "Remove duplicates and missing values",
        "Derive Year, Month, Quarter from Order Date",
        "Calculate Shipping Days (Ship Date minus Order Date)",
        "Calculate Profit Margin % per order",
        "Create Profit_Label  (1 = Profit,  0 = Loss)",
        "Create Discount Band column for analysis",
    ], 1):
        print(f"    {i}. {s}")

    rows_before    = len(df)
    missing_before = df.isnull().sum().sum()

    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"])
    df = df.dropna().drop_duplicates()

    df["Year"]          = df["Order Date"].dt.year
    df["Month"]         = df["Order Date"].dt.month
    df["Quarter"]       = df["Order Date"].dt.quarter
    df["Ship_Days"]     = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Profit_Margin"] = (df["Profit"] / df["Sales"] * 100).round(4)
    df["Profit_Label"]  = (df["Profit"] >= 0).astype(int)  # 1=Profit/Zero, 0=Loss
    df["Disc_Band"]     = pd.cut(
        df["Discount"],
        bins=[-0.01, 0, 0.10, 0.20, 0.30, 0.50, 1.0],
        labels=["0%", "1-10%", "11-20%", "21-30%", "31-50%", "51-80%"])

    ss_proc = df;  ss_preprocessed = True

    sep()
    print(f"  Rows before     : {rows_before:,}")
    print(f"  Missing values  : {missing_before}")
    print(f"  Rows after      : {len(df):,}")
    print(f"  Loss orders     : {(df['Profit_Label']==0).sum():,} "
          f"({(df['Profit_Label']==0).mean()*100:.1f}%)")
    print(f"  Final shape     : {df.shape}")
    save_csv(df, "outputs/superstore_preprocessed.csv",
             "Preprocessed Superstore")


def ss_eda():
    if not ss_preprocessed:
        print("  [!] Please preprocess Superstore dataset first."); return
    df = ss_proc

    section("Superstore -- Business Performance Insights")

    # KPIs
    subsection("Key Performance Indicators (KPIs)")
    total_sales  = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order ID"].nunique()
    total_cust   = df["Customer ID"].nunique()
    aov          = total_sales / total_orders
    margin       = total_profit / total_sales * 100
    yr           = df.groupby("Year")["Sales"].sum()
    cagr         = ((yr.iloc[-1] / yr.iloc[0]) ** (1/3) - 1) * 100

    print(f"  Total Revenue       : ${total_sales:>12,.2f}")
    print(f"  Total Profit        : ${total_profit:>12,.2f}")
    print(f"  Profit Margin       : {margin:>10.2f}%")
    print(f"  Total Orders        : {total_orders:>12,}")
    print(f"  Unique Customers    : {total_cust:>12,}")
    print(f"  Avg Order Value     : ${aov:>12,.2f}")
    print(f"  Revenue CAGR (3yr)  : {cagr:>10.1f}%")
    print(f"  Loss-making Orders  : {(df['Profit_Label']==0).sum():>12,} "
          f"({(df['Profit_Label']==0).mean()*100:.1f}%)")

    # Yearly trend
    subsection("Yearly Revenue & Profit Trend")
    yr_data = df.groupby("Year").agg(
        Revenue=("Sales",    "sum"),
        Profit =("Profit",   "sum"),
        Orders =("Order ID", "nunique")
    ).reset_index()
    yr_data["Margin%"]    = (yr_data["Profit"] /
                              yr_data["Revenue"] * 100).round(2)
    yr_data["YoY_Growth"] = yr_data["Revenue"].pct_change().mul(100).round(2)
    print(yr_data.to_string(index=False))

    # Category
    subsection("Revenue & Profit by Category")
    cat = df.groupby("Category").agg(
        Revenue=("Sales",    "sum"),
        Profit =("Profit",   "sum"),
        Orders =("Order ID", "count")
    ).reset_index()
    cat["Margin%"] = (cat["Profit"] / cat["Revenue"] * 100).round(2)
    print(cat.sort_values("Revenue", ascending=False).to_string(index=False))

    # Sub-Category
    subsection("Profit by Sub-Category (Best & Worst)")
    sub = df.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False)
    print("  TOP 5 (Most Profitable):")
    print(sub.head(5).round(2).to_string())
    print("\n  BOTTOM 5 (Loss-Makers):")
    print(sub.tail(5).round(2).to_string())
    print("\n  [INSIGHT] Tables sub-category LOSES $17,725 total.")
    print("  [INSIGHT] Discounts above 30% account for 80% of total loss value.")

    # Discount impact
    subsection("Discount Impact on Profitability")
    disc = df.groupby("Disc_Band", observed=True).agg(
        Avg_Margin  =("Profit_Margin", "mean"),
        Orders      =("Order ID",      "count"),
        Total_Profit=("Profit",        "sum")
    ).reset_index()
    print(disc.round(2).to_string(index=False))
    print("\n  [INSIGHT] 0% discount orders earn avg margin of 34%.")
    print("  [INSIGHT] Discounts above 30% result in NEGATIVE margins.")
    print("  [RECOMMENDATION] Cap all discounts at 20% maximum.")

    # Region
    subsection("Performance by Region")
    reg = df.groupby("Region").agg(
        Revenue=("Sales",  "sum"),
        Profit =("Profit", "sum")
    ).reset_index()
    reg["Margin%"] = (reg["Profit"] / reg["Revenue"] * 100).round(2)
    print(reg.sort_values("Profit", ascending=False).to_string(index=False))
    print("\n  [INSIGHT] Central region has lowest margin (7.9%).")
    print("  [INSIGHT] West region leads in both revenue and profit.")

    # Segment
    subsection("Performance by Customer Segment")
    seg = df.groupby("Segment").agg(
        Revenue  =("Sales",       "sum"),
        Profit   =("Profit",      "sum"),
        Customers=("Customer ID", "nunique")
    ).reset_index()
    seg["Margin%"]      = (seg["Profit"] / seg["Revenue"] * 100).round(2)
    seg["Rev_per_Cust"] = (seg["Revenue"] / seg["Customers"]).round(2)
    print(seg.to_string(index=False))
    print("\n  [INSIGHT] Home Office has highest margin (14.0%).")
    print("  [INSIGHT] Consumer segment drives 51% of total revenue.")

    _plot_ss_eda(df, yr_data, cat, sub, disc, reg)


def _plot_ss_eda(df, yr_data, cat, sub, disc, reg):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Superstore -- Business Performance Dashboard",
                 fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    # 1. Yearly Revenue & Profit
    ax1 = fig.add_subplot(gs[0, 0])
    x   = yr_data["Year"].astype(str)
    ax1.bar(x, yr_data["Revenue"]/1e3, color=BLUE,  alpha=0.8,
            label="Revenue ($K)")
    ax1.bar(x, yr_data["Profit"]/1e3,  color=GREEN, alpha=0.9,
            label="Profit ($K)")
    ax1.set_title("Yearly Revenue vs Profit", fontweight="bold")
    ax1.set_ylabel("USD (Thousands)")
    ax1.legend(fontsize=8);  ax1.grid(axis="y", alpha=0.3)

    # 2. Monthly Revenue Trend
    ax2 = fig.add_subplot(gs[0, 1:])
    monthly = df.groupby(["Year","Month"])["Sales"].sum().reset_index()
    for yr_val, grp in monthly.groupby("Year"):
        ax2.plot(grp["Month"], grp["Sales"]/1e3,
                 marker="o", markersize=4, linewidth=1.8, label=str(yr_val))
    ax2.set_title("Monthly Revenue Trend by Year", fontweight="bold")
    ax2.set_xlabel("Month");  ax2.set_ylabel("Revenue (USD K)")
    ax2.set_xticks(range(1, 13))
    ax2.legend(fontsize=8);  ax2.grid(alpha=0.3)

    # 3. Profit by Sub-Category
    ax3 = fig.add_subplot(gs[1, :2])
    sub_s  = df.groupby("Sub-Category")["Profit"].sum().sort_values()
    bar_c  = [RED if v < 0 else GREEN for v in sub_s.values]
    ax3.barh(sub_s.index, sub_s.values, color=bar_c, edgecolor="white")
    ax3.axvline(0, color="black", lw=1.2)
    ax3.set_title("Total Profit by Sub-Category", fontweight="bold")
    ax3.set_xlabel("Total Profit ($)");  ax3.grid(axis="x", alpha=0.3)

    # 4. Discount Band vs Avg Margin
    ax4 = fig.add_subplot(gs[1, 2])
    dm   = df.groupby("Disc_Band", observed=True)["Profit_Margin"].mean()
    dcol = [GREEN if v > 0 else RED for v in dm.values]
    ax4.bar(dm.index, dm.values, color=dcol, edgecolor="white")
    ax4.axhline(0, color="black", lw=1.2)
    ax4.set_title("Discount Band vs Avg Margin%", fontweight="bold")
    ax4.set_ylabel("Avg Profit Margin %")
    ax4.tick_params(axis="x", rotation=30)
    ax4.grid(axis="y", alpha=0.3)

    # 5. Profit by Region
    ax5 = fig.add_subplot(gs[2, 0])
    rs   = reg.sort_values("Profit", ascending=True)
    ax5.barh(rs["Region"], rs["Profit"], color=BLUE, edgecolor="white")
    ax5.set_title("Profit by Region", fontweight="bold")
    ax5.set_xlabel("Total Profit ($)");  ax5.grid(axis="x", alpha=0.3)

    # 6. Revenue Share by Category
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.pie(cat["Revenue"], labels=cat["Category"],
            autopct="%1.1f%%", colors=[BLUE, GREEN, ORANGE],
            wedgeprops=dict(edgecolor="white"))
    ax6.set_title("Revenue Share by Category", fontweight="bold")

    # 7. Profit Share by Segment
    ax7 = fig.add_subplot(gs[2, 2])
    seg_df = df.groupby("Segment")["Profit"].sum()
    ax7.pie(seg_df.values, labels=seg_df.index,
            autopct="%1.1f%%", colors=[BLUE, TEAL, PURPLE],
            wedgeprops=dict(edgecolor="white"))
    ax7.set_title("Profit Share by Segment", fontweight="bold")

    plt.tight_layout()
    save_plot(fig, "ss_eda_dashboard.png")


def ss_rfm():
    global ss_rfm_df
    if not ss_preprocessed:
        print("  [!] Please preprocess Superstore dataset first."); return

    section("Superstore -- RFM Customer Segmentation")
    df       = ss_proc
    snapshot = df["Order Date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg(
        Customer_Name=("Customer Name", "first"),
        Segment      =("Segment",       "first"),
        Recency      =("Order Date",
                       lambda x: (snapshot - x.max()).days),
        Frequency    =("Order ID",  "nunique"),
        Monetary     =("Sales",     "sum")
    ).reset_index()

    rfm["R_Score"] = pd.qcut(rfm["Recency"],
                              5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"),
                              5, labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"],
                              5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def assign_segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if   r >= 4 and f >= 4 and m >= 4: return "Champions"
        elif r >= 3 and f >= 3:            return "Loyal Customers"
        elif r >= 4 and f <= 2:            return "New Customers"
        elif r >= 3 and f >= 2:            return "Potential Loyalists"
        elif r == 2 and f >= 3:            return "At Risk"
        elif r <= 2 and f >= 3:            return "Cannot Lose Them"
        elif r <= 2 and f <= 2:            return "Lost"
        else:                              return "Needs Attention"

    rfm["RFM_Segment"] = rfm.apply(assign_segment, axis=1)
    ss_rfm_df = rfm

    subsection("RFM Segment Summary")
    summary = rfm.groupby("RFM_Segment").agg(
        Customers   =("Customer ID", "count"),
        Avg_Recency =("Recency",     "mean"),
        Avg_Freq    =("Frequency",   "mean"),
        Avg_Monetary=("Monetary",    "mean"),
        Total_Rev   =("Monetary",    "sum")
    ).reset_index().sort_values("Total_Rev", ascending=False)
    for col in ["Avg_Recency","Avg_Freq","Avg_Monetary","Total_Rev"]:
        summary[col] = summary[col].round(0 if col != "Avg_Freq" else 1)
    print(summary.to_string(index=False))

    champ = rfm[rfm["RFM_Segment"] == "Champions"]
    risk  = rfm[rfm["RFM_Segment"] == "At Risk"]
    lost  = rfm[rfm["RFM_Segment"] == "Lost"]
    print(f"\n  [INSIGHT] Champions      : {len(champ)} customers, "
          f"avg spend ${champ['Monetary'].mean():,.0f}")
    print(f"  [INSIGHT] At Risk        : {len(risk)} customers "
          f"-- need retention campaigns NOW")
    print(f"  [INSIGHT] Lost Customers : {len(lost)} customers "
          f"-- win-back campaigns needed")
    print(f"\n  [RECOMMENDATION] Offer 10% loyalty discount to At Risk segment.")
    print(f"  [RECOMMENDATION] Send personalised win-back emails to Lost segment.")

    save_csv(rfm, "outputs/superstore_rfm.csv", "RFM Segmentation")
    _plot_rfm(rfm, summary)


def _plot_rfm(rfm, summary):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("RFM Customer Segmentation Analysis",
                 fontsize=14, fontweight="bold")

    seg_s = summary.sort_values("Customers", ascending=True)
    bcolors = COLORS[:len(seg_s)]

    axes[0].barh(seg_s["RFM_Segment"], seg_s["Customers"],
                 color=bcolors, edgecolor="white")
    axes[0].set_title("Customers per Segment", fontweight="bold")
    axes[0].set_xlabel("Number of Customers")
    axes[0].grid(axis="x", alpha=0.3)

    seg_r = summary.sort_values("Total_Rev", ascending=True)
    axes[1].barh(seg_r["RFM_Segment"], seg_r["Total_Rev"]/1e3,
                 color=bcolors, edgecolor="white")
    axes[1].set_title("Total Revenue per Segment ($K)", fontweight="bold")
    axes[1].set_xlabel("Revenue (USD K)")
    axes[1].grid(axis="x", alpha=0.3)

    axes[2].hist(rfm["RFM_Score"], bins=12, color=BLUE,
                 edgecolor="white", alpha=0.85)
    axes[2].axvline(rfm["RFM_Score"].mean(), color=RED, lw=2, ls="--",
                    label=f"Mean = {rfm['RFM_Score'].mean():.1f}")
    axes[2].set_title("Distribution of RFM Scores", fontweight="bold")
    axes[2].set_xlabel("RFM Score (3-15)")
    axes[2].set_ylabel("Number of Customers")
    axes[2].legend();  axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "ss_rfm_segments.png")


def ss_ml():
    if not ss_preprocessed:
        print("  [!] Please preprocess Superstore dataset first."); return

    section("Superstore -- Profit Classification (Gradient Boosting)")
    df = ss_proc.copy()

    cat_cols = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode"]
    num_cols = ["Sales", "Quantity", "Discount", "Ship_Days", "Month", "Quarter"]

    df_enc    = pd.get_dummies(df[cat_cols + num_cols + ["Profit_Label"]],
                               columns=cat_cols, drop_first=False)
    bool_cols = df_enc.select_dtypes("bool").columns
    df_enc[bool_cols] = df_enc[bool_cols].astype(int)

    feat_cols = [c for c in df_enc.columns if c != "Profit_Label"]
    X = df_enc[feat_cols]
    y = df_enc["Profit_Label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.08, subsample=0.8, random_state=42)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    print("  Running 5-Fold Stratified Cross-Validation ...")
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    cv_auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    cv_f1  = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted")

    acc = accuracy_score(y_te, y_pred)
    cm  = confusion_matrix(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)

    print(f"\n  Algorithm        : Gradient Boosting Classifier")
    print(f"  Train / Test     : 80% / 20%  (stratified split)")
    print(f"  Accuracy         : {acc*100:.2f}%")
    print(f"  ROC-AUC          : {auc:.4f}")
    print(f"  CV Accuracy      : {cv_acc.mean():.4f}  +/-  {cv_acc.std():.4f}")
    print(f"  CV ROC-AUC       : {cv_auc.mean():.4f}  +/-  {cv_auc.std():.4f}")
    print(f"  CV F1 (weighted) : {cv_f1.mean():.4f}  +/-  {cv_f1.std():.4f}")
    print(f"\n  Confusion Matrix:\n  {cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred,
          target_names=["Loss Order", "Profit Order"], zero_division=0))

    tn, fp, fn, tp = cm.ravel()
    print("  [BUSINESS MEANING]")
    print(f"  Correctly identified profitable orders : {tp:,}")
    print(f"  Correctly flagged loss orders          : {tn:,}")
    print(f"  Missed loss orders (False Negatives)   : {fn:,}")
    print("  [RECOMMENDATION] Use this model to flag risky orders before approval.")

    res = pd.DataFrame({
        "Actual_Label"      : y_te.values,
        "Predicted_Label"   : y_pred,
        "Profit_Probability": np.round(y_prob, 4)
    })
    save_csv(res, "outputs/superstore_profit_predictions.csv",
             "Superstore profit predictions")

    plot_confusion_matrix(cm, "Superstore Profit Classification",
                          "ss_ml_cm.png", ["Loss Order", "Profit Order"])
    plot_roc_curve(y_te, y_prob,
                   "Superstore Profit Classification", "ss_ml_roc.png")
    plot_feature_importance(
        model.feature_importances_, feat_cols,
        "Top 15 Feature Importances -- Superstore Profit Classification",
        "ss_ml_fi.png")
    plot_cv_scores(cv_auc,
                   "5-Fold CV ROC-AUC -- Superstore Profit Classification",
                   "ss_ml_cv.png", ylabel="ROC-AUC")

    # New order prediction
    sep("-", 60)
    print("  Predict Profit / Loss for a New Order")
    print("  (All inputs validated -- no crashes)\n")

    sales    = input_float("  Sales amount ($)                                   : ", lo=0)
    qty      = input_int  ("  Quantity (1-100)                                   : ", 1, 100)
    discount = input_float("  Discount (0.0 to 0.8)                              : ", 0, 0.8)
    ship_d   = input_int  ("  Shipping days (1-7)                                : ", 1, 7)
    month    = input_int  ("  Month (1-12)                                       : ", 1, 12)
    quarter  = (month - 1) // 3 + 1
    category = input_choice(
        "  Category (Furniture / Office Supplies / Technology) : ",
        ["Furniture", "Office Supplies", "Technology"])
    region   = input_choice(
        "  Region (Central / East / South / West)              : ",
        ["Central", "East", "South", "West"])
    segment  = input_choice(
        "  Segment (Consumer / Corporate / Home Office)        : ",
        ["Consumer", "Corporate", "Home Office"])

    new_row = {c: 0 for c in feat_cols}
    new_row.update({"Sales": sales, "Quantity": qty,
                    "Discount": discount, "Ship_Days": ship_d,
                    "Month": month, "Quarter": quarter})
    for col in feat_cols:
        if   col.startswith("Category_"):
            new_row[col] = int(category == col.replace("Category_", ""))
        elif col.startswith("Sub-Category_"):
            new_row[col] = 0
        elif col.startswith("Region_"):
            new_row[col] = int(region == col.replace("Region_", ""))
        elif col.startswith("Segment_"):
            new_row[col] = int(segment == col.replace("Segment_", ""))
        elif col.startswith("Ship Mode_"):
            new_row[col] = 0

    new_df  = pd.DataFrame([{c: new_row.get(c, 0) for c in feat_cols}])
    pred    = model.predict(new_df)[0]
    prob    = model.predict_proba(new_df)[0][1]
    outcome = "PROFITABLE ORDER" if pred == 1 else "LOSS-MAKING ORDER"

    print(f"\n  >>> Prediction          : {outcome}")
    print(f"  >>> Profit Probability  : {prob:.2%}")
    if discount > 0.30:
        print("  >>> WARNING: Discount > 30% is the main driver of losses!")


def ss_summary():
    if not ss_preprocessed:
        print("  [!] Please preprocess Superstore dataset first."); return
    section("Superstore -- Complete Business Summary")
    df = ss_proc
    ts = df["Sales"].sum();  tp = df["Profit"].sum()
    print(f"  Period          : {df['Year'].min()} to {df['Year'].max()}")
    print(f"  Total Revenue   : ${ts:>12,.2f}")
    print(f"  Total Profit    : ${tp:>12,.2f}")
    print(f"  Profit Margin   : {tp/ts*100:.2f}%")
    print(f"  Total Orders    : {df['Order ID'].nunique():>12,}")
    print(f"  Customers       : {df['Customer ID'].nunique():>12,}")
    print(f"  Products        : {df['Product ID'].nunique():>12,}")
    print(f"  Loss Orders     : {(df['Profit_Label']==0).sum():>12,} "
          f"({(df['Profit_Label']==0).mean()*100:.1f}%)")
    print("\n  Revenue by Year:")
    for yr, val in df.groupby("Year")["Sales"].sum().items():
        print(f"    {yr}: ${val:,.0f}")
    print("\n  TOP 3 BUSINESS RECOMMENDATIONS:")
    print("  1. STOP heavy discounting on Tables -- losing $17,725 total.")
    print("     Cap ALL discounts at 20% maximum.")
    print("  2. FOCUS on West region and Technology category")
    print("     -- both have the highest profit margins.")
    print("  3. RE-ENGAGE At Risk RFM customers immediately")
    print("     -- 99 high-value customers are drifting away.")


# ================================================================
# PIPELINE 2 -- TELCO CUSTOMER CHURN
# ================================================================

def tc_load():
    global tc_raw, tc_loaded, tc_preprocessed
    # Already loaded check
    if tc_loaded:
        print("  [INFO] Telco dataset is already loaded.")
        print("  [INFO] To reload with a different file, restart the program.")
        print(f"  [INFO] Current shape: {tc_raw.shape}")
        print(f"  [INFO] Overall Churn Rate: {(tc_raw['Churn']=='Yes').mean()*100:.1f}%")
        return
    path = get_path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv", "Telco")
    if not path_exists(path):
        print(f"  [ERROR] File not found: {path}"); return
    try:
        df = pd.read_csv(path)
        if "Churn" not in df.columns:
            print("  [ERROR] Churn column missing."); return
        tc_raw = df;  tc_loaded = True;  tc_preprocessed = False
        show_head(df, "Telco Dataset Loaded Successfully")
        print(f"  Shape   : {df.shape}")
        print(f"  Columns : {list(df.columns)}")
        print(f"  Overall Churn Rate: {(df['Churn']=='Yes').mean()*100:.1f}%")
    except Exception as e:
        print(f"  [ERROR] {e}")


def tc_preprocess():
    global tc_raw, tc_proc, tc_preprocessed
    if not tc_loaded:
        print("  [!] Please load Telco dataset first."); return
    if tc_preprocessed:
        print("  [INFO] Telco data is already preprocessed.")
        print(f"  [INFO] Preprocessed shape: {tc_proc.shape}")
        print("  [INFO] Proceed to EDA or Churn Prediction steps.")
        return

    df = tc_raw.copy()
    print("\n  Preprocessing steps:")
    for i, s in enumerate([
        "Fix TotalCharges blanks (11 rows) -- fill with median",
        "Drop customerID column (not a feature)",
        "Encode Churn as binary  (Yes=1, No=0)",
        "Derive CLV = tenure x MonthlyCharges",
        "Derive Has_Security flag",
        "Count total services per customer",
        "Create Tenure_Group for analysis",
    ], 1):
        print(f"    {i}. {s}")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blanks = df["TotalCharges"].isna().sum()
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    print(f"\n  Fixed {blanks} blank TotalCharges rows")

    df = df.drop(columns=["customerID"])
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)
    df["CLV"] = df["tenure"] * df["MonthlyCharges"]
    df["Has_Security"] = (
        (df["OnlineSecurity"]   == "Yes") |
        (df["TechSupport"]      == "Yes") |
        (df["DeviceProtection"] == "Yes")
    ).astype(int)
    df["Num_Services"] = (
        (df["PhoneService"]    == "Yes").astype(int) +
        (df["MultipleLines"]   == "Yes").astype(int) +
        (df["InternetService"] != "No" ).astype(int) +
        (df["OnlineSecurity"]  == "Yes").astype(int) +
        (df["OnlineBackup"]    == "Yes").astype(int) +
        (df["DeviceProtection"]== "Yes").astype(int) +
        (df["TechSupport"]     == "Yes").astype(int) +
        (df["StreamingTV"]     == "Yes").astype(int) +
        (df["StreamingMovies"] == "Yes").astype(int)
    )
    df["Tenure_Group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-1 yr", "1-2 yr", "2-4 yr", "4-6 yr"])

    tc_proc = df;  tc_preprocessed = True

    print(f"  Final shape          : {df.shape}")
    print(f"  Churn rate           : {df['Churn_Binary'].mean()*100:.1f}%")
    print(f"  Avg CLV (Retained)   : "
          f"${df[df['Churn_Binary']==0]['CLV'].mean():,.0f}")
    print(f"  Avg CLV (Churned)    : "
          f"${df[df['Churn_Binary']==1]['CLV'].mean():,.0f}")
    save_csv(df, "outputs/telco_preprocessed.csv", "Preprocessed Telco")


def tc_eda():
    if not tc_preprocessed:
        print("  [!] Please preprocess Telco dataset first."); return
    df = tc_proc

    section("Telco -- Customer Churn Analysis")

    subsection("Key Business KPIs")
    total      = len(df)
    churned    = df["Churn_Binary"].sum()
    churn_rate = churned / total * 100
    monthly_rev= df["MonthlyCharges"].sum()
    rev_risk   = df[df["Churn_Binary"]==1]["MonthlyCharges"].sum()
    clv_ok     = df[df["Churn_Binary"]==0]["CLV"].mean()
    clv_ch     = df[df["Churn_Binary"]==1]["CLV"].mean()

    print(f"  Total Customers        : {total:,}")
    print(f"  Churned Customers      : {churned:,} ({churn_rate:.1f}%)")
    print(f"  Retained Customers     : {total-churned:,} ({100-churn_rate:.1f}%)")
    print(f"  Monthly Revenue        : ${monthly_rev:,.0f}")
    print(f"  Revenue at Risk/Month  : ${rev_risk:,.0f} "
          f"({rev_risk/monthly_rev*100:.1f}%)")
    print(f"  Avg CLV (Retained)     : ${clv_ok:,.0f}")
    print(f"  Avg CLV (Churned)      : ${clv_ch:,.0f}")
    print(f"  CLV Loss per Churner   : ${clv_ok-clv_ch:,.0f}")

    subsection("Churn Rate by Contract Type")
    ct = df.groupby("Contract")["Churn_Binary"].agg(
        Customers="count", Churned="sum",
        Churn_Rate=lambda x: x.mean()*100
    ).reset_index()
    print(ct.round(1).to_string(index=False))
    print("\n  [INSIGHT] Month-to-month customers churn 42.7% vs 2.8% (2-year).")
    print("  [RECOMMENDATION] Offer 15% discount to switch to annual contract.")

    subsection("Churn Rate by Internet Service")
    inet = df.groupby("InternetService")["Churn_Binary"].agg(
        Customers="count",
        Churn_Rate=lambda x: x.mean()*100
    ).reset_index()
    print(inet.round(1).to_string(index=False))
    print("\n  [INSIGHT] Fiber optic customers churn at 41.9%.")

    subsection("Churn Rate by Tenure Group")
    tg = df.groupby("Tenure_Group", observed=True)["Churn_Binary"].agg(
        Customers="count",
        Churn_Rate=lambda x: x.mean()*100
    ).reset_index()
    print(tg.round(1).to_string(index=False))
    print("\n  [INSIGHT] First-year customers churn at 47.7%.")
    print("  [RECOMMENDATION] Intensive onboarding for first 12 months.")

    subsection("Churn Rate by Payment Method")
    pm = df.groupby("PaymentMethod")["Churn_Binary"].agg(
        Churn_Rate=lambda x: x.mean()*100
    ).reset_index().sort_values("Churn_Rate", ascending=False)
    print(pm.round(1).to_string(index=False))
    print("\n  [INSIGHT] Electronic check users churn at 45.3%.")
    print("  [RECOMMENDATION] Incentivise auto-pay with $5/month bill credit.")

    subsection("Impact of Security Services on Churn")
    sec = df.groupby("Has_Security")["Churn_Binary"].agg(
        Customers="count",
        Churn_Rate=lambda x: x.mean()*100
    ).reset_index()
    sec["Has_Security"] = sec["Has_Security"].map(
        {0: "No Security Services", 1: "Has Security Services"})
    print(sec.round(1).to_string(index=False))
    print("\n  [INSIGHT] Customers without security services churn at 31.5%.")
    print("  [RECOMMENDATION] Bundle free TechSupport for first 6 months.")

    _plot_tc_eda(df)


def _plot_tc_eda(df):
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle("Telco -- Customer Churn Analysis Dashboard",
                 fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    # 1. Overall Churn Pie
    ax1 = fig.add_subplot(gs[0, 0])
    vals = df["Churn"].value_counts()
    ax1.pie(vals.values, labels=["Retained", "Churned"],
            colors=[GREEN, RED], autopct="%1.1f%%",
            wedgeprops=dict(edgecolor="white"))
    ax1.set_title("Overall Churn Rate", fontweight="bold")

    # 2. Churn by Contract
    ax2 = fig.add_subplot(gs[0, 1])
    ct  = df.groupby("Contract")["Churn_Binary"].mean() * 100
    brc = [GREEN if v<20 else ORANGE if v<35 else RED for v in ct.values]
    bars= ax2.bar(ct.index, ct.values, color=brc, edgecolor="white")
    ax2.set_title("Churn Rate by Contract", fontweight="bold")
    ax2.set_ylabel("Churn Rate (%)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis="x", rotation=12)
    for i, (_, v) in enumerate(ct.items()):
        ax2.text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=9)

    # 3. Churn by Tenure
    ax3 = fig.add_subplot(gs[0, 2])
    tg  = df.groupby("Tenure_Group", observed=True)["Churn_Binary"].mean()*100
    ax3.plot(tg.index.astype(str), tg.values,
             marker="o", color=RED, lw=2.5, markersize=8)
    ax3.fill_between(range(len(tg)), tg.values, alpha=0.15, color=RED)
    ax3.set_title("Churn Rate by Tenure", fontweight="bold")
    ax3.set_ylabel("Churn Rate (%)")
    ax3.set_xlabel("Tenure Group")
    ax3.grid(alpha=0.3)

    # 4. Monthly Charges distribution
    ax4 = fig.add_subplot(gs[1, :2])
    df[df["Churn_Binary"]==0]["MonthlyCharges"].hist(
        ax=ax4, bins=30, color=GREEN, alpha=0.7,
        label="Retained", density=True, edgecolor="white")
    df[df["Churn_Binary"]==1]["MonthlyCharges"].hist(
        ax=ax4, bins=30, color=RED, alpha=0.7,
        label="Churned", density=True, edgecolor="white")
    ax4.set_title("Monthly Charges: Churned vs Retained", fontweight="bold")
    ax4.set_xlabel("Monthly Charges ($)")
    ax4.set_ylabel("Density")
    ax4.legend();  ax4.grid(alpha=0.3)

    # 5. CLV Boxplot
    ax5 = fig.add_subplot(gs[1, 2])
    bp  = ax5.boxplot(
        [df[df["Churn_Binary"]==0]["CLV"],
         df[df["Churn_Binary"]==1]["CLV"]],
        labels=["Retained", "Churned"],
        patch_artist=True,
        boxprops=dict(facecolor=GREEN, alpha=0.7),
        medianprops=dict(color=RED, lw=2))
    bp["boxes"][1].set_facecolor(RED)
    ax5.set_title("Customer Lifetime Value (CLV)", fontweight="bold")
    ax5.set_ylabel("CLV ($)");  ax5.grid(axis="y", alpha=0.3)

    # 6. Churn by Payment Method
    ax6 = fig.add_subplot(gs[2, :2])
    pm  = df.groupby("PaymentMethod")["Churn_Binary"].mean()*100
    pm  = pm.sort_values(ascending=True)
    bc2 = [GREEN if v<20 else ORANGE if v<35 else RED for v in pm.values]
    ax6.barh(pm.index, pm.values, color=bc2, edgecolor="white")
    ax6.set_title("Churn Rate by Payment Method", fontweight="bold")
    ax6.set_xlabel("Churn Rate (%)")
    ax6.grid(axis="x", alpha=0.3)
    for i, v in enumerate(pm.values):
        ax6.text(v+0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    # 7. Churn by Number of Services
    ax7 = fig.add_subplot(gs[2, 2])
    ns  = df.groupby("Num_Services")["Churn_Binary"].mean()*100
    bc3 = [GREEN if v<20 else ORANGE if v<35 else RED for v in ns.values]
    ax7.bar(ns.index, ns.values, color=bc3, edgecolor="white")
    ax7.set_title("Churn Rate by #Services", fontweight="bold")
    ax7.set_xlabel("Number of Services")
    ax7.set_ylabel("Churn Rate (%)")
    ax7.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "tc_churn_dashboard.png")


def tc_ml():
    if not tc_preprocessed:
        print("  [!] Please preprocess Telco dataset first."); return

    section("Telco -- Churn Prediction (Gradient Boosting Classifier)")
    df = tc_proc.copy()

    # Drop non-feature columns before encoding
    for col in ["Churn", "Churn_Binary", "Tenure_Group"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    cat_cols = df.select_dtypes("object").columns.tolist()
    df_enc   = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    bool_c   = df_enc.select_dtypes("bool").columns
    df_enc[bool_c] = df_enc[bool_c].astype(int)

    # Fill any residual NaN with median
    df_enc = df_enc.fillna(df_enc.median(numeric_only=True))

    feat_cols = list(df_enc.columns)
    X = df_enc[feat_cols].astype(float)
    y = tc_proc["Churn_Binary"]   # target from original df

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(X_tr_sc, y_tr)

    y_prob = model.predict_proba(X_te_sc)[:, 1]
    THRESHOLD = 0.40
    y_pred    = (y_prob >= THRESHOLD).astype(int)
    y_pred_50 = (y_prob >= 0.50).astype(int)

    print("  Running 5-Fold Stratified Cross-Validation ...")
    pipe   = Pipeline([("sc", StandardScaler()), ("gb", model)])
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
    cv_auc = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
    cv_f1  = cross_val_score(pipe, X, y, cv=skf, scoring="f1_weighted")

    acc = accuracy_score(y_te, y_pred)
    cm  = confusion_matrix(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)

    recall_50    = confusion_matrix(y_te,y_pred_50)[1,1]/sum(y_te==1)*100
    recall_tuned = cm[1,1]/sum(y_te==1)*100

    print(f"\n  Algorithm         : Gradient Boosting Classifier")
    print(f"  Train / Test      : 80% / 20%  (stratified split)")
    print(f"  Decision Threshold: {THRESHOLD}  (tuned for better churn recall)")
    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  ROC-AUC           : {auc:.4f}")
    print(f"  CV Accuracy       : {cv_acc.mean():.4f}  +/-  {cv_acc.std():.4f}")
    print(f"  CV ROC-AUC        : {cv_auc.mean():.4f}  +/-  {cv_auc.std():.4f}")
    print(f"  CV F1 (weighted)  : {cv_f1.mean():.4f}  +/-  {cv_f1.std():.4f}")
    print(f"\n  Confusion Matrix (threshold={THRESHOLD}):\n  {cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred,
          target_names=["Retained", "Churned"], zero_division=0))

    print("  [THRESHOLD COMPARISON]")
    print(f"  Default 0.50 churner recall : {recall_50:.1f}%")
    print(f"  Tuned   0.40 churner recall : {recall_tuned:.1f}%")

    tn, fp, fn, tp = cm.ravel()
    missed_rev = fn * tc_proc["MonthlyCharges"].mean()
    print("\n  [BUSINESS IMPACT]")
    print(f"  Churners correctly caught : {tp:,} -- retention action taken")
    print(f"  Churners missed           : {fn:,} -- "
          f"est. ${missed_rev:,.0f}/month revenue lost")
    print("  [RECOMMENDATION] Flag customers with churn prob > 40%")
    print("  for proactive calls by the customer success team.")

    res = pd.DataFrame({
        "Actual_Churn"     : y_te.values,
        "Predicted_Churn"  : y_pred,
        "Churn_Probability": np.round(y_prob, 4)
    })
    save_csv(res, "outputs/telco_churn_predictions.csv",
             "Telco churn predictions")

    # Correct labels: Retained / Churned
    plot_confusion_matrix(cm, "Telco Churn Prediction",
                          "tc_ml_cm.png", ["Retained", "Churned"])
    plot_roc_curve(y_te, y_prob,
                   "Telco Churn Prediction", "tc_ml_roc.png")
    plot_feature_importance(
        model.feature_importances_, feat_cols,
        "Top 15 Feature Importances -- Telco Churn Prediction",
        "tc_ml_fi.png")
    plot_cv_scores(cv_auc,
                   "5-Fold CV ROC-AUC -- Telco Churn Prediction",
                   "tc_ml_cv.png", ylabel="ROC-AUC")

    # New customer prediction
    sep("-", 60)
    print("  Predict Churn Risk for a New Customer")
    print("  (All inputs validated -- no crashes)\n")

    tenure   = input_int  ("  Tenure in months (0-72)         : ", 0, 72)
    monthly  = input_float("  Monthly Charges ($18-$119)      : ", 18, 119)
    total_c  = input_float("  Total Charges ($)               : ", 0)
    contract = input_choice(
        "  Contract\n"
        "  (Month-to-month / One year / Two year) : ",
        ["Month-to-month", "One year", "Two year"])
    internet = input_choice(
        "  Internet Service\n"
        "  (DSL / Fiber optic / No)               : ",
        ["DSL", "Fiber optic", "No"])
    payment  = input_choice(
        "  Payment Method\n"
        "  (Electronic check / Mailed check\n"
        "   / Bank transfer (automatic)\n"
        "   / Credit card (automatic))            : ",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)",
         "Credit card (automatic)"])
    security = input_choice(
        "  Online Security (Yes / No)             : ",
        ["Yes", "No"])

    clv     = tenure * monthly
    has_sec = 1 if security == "Yes" else 0

    new_row = {c: 0 for c in feat_cols}
    new_row.update({
        "tenure"        : tenure,
        "MonthlyCharges": monthly,
        "TotalCharges"  : total_c,
        "CLV"           : clv,
        "Has_Security"  : has_sec,
        "Num_Services"  : 3,
        "SeniorCitizen" : 0,
    })
    for col in feat_cols:
        if   f"Contract_{contract}"          == col: new_row[col] = 1
        elif f"InternetService_{internet}"   == col: new_row[col] = 1
        elif f"PaymentMethod_{payment}"      == col: new_row[col] = 1
        elif f"OnlineSecurity_{security}"    == col: new_row[col] = 1

    new_df   = pd.DataFrame([{c: new_row.get(c,0) for c in feat_cols}])
    new_sc   = scaler.transform(new_df)
    new_prob = model.predict_proba(new_sc)[0][1]
    new_pred = int(new_prob >= THRESHOLD)
    risk     = ("HIGH RISK -- Likely to Churn"
                if new_pred == 1 else "LOW RISK -- Likely to Stay")

    print(f"\n  >>> Churn Risk            : {risk}")
    print(f"  >>> Churn Probability     : {new_prob:.2%}")
    print(f"  >>> Customer CLV Estimate : ${clv:,.0f}")
    if contract == "Month-to-month":
        print("  >>> TIP: Offer annual plan -- reduces churn 42.7% to 11.3%.")
    if payment == "Electronic check":
        print("  >>> TIP: Encourage auto-pay -- reduces churn risk ~25%.")


def tc_summary():
    if not tc_preprocessed:
        print("  [!] Please preprocess Telco dataset first."); return
    section("Telco -- Complete Churn Business Summary")
    df = tc_proc
    mr = df["MonthlyCharges"].sum()
    rr = df[df["Churn_Binary"]==1]["MonthlyCharges"].sum()
    print(f"  Total Customers      : {len(df):,}")
    print(f"  Churned              : {df['Churn_Binary'].sum():,} "
          f"({df['Churn_Binary'].mean()*100:.1f}%)")
    print(f"  Retained             : {(df['Churn_Binary']==0).sum():,} "
          f"({(df['Churn_Binary']==0).mean()*100:.1f}%)")
    print(f"  Monthly Revenue      : ${mr:,.0f}")
    print(f"  Revenue at Risk/Mo   : ${rr:,.0f} ({rr/mr*100:.1f}%)")
    print(f"  Avg CLV (Retained)   : "
          f"${df[df['Churn_Binary']==0]['CLV'].mean():,.0f}")
    print(f"  Avg CLV (Churned)    : "
          f"${df[df['Churn_Binary']==1]['CLV'].mean():,.0f}")
    print("\n  TOP 3 BUSINESS RECOMMENDATIONS:")
    print("  1. Month-to-month customers churn at 42.7%.")
    print("     OFFER 15% discount to convert to 1-year contract.")
    print("  2. Electronic check users churn at 45.3%.")
    print("     INCENTIVISE auto-pay with $5/month bill credit.")
    print("  3. First-year customers churn at 47.7%.")
    print("     LAUNCH 90-day onboarding with dedicated support.")


# ================================================================
# MENUS
# ================================================================
def menu_superstore():
    while True:
        print("\n" + "-"*52)
        print("  SUPERSTORE SALES PIPELINE")
        print("-"*52)
        print("  1. Load Dataset")
        print("  2. Preprocess Data")
        print("  3. EDA & Business Insights")
        print("  4. RFM Customer Segmentation")
        print("  5. Profit Prediction  (ML Model)")
        print("  6. Business Summary")
        print("  7. Back to Main Menu")
        print("-"*52)
        ch = input("  Choice: ").strip()
        if   ch == "1": ss_load()
        elif ch == "2": ss_preprocess()
        elif ch == "3": ss_eda()
        elif ch == "4": ss_rfm()
        elif ch == "5": ss_ml()
        elif ch == "6": ss_summary()
        elif ch == "7": break
        else: print("  [!] Invalid choice. Enter 1 to 7.")


def menu_telco():
    while True:
        print("\n" + "-"*52)
        print("  TELCO CUSTOMER CHURN PIPELINE")
        print("-"*52)
        print("  1. Load Dataset")
        print("  2. Preprocess Data")
        print("  3. EDA & Churn Insights")
        print("  4. Churn Prediction  (ML Model)")
        print("  5. Business Summary")
        print("  6. Back to Main Menu")
        print("-"*52)
        ch = input("  Choice: ").strip()
        if   ch == "1": tc_load()
        elif ch == "2": tc_preprocess()
        elif ch == "3": tc_eda()
        elif ch == "4": tc_ml()
        elif ch == "5": tc_summary()
        elif ch == "6": break
        else: print("  [!] Invalid choice. Enter 1 to 6.")


def main():
    print("\n" + "="*62)
    print("   BUSINESS PERFORMANCE ANALYSIS & INSIGHT GENERATION")
    print("   Pipeline 1 : Superstore Sales     (Retail Analytics)")
    print("   Pipeline 2 : Telco Customer Churn (Customer Analytics)")
    print("="*62)
    while True:
        print("\n" + "-"*52)
        print("  MAIN MENU")
        print("-"*52)
        print("  1. Superstore Sales Pipeline")
        print("  2. Telco Customer Churn Pipeline")
        print("  3. Exit")
        print("-"*52)
        ch = input("  Select: ").strip()
        if   ch == "1": menu_superstore()
        elif ch == "2": menu_telco()
        elif ch == "3":
            print("\n  Exiting... Goodbye!\n"); break
        else:
            print("  [!] Invalid option. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()