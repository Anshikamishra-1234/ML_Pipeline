import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Studio",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --surface2:  #21262d;
    --border:    #30363d;
    --primary:   #00b4d8;
    --primary2:  #48cae4;
    --accent:    #ff6b35;
    --accent2:   #ffd166;
    --accent3:   #06d6a0;
    --text:      #f0f6fc;
    --text-muted:#8b949e;
    --success:   #06d6a0;
    --mono:      'Fira Code', monospace;
    --sans:      'Outfit', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem 2.5rem !important; max-width: 1350px; }

.hero-wrap {
    background: linear-gradient(135deg, #0d1117 0%, #0a1628 40%, #0d2137 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--primary);
    border-radius: 0 0 16px 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hero-title { font-size: 1.5rem; font-weight: 800; color: var(--text); margin: 0; letter-spacing: -0.02em; }
.hero-title span { color: var(--primary); }
.hero-sub { color: var(--text-muted); font-size: 0.82rem; margin: 0.15rem 0 0; }

.pipe-nav {
    display: flex; align-items: center; gap: 0;
    overflow-x: auto; padding: 0.5rem 0 1.2rem; scrollbar-width: none;
}
.pipe-nav::-webkit-scrollbar { display: none; }
.pn-step {
    display: flex; align-items: center; gap: 0.4rem;
    padding: 0.4rem 0.9rem; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--text-muted); font-size: 0.78rem; font-weight: 500; white-space: nowrap;
}
.pn-step.done  { border-color: var(--accent3); color: var(--accent3); background: rgba(6,214,160,0.07); }
.pn-step.active{ border-color: var(--primary); color: #0d1117; background: var(--primary); font-weight: 700; box-shadow: 0 0 18px rgba(0,180,216,0.4); }
.pn-arrow { color: var(--border); font-size: 0.9rem; padding: 0 0.2rem; }

.s-badge {
    display: inline-flex; align-items: center;
    background: var(--surface2); border: 1px solid var(--primary);
    color: var(--primary); font-family: var(--mono); font-size: 0.72rem;
    font-weight: 600; padding: 0.22rem 0.75rem; border-radius: 6px;
    margin-bottom: 0.5rem; letter-spacing: 0.06em;
}
.s-title { font-size: 1.6rem; font-weight: 800; color: var(--text); margin-bottom: 1.4rem; letter-spacing: -0.02em; }

.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem; }
.card-accent {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 100%);
    border: 1.5px solid var(--primary); border-radius: 12px;
    padding: 2rem; text-align: center; margin-bottom: 1rem;
}
.big-metric { font-family: var(--mono); font-size: 3.5rem; font-weight: 700; color: var(--primary); line-height: 1; }
.big-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-muted); margin-top: 0.4rem; }

.sg  { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.9rem; margin: 1.2rem 0; }
.sg3 { grid-template-columns: repeat(3,1fr); }
.sg2 { grid-template-columns: repeat(2,1fr); }
.sb  { background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: 1.1rem; text-align: center; }
.sn  { font-family: var(--mono); font-size: 1.8rem; font-weight: 700; }
.ss  { font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); margin-top: 0.2rem; }

.ok-box { background: rgba(6,214,160,0.07); border: 1.5px solid var(--success); border-radius: 12px; padding: 1.4rem; margin-top: 1rem; }
.ok-title { font-family: var(--mono); font-size: 1rem; font-weight: 600; color: var(--success); }

.prog-wrap { background: var(--surface2); border-radius: 999px; height: 6px; margin: 0.5rem 0 0; overflow: hidden; width: 160px; }
.prog-bar  { background: linear-gradient(90deg, var(--primary), var(--accent3)); height: 100%; border-radius: 999px; }

.stButton > button {
    background: var(--primary) !important; color: #0d1117 !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 0.88rem !important;
    padding: 0.55rem 1.5rem !important; font-family: var(--sans) !important;
    box-shadow: 0 3px 12px rgba(0,180,216,0.3) !important; transition: all 0.2s !important;
}
.stButton > button:hover { background: var(--primary2) !important; transform: translateY(-1px) !important; }
div[data-testid="stRadio"] label { color: var(--text) !important; }
div[data-testid="stSelectbox"] > div > div { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; color: var(--text) !important; }
div[data-testid="stMultiSelect"] > div { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
div[data-testid="stFileUploader"] { background: var(--surface2) !important; border: 1.5px dashed var(--border) !important; border-radius: 10px !important; }
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; font-family: var(--sans) !important; }
.stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom: 2px solid var(--primary) !important; }
.stCheckbox label span { color: var(--text) !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
STEPS = ["Problem","Data","EDA","Engineering","Features","Split","Model","Training","K-Fold Validation","Metrics","Prediction","Tuning"]
ICONS = ["🎯","📂","🔬","⚙️","🧬","✂️","🤖","🏋️","📊","📈","🔮","🚀"]

defaults = dict(
    step=0, problem_type="Regression",
    df=None, target_col=None, feature_cols=[],
    final_features=[], test_size=0.2, random_state=42,
    model_name="Linear Regression", model=None,
    X_train=None, X_test=None, y_train=None, y_test=None,
    y_pred=None, r2=None, rmse=None, mae=None,
    pipeline_done=False, outlier_mask=None,
)
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def nav(d): st.session_state.step = max(0, min(len(STEPS)-1, st.session_state.step+d))

def hero():
    pct = int((st.session_state.step/(len(STEPS)-1))*100)
    st.markdown(f"""
    <div class="hero-wrap">
      <div style="display:flex;align-items:center;gap:1rem">
        <span style="font-size:2.2rem">⚗️</span>
        <div>
          <p class="hero-title">Auto<span>ML</span> Studio</p>
          <p class="hero-sub">Interactive end-to-end machine learning pipeline builder</p>
        </div>
      </div>
      <div style="font-family:var(--mono);font-size:0.75rem;color:var(--text-muted);text-align:right">
        <b style="color:var(--accent2)">Step {st.session_state.step+1} / {len(STEPS)}</b><br>
        {ICONS[st.session_state.step]} {STEPS[st.session_state.step]}<br>
        <div class="prog-wrap" style="margin-left:auto">
          <div class="prog-bar" style="width:{pct}%"></div>
        </div>
        <span style="font-size:0.68rem">{pct}% complete</span>
      </div>
    </div>""", unsafe_allow_html=True)

def pipe_nav():
    cur = st.session_state.step
    parts = []
    for i,(s,ic) in enumerate(zip(STEPS,ICONS)):
        cls = "pn-step active" if i==cur else ("pn-step done" if i<cur else "pn-step")
        parts.append(f'<div class="{cls}">{ic} {s}</div>')
        if i < len(STEPS)-1: parts.append('<span class="pn-arrow">›</span>')
    st.markdown(f'<div class="pipe-nav">{"".join(parts)}</div>', unsafe_allow_html=True)

def step_hdr(num, icon, title):
    st.markdown(f'<div class="s-badge">// STEP {num:02d}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="s-title">{icon} {title}</div>', unsafe_allow_html=True)

def nav_btns(back=True, fwd=True, fwd_label="Next Step →", back_label="← Back", pre_fn=None):
    st.markdown("---")
    c1,_,c3 = st.columns([1,4,1])
    if back:
        with c1:
            if st.button(back_label, key=f"bk{st.session_state.step}"): nav(-1); st.rerun()
    if fwd:
        with c3:
            if st.button(fwd_label, key=f"fw{st.session_state.step}"):
                if pre_fn: pre_fn()
                nav(1); st.rerun()

def dplot(fig, h=400):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,27,34,0.9)",
        font=dict(color="#f0f6fc", family="Outfit"),
        xaxis=dict(gridcolor="#30363d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#30363d", linecolor="#30363d"),
        height=h,
    )
    return fig

# ─────────────────────────────────────────────
# STEP RENDERERS
# ─────────────────────────────────────────────
def step_problem():
    step_hdr(1,"🎯","Define Your ML Problem")
    col1,col2 = st.columns([2,1])
    with col1:
        st.markdown("##### What kind of problem are you working on?")
        ptype = st.radio("Problem type", ["Regression","Classification"],
                         index=0 if st.session_state.problem_type=="Regression" else 1, horizontal=True)
        st.session_state.problem_type = ptype
        info = {
            "Regression":     ("Predict a **continuous numeric value**","e.g. house price, sales revenue","#00b4d8","📉"),
            "Classification": ("Predict a **category or class label**","e.g. spam detection, disease diagnosis","#ff6b35","🏷️"),
        }
        desc,ex,clr,ic = info[ptype]
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {clr};margin-top:1rem">
            <div style="font-size:1.8rem;margin-bottom:0.5rem">{ic}</div>
            <div style="font-weight:700;color:{clr};font-size:1rem">{ptype}</div>
            <div style="color:#f0f6fc;margin:0.3rem 0">{desc}</div>
            <div style="color:#8b949e;font-size:0.82rem">{ex}</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Confirm & Continue →"): nav(1); st.rerun()
    with col2:
        clr = "#00b4d8" if ptype=="Regression" else "#ff6b35"
        ic  = "📉" if ptype=="Regression" else "🏷️"
        st.markdown(f"""
        <div style="border:2px solid {clr};border-radius:14px;padding:2.5rem 1.5rem;
             text-align:center;background:{'rgba(0,180,216,0.07)' if ptype=='Regression' else 'rgba(255,107,53,0.07)'};
             margin-top:0.5rem">
            <div style="font-size:4rem">{ic}</div>
            <div style="font-family:'Fira Code',monospace;font-size:1.1rem;font-weight:600;color:{clr};margin:0.8rem 0 0.2rem">{ptype}</div>
            <div style="font-size:0.72rem;letter-spacing:0.12em;color:#8b949e">SELECTED MODE</div>
        </div>""", unsafe_allow_html=True)


def step_data():
    step_hdr(2,"📂","Load Your Dataset")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"], help="CSV up to 200 MB")

    if uploaded:
        df = pd.read_csv(uploaded)

        # CLEAN DATASET
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # RESET STATE (VERY IMPORTANT)
        st.session_state.df = df
        st.session_state.target_col = None
        st.session_state.feature_cols = []

        st.success(f"✅ Dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

    df = st.session_state.df

    if df is None:
        st.info("⬆️ Upload a CSV file to get started.")
        nav_btns(back=True,fwd=False)
        return

    st.markdown("---")

    c1,c2 = st.columns(2)

    with c1:
        st.markdown("##### 🎯 Target Column")

        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) == 0:
            st.error("No numeric columns found!")
            return

        # SAFE RESET
        if st.session_state.target_col not in num_cols:
            st.session_state.target_col = num_cols[0]

        target = st.selectbox(
            "Select target feature:",
            num_cols,
            index=num_cols.index(st.session_state.target_col)
        )

        st.session_state.target_col = target

    with c2:
        st.markdown("##### 🧬 Input Features")

        feat_opts = [c for c in num_cols if c != target]

        # KEEP ONLY VALID FEATURES
        valid_features = [
            f for f in st.session_state.feature_cols
            if f in feat_opts
        ]

        feats = st.multiselect(
            "Select features:",
            feat_opts,
            default=valid_features if valid_features else feat_opts
        )

        st.session_state.feature_cols = feats

    # PCA (unchanged logic, just safe)
    if feats and len(feats) >= 2:
        st.markdown("---")
        st.markdown("##### 📊 PCA Projection")

        t2,t3 = st.tabs(["2D View","3D View"])

        X = df[feats].dropna()

        if len(X) > 0:
            y = df.loc[X.index, target]

            Xs = StandardScaler().fit_transform(X)
            n  = min(3,len(feats))

            pca = PCA(n_components=n)
            comps = pca.fit_transform(Xs)
            ev = pca.explained_variance_ratio_

            with t2:
                fig = px.scatter(
                    x=comps[:,0],
                    y=comps[:,1],
                    color=y,
                    labels={
                        "x":f"PC1 ({ev[0]*100:.1f}%)",
                        "y":f"PC2 ({ev[1]*100:.1f}%)",
                        "color":target
                    }
                )
                st.plotly_chart(dplot(fig), use_container_width=True)

            if n == 3:
                with t3:
                    fig3 = px.scatter_3d(
                        x=comps[:,0],
                        y=comps[:,1],
                        z=comps[:,2],
                        color=y
                    )
                    st.plotly_chart(dplot(fig3,500), use_container_width=True)

    nav_btns(back=True, fwd=bool(feats), fwd_label="Go to EDA →")


def step_eda():
    step_hdr(3,"🔬","Exploratory Data Analysis")
    df = st.session_state.df
    if df is None: st.warning("No dataset loaded."); nav_btns(); return

    t1,t2,t3 = st.tabs(["📋 Statistics","🔗 Correlation","❓ Missing Data"])
    num_df = df.select_dtypes(include=np.number)
    with t1: st.dataframe(df.describe().T.round(4), use_container_width=True)
    with t2:
        fig = px.imshow(num_df.corr(), color_continuous_scale=["#ff6b35","#0d1117","#00b4d8"],
                        zmin=-1,zmax=1,aspect="auto",title="Correlation Heatmap")
        st.plotly_chart(dplot(fig,450), use_container_width=True)
    with t3:
        miss = df.isnull().sum()
        st.dataframe(pd.DataFrame({"Column":miss.index,"Missing":miss.values,"Pct%":(miss/len(df)*100).round(2).values}), use_container_width=True)
        if miss.sum()==0: st.success("🎉 Zero missing values — your dataset is clean!")

    nr,nf = df.shape; nn = num_df.shape[1]; nm = df.isnull().sum().sum()
    st.markdown(f"""
    <div class="sg">
      <div class="sb"><div class="sn" style="color:#00b4d8">{nr:,}</div><div class="ss">Rows</div></div>
      <div class="sb"><div class="sn" style="color:#ffd166">{nf}</div><div class="ss">Columns</div></div>
      <div class="sb"><div class="sn" style="color:#06d6a0">{nn}</div><div class="ss">Numeric</div></div>
      <div class="sb"><div class="sn" style="color:{'#ff4444' if nm else '#06d6a0'}">{nm}</div><div class="ss">Missing</div></div>
    </div>""", unsafe_allow_html=True)
    nav_btns(fwd_label="Go to Engineering →")


def step_engineering():
    step_hdr(4,"⚙️","Data Engineering")
    df = st.session_state.df
    if df is None: st.warning("No data."); nav_btns(); return

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### 💧 Handle Missing Values")
        method = st.selectbox("Strategy:", ["Mean","Median","Mode","Drop Rows"], key="eng_imp")
        cols   = st.multiselect("Apply to columns:", df.columns.tolist(), default=df.columns.tolist(), key="eng_cols")
        if st.button("Apply Imputation"):
            for c in cols:
                if df[c].isnull().sum()>0:
                    if method=="Mean":       df[c].fillna(df[c].mean(),inplace=True)
                    elif method=="Median":   df[c].fillna(df[c].median(),inplace=True)
                    elif method=="Mode":     df[c].fillna(df[c].mode()[0],inplace=True)
                    else:                    df.dropna(subset=[c],inplace=True)
            st.session_state.df=df; st.success("✅ Imputation applied!")

    with c2:
        st.markdown("#### 🎯 Detect Outliers")
        out_m = st.selectbox("Method:", ["IQR","Z-Score"], key="eng_out")
        out_f = st.multiselect("Features:", df.select_dtypes(include=np.number).columns.tolist(),
                                default=df.select_dtypes(include=np.number).columns.tolist()[:4], key="eng_of")
        if st.button("Run Outlier Detection"):
            mask = pd.Series([False]*len(df),index=df.index)
            for c in out_f:
                if out_m=="IQR":
                    Q1,Q3=df[c].quantile(0.25),df[c].quantile(0.75); IQR=Q3-Q1
                    mask |= (df[c]<Q1-1.5*IQR)|(df[c]>Q3+1.5*IQR)
                else:
                    from scipy import stats
                    z=np.abs(stats.zscore(df[c].fillna(df[c].mean())))
                    mask |= pd.Series(z,index=df.index)>3
            st.session_state.outlier_mask=mask; n_out=mask.sum()
            if out_f:
                xc,yc=out_f[0],out_f[min(1,len(out_f)-1)]
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=df[~mask][xc],y=df[~mask][yc],mode="markers",
                                          marker=dict(color="#00b4d8",size=5,opacity=0.7),name="Normal"))
                fig.add_trace(go.Scatter(x=df[mask][xc],y=df[mask][yc],mode="markers",
                                          marker=dict(color="#ff6b35",size=9,symbol="x"),name="Outlier"))
                fig.update_layout(title=f"Outlier Map — {n_out} outliers")
                st.plotly_chart(dplot(fig,350), use_container_width=True)
                st.info(f"Found **{n_out}** outliers ({n_out/len(df)*100:.1f}%)")

    if st.checkbox("🗑️ Remove detected outliers"):
        if st.session_state.outlier_mask is not None:
            df=df[~st.session_state.outlier_mask].reset_index(drop=True)
            st.session_state.df=df; st.success(f"Done. {len(df)} rows remain.")

    nav_btns(fwd_label="Go to Feature Selection →")


def step_features():
    step_hdr(5,"🧬","Feature Selection")
    df = st.session_state.df
    if df is None or not st.session_state.target_col: st.warning("Missing data/target."); nav_btns(); return

    num_cols=[c for c in df.select_dtypes(include=np.number).columns if c!=st.session_state.target_col]
    t1,t2,t3 = st.tabs(["📐 Variance Filter","🔗 Correlation Filter","ℹ️ Mutual Info"])
    with t1:
        thresh=st.slider("Drop variance below:",0.0,1.0,0.01,0.01)
        if st.button("Apply Variance Filter"):
            kept=df[num_cols].var()[df[num_cols].var()>thresh].index.tolist()
            st.session_state.final_features=kept; st.success(f"Kept {len(kept)}/{len(num_cols)} features")
    with t2:
        ct=st.slider("Drop correlated above:",0.5,1.0,0.95,0.01)
        if st.button("Apply Correlation Filter"):
            cm=df[num_cols].corr().abs(); up=cm.where(np.triu(np.ones(cm.shape),k=1).astype(bool))
            drop=[c for c in up.columns if any(up[c]>ct)]
            kept=[c for c in num_cols if c not in drop]
            st.session_state.final_features=kept; st.success(f"Kept {len(kept)} features")
    with t3:
        if st.button("Calculate Mutual Information"):
            mi=mutual_info_regression(df[num_cols].fillna(0),df[st.session_state.target_col])
            mi_df=pd.DataFrame({"Feature":num_cols,"Score":mi}).sort_values("Score",ascending=False)
            fig=px.bar(mi_df,x="Score",y="Feature",orientation="h",color="Score",
                       color_continuous_scale="teal",title="Mutual Information Scores")
            st.plotly_chart(dplot(fig,400), use_container_width=True)

    if not st.session_state.final_features: st.session_state.final_features=num_cols
    st.markdown("##### ✅ Final Feature Set")
    final=st.multiselect("Adjust features:",num_cols,default=st.session_state.final_features,key="ff_ms")
    st.session_state.final_features=final
    nav_btns(fwd_label="Go to Split →")


def step_split():
    step_hdr(6,"✂️","Train / Test Split")
    df=st.session_state.df
    if df is None: st.warning("No data."); nav_btns(); return

    c1,c2=st.columns(2)
    with c1:
        ts=st.slider("Test size:",0.1,0.5,st.session_state.test_size,0.05); st.session_state.test_size=ts
        rs=st.number_input("Random seed:",0,9999,st.session_state.random_state); st.session_state.random_state=rs

    features=st.session_state.final_features or [c for c in df.select_dtypes(include=np.number).columns if c!=st.session_state.target_col]
    target=st.session_state.target_col
    if features and target:
        X=df[features].fillna(0); y=df[target]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=ts,random_state=rs)
        st.session_state.X_train=Xtr; st.session_state.X_test=Xte
        st.session_state.y_train=ytr; st.session_state.y_test=yte
        with c2:
            fig=px.pie(names=["Training","Testing"],values=[len(Xtr),len(Xte)],hole=0.55,
                       color_discrete_sequence=["#00b4d8","#ff6b35"],title="Data Split")
            st.plotly_chart(dplot(fig,300), use_container_width=True)
        st.markdown(f"""
        <div class="sg">
          <div class="sb"><div class="sn" style="color:#00b4d8">{len(Xtr)}</div><div class="ss">Train</div></div>
          <div class="sb"><div class="sn" style="color:#ff6b35">{len(Xte)}</div><div class="ss">Test</div></div>
          <div class="sb"><div class="sn" style="color:#06d6a0">{len(features)}</div><div class="ss">Features</div></div>
          <div class="sb"><div class="sn" style="color:#ffd166">{int((1-ts)*100)}%</div><div class="ss">Train %</div></div>
        </div>""", unsafe_allow_html=True)
    nav_btns(fwd_label="Go to Model →")


def step_model():
    step_hdr(7,"🤖","Model Selection")
    models=(["Linear Regression","SVM (Kernel)","Random Forest","Decision Tree","Gradient Boosting"]
            if st.session_state.problem_type=="Regression"
            else ["Logistic Regression","SVM (Kernel)","Random Forest","Decision Tree","Gradient Boosting"])
    sel=st.radio("Choose algorithm:",models,
                 index=models.index(st.session_state.model_name) if st.session_state.model_name in models else 0)
    st.session_state.model_name=sel
    tips={
        "Linear Regression":  ("Simple and fast","Works well when relationships are linear","📉"),
        "SVM (Kernel)":       ("Support Vector Machine","Handles non-linear data with kernel trick","🔷"),
        "Random Forest":      ("Ensemble of trees","Robust, handles overfitting well","🌲"),
        "Decision Tree":      ("Single tree","Easy to interpret and visualize","🌿"),
        "Gradient Boosting":  ("Sequential boosting","Often gives best accuracy","🚀"),
        "Logistic Regression":("Linear classifier","Fast, interpretable baseline","📊"),
    }
    n,d,ic=tips.get(sel,(sel,"","🤖"))
    st.markdown(f"""
    <div class="card" style="border-left:4px solid #00b4d8;margin-top:1rem">
        <div style="display:flex;align-items:center;gap:0.8rem">
            <span style="font-size:2.5rem">{ic}</span>
            <div>
                <div style="font-weight:700;color:#00b4d8;font-size:1rem">{n}</div>
                <div style="color:#f0f6fc;font-size:0.88rem;margin-top:0.2rem">{d}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    nav_btns(fwd_label="Go to Training →")


def step_training():
    step_hdr(8,"🏋️","Train the Model")
    Xtr=st.session_state.X_train
    if Xtr is None: st.warning("Complete the Split step first."); nav_btns(); return

    name=st.session_state.model_name
    st.markdown(f"""
    <div class="card">
        <div style="color:#8b949e;font-size:0.8rem;margin-bottom:0.6rem;letter-spacing:0.08em">TRAINING SUMMARY</div>
        <div>Algorithm: <b style="color:#00b4d8">{name}</b></div>
        <div>Training rows: <b>{len(Xtr):,}</b> &nbsp;|&nbsp; Features: <b>{Xtr.shape[1]}</b> &nbsp;|&nbsp; Test rows: <b>{len(st.session_state.X_test):,}</b></div>
    </div>""", unsafe_allow_html=True)

    model_map={
        "Linear Regression":  LinearRegression(),
        "SVM (Kernel)":       SVR(),
        "Random Forest":      RandomForestRegressor(n_estimators=100,random_state=42),
        "Decision Tree":      DecisionTreeRegressor(random_state=42),
        "Gradient Boosting":  GradientBoostingRegressor(random_state=42),
    }
    if st.button("🏋️ Start Training"):
        model=model_map.get(name,LinearRegression())
        with st.spinner("Training in progress..."):
            model.fit(Xtr,st.session_state.y_train)
            yp=model.predict(st.session_state.X_test)
            st.session_state.model=model; st.session_state.y_pred=yp
            st.session_state.r2  =r2_score(st.session_state.y_test,yp)
            st.session_state.rmse=np.sqrt(mean_squared_error(st.session_state.y_test,yp))
            st.session_state.mae =mean_absolute_error(st.session_state.y_test,yp)
        st.success("✅ Training complete!")
        st.markdown(f"""
<div class="sg sg3">
  <div class="sb"><div class="sn" style="color:#06d6a0">{st.session_state.r2:.4f}</div><div class="ss">R² Score</div></div>
  <div class="sb"><div class="sn" style="color:#ffd166">{st.session_state.rmse:.4f}</div><div class="ss">RMSE</div></div>
  <div class="sb"><div class="sn" style="color:#ff6b35">{st.session_state.mae:.4f}</div><div class="ss">MAE</div></div>
</div>""", unsafe_allow_html=True)


# 🔥 MODEL COMPARISON (ADD HERE)
    if st.checkbox("Compare with other models"):

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor()
        }

        results = {}

        for name, m in models.items():
            m.fit(st.session_state.X_train, st.session_state.y_train)
            preds = m.predict(st.session_state.X_test)
            results[name] = r2_score(st.session_state.y_test, preds)

        comp_df = pd.DataFrame(results.items(), columns=["Model", "R2 Score"])

        st.markdown("### 📊 Model Comparison")
        st.dataframe(comp_df)

        fig = px.bar(comp_df, x="Model", y="R2 Score", color="Model")
        st.plotly_chart(dplot(fig), use_container_width=True)


    # 👇 KEEP THIS LAST
    nav_btns(fwd_label="Go to Validation →")


from sklearn.model_selection import cross_val_score

def step_validation():
    step_hdr(9,"📊","K-Fold Validation")

    if st.session_state.X_train is None:
        st.warning("Complete Training step first.")
        nav_btns()
        return

    X = st.session_state.X_train
    y = st.session_state.y_train

    model = st.session_state.model

    if model is None:
        st.warning("Train the model first.")
        nav_btns()
        return

    st.markdown("""
    <div class="card">
        <b>What is happening?</b><br>
        <span style="color:#8b949e;font-size:0.88rem">
        K-Fold splits the training data into multiple parts and evaluates the model multiple times 
        to give a more reliable performance estimate.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Select K
    k = st.slider("Select number of folds (K)", 2, 10, 5)

    # Run validation
    if st.button("Run K-Fold Validation"):
        with st.spinner("Running cross-validation..."):

            scores = cross_val_score(model, X, y, cv=k)

            mean_score = scores.mean()
            std_score = scores.std()

        # Display results
        st.markdown(f"""
        <div class="sg sg3">
          <div class="sb"><div class="sn" style="color:#06d6a0">{mean_score:.4f}</div><div class="ss">Mean Score</div></div>
          <div class="sb"><div class="sn" style="color:#ffd166">{std_score:.4f}</div><div class="ss">Std Dev</div></div>
          <div class="sb"><div class="sn" style="color:#00b4d8">{k}</div><div class="ss">Folds</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Plot scores
        fig = px.line(
            x=list(range(1, k+1)),
            y=scores,
            markers=True,
            title="K-Fold Scores per Fold"
        )
        st.plotly_chart(dplot(fig), use_container_width=True)

    nav_btns(fwd_label="Go to Metrics →")


def step_metrics():
    step_hdr(9,"📈","Performance Metrics")
    if st.session_state.model is None: st.warning("Train your model first."); nav_btns(); return

    yt=st.session_state.y_test; yp=st.session_state.y_pred
    st.markdown(f"""
    <div class="card-accent">
        <div class="big-metric">{st.session_state.r2:.4f}</div>
        <div class="big-label">R² Score (Coefficient of Determination)</div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    c1.metric("RMSE",f"{st.session_state.rmse:.4f}"); c2.metric("MAE",f"{st.session_state.mae:.4f}"); c3.metric("Test Samples",len(yt))

    col1,col2=st.columns(2)
    with col1:
        mn,mx=float(yt.min()),float(yt.max())
        fig=px.scatter(x=yt,y=yp,labels={"x":"Actual","y":"Predicted"},
                       title="Actual vs Predicted",opacity=0.7,color_discrete_sequence=["#00b4d8"])
        fig.add_shape(type="line",x0=mn,y0=mn,x1=mx,y1=mx,line=dict(color="#ff6b35",dash="dash",width=2))
        st.plotly_chart(dplot(fig,380), use_container_width=True)
    with col2:
        res=np.array(yt)-np.array(yp)
        fig2=px.histogram(res,nbins=30,title="Residuals Distribution",color_discrete_sequence=["#06d6a0"])
        fig2.add_vline(x=0,line_color="#ff6b35",line_dash="dash",line_width=2)
        st.plotly_chart(dplot(fig2,380), use_container_width=True)

    # 🔥 FEATURE IMPORTANCE
    if hasattr(st.session_state.model, "feature_importances_"):
        imp = st.session_state.model.feature_importances_
        feat_names = st.session_state.final_features

        imp_df = pd.DataFrame({
            "Feature": feat_names,
            "Importance": imp
        }).sort_values(by="Importance", ascending=False)

        st.markdown("### 📊 Feature Importance")
        st.dataframe(imp_df)

        fig = px.bar(imp_df, x="Feature", y="Importance", color="Importance")
        st.plotly_chart(dplot(fig), use_container_width=True)

        # 🔥 Insight (very important)
        top_feature = imp_df.iloc[0]["Feature"]
        st.success(f"📌 Most influential feature: {top_feature}")

            

    nav_btns(fwd_label="Go to Prediction →")

def step_prediction():
    step_hdr(11,"🔮","Make Prediction")

    if st.session_state.model is None:
        st.warning("Train model first.")
        nav_btns()
        return

    features = st.session_state.final_features

    input_data = {}

    st.markdown("### Enter feature values")

    for f in features:
        input_data[f] = st.number_input(f"{f}")

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        pred = st.session_state.model.predict(input_df)

        st.success(f"Prediction: {round(pred[0],2)}")

    nav_btns(back=True, fwd=True, fwd_label="Go to Tuning →")


def step_tuning():
    step_hdr(10,"🚀","Hyperparameter Tuning & Finalise")
    st.markdown("""
    <div class="card">
        <b>What happens here?</b><br>
        <span style="color:#8b949e;font-size:0.88rem">
        Grid search tries multiple hyperparameter combinations and picks the best one,
        potentially improving your model's accuracy beyond the defaults.
        </span>
    </div>""", unsafe_allow_html=True)

    if st.button("🚀 Run Hyperparameter Tuning"):
        Xtr,ytr=st.session_state.X_train,st.session_state.y_train
        Xte,yte=st.session_state.X_test,st.session_state.y_test
        name=st.session_state.model_name
        grids={
            "Random Forest":    (RandomForestRegressor(random_state=42),{"n_estimators":[50,100],"max_depth":[None,5,10]}),
            "Gradient Boosting":(GradientBoostingRegressor(random_state=42),{"n_estimators":[50,100],"learning_rate":[0.05,0.1]}),
            "SVM (Kernel)":     (SVR(),{"C":[0.1,1,10],"kernel":["rbf","linear"]}),
        }
        with st.spinner("Running grid search..."):
            if name in grids:
                est,params=grids[name]; gs=GridSearchCV(est,params,cv=3,scoring="r2",n_jobs=-1)
                gs.fit(Xtr,ytr); tuned=gs.best_estimator_
                yp_t=tuned.predict(Xte); r2_t=r2_score(yte,yp_t)
                st.info(f"Best params: `{gs.best_params_}`")
                delta=r2_t-st.session_state.r2
                cc1,cc2=st.columns(2)
                cc1.metric("Base R²",f"{st.session_state.r2:.4f}")
                cc2.metric("Tuned R²",f"{r2_t:.4f}",delta=f"{delta:+.4f}")
            else:
                st.info(f"No tuning grid for {name}. Base model is already optimal.")
        st.session_state.pipeline_done=True

    if st.session_state.pipeline_done:
        st.markdown("""
        <div class="ok-box" style="margin-top:1.5rem">
            <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.7rem">
                <span style="background:#06d6a0;color:#0d1117;padding:0.25rem 0.7rem;border-radius:6px;
                             font-size:0.72rem;font-weight:700;letter-spacing:0.08em">SUCCESS</span>
                <span class="ok-title">✅ PIPELINE COMPLETE</span>
            </div>
            <p style="color:#8b949e;margin:0;font-size:0.9rem">
                Your AutoML pipeline has finished. The model is trained, evaluated, and optimised.
            </p>
        </div>""", unsafe_allow_html=True)
        if st.button("🔄 Start Over"):
            for k,v in defaults.items(): st.session_state[k]=v
            st.rerun()
    nav_btns(back=True,fwd=False)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
hero()
pipe_nav()
st.markdown('<hr style="margin:0 0 1.8rem">', unsafe_allow_html=True)

RENDERERS=[step_problem,step_data,step_eda,step_engineering,
           step_features,step_split,step_model,step_training,
           step_validation,step_metrics,step_prediction,step_tuning]

RENDERERS[st.session_state.step]()
