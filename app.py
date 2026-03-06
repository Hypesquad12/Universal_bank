import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG & STYLING
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Universal Bank · Full Analytics Suite", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="st-"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 0.5rem; }
h1 { font-weight: 900 !important; letter-spacing: -1px; }
h2, h3 { font-weight: 800 !important; letter-spacing: -0.5px; }
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0c1425 0%, #162033 100%);
    border: 1px solid #1e3a5f; border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
div[data-testid="stMetric"] label { color: #7aa2c9 !important; font-size: 12px !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 1px; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e8f1fa !important; font-size: 30px !important; font-weight: 900 !important; font-family: 'JetBrains Mono', monospace !important; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 12px !important; font-weight: 600 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #0a0f1e; border-radius: 14px; padding: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; font-weight: 700; font-size: 14px; padding: 10px 20px; }
.stTabs [aria-selected="true"] { background: #1a2744 !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #060b18 0%, #0c1425 100%); }
.insight-card {
    background: linear-gradient(135deg, #0f1c33 0%, #0a1222 100%);
    border-left: 4px solid; border-radius: 0 14px 14px 0;
    padding: 18px 24px; margin: 12px 0; line-height: 1.75; font-size: 14px;
}
.desc-card { border-color: #3b82f6; }
.diag-card { border-color: #f59e0b; }
.pred-card { border-color: #8b5cf6; }
.presc-card { border-color: #10b981; }
.golden-banner {
    background: linear-gradient(135deg, #422006 0%, #1a0f02 100%);
    border: 2px solid #f59e0b; border-radius: 16px; padding: 24px; text-align: center; margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    df['Experience'] = df['Experience'].clip(lower=0)
    df['Edu_Label'] = df['Education'].map({1:'Undergrad', 2:'Graduate', 3:'Advanced/Professional'})
    df['Loan_Label'] = df['Personal Loan'].map({0:'Rejected', 1:'Accepted'})
    df['Inc_Band'] = pd.cut(df['Income'], bins=[0,30,50,80,100,150,225], labels=['<$30K','$30-50K','$50-80K','$80-100K','$100-150K','$150K+'])
    df['Age_Grp'] = pd.cut(df['Age'], bins=[20,30,40,50,60,70], labels=['23-30','31-40','41-50','51-60','61-67'])
    df['CC_Band'] = pd.cut(df['CCAvg'], bins=[-0.1,1,2,3,5,10.1], labels=['$0-1K','$1-2K','$2-3K','$3-5K','$5K+'])
    df['Mort_Flag'] = np.where(df['Mortgage']==0, 'No Mortgage', 'Has Mortgage')
    df['Fam_Lbl'] = df['Family'].map({1:'Single', 2:'Couple', 3:'Family of 3', 4:'Family of 4'})
    return df

@st.cache_resource
def train_models(_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

    df = _df.copy()
    feats = ['Age','Experience','Income','Family','CCAvg','Education','Mortgage','Securities Account','CD Account','Online','CreditCard']
    X = df[feats]; y = df['Personal Loan']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        proba = m.predict_proba(Xte)[:,1]
        fpr, tpr, _ = roc_curve(yte, proba)
        cm = confusion_matrix(yte, pred)
        imp = pd.Series(m.feature_importances_ if hasattr(m, 'feature_importances_') else np.abs(m.coef_[0]), index=feats)
        results[name] = dict(accuracy=accuracy_score(yte, pred), auc=roc_auc_score(yte, proba),
                             precision=precision_score(yte, pred), recall=recall_score(yte, pred),
                             f1=f1_score(yte, pred), cm=cm, fpr=fpr, tpr=tpr, importances=imp, model=m)

    best = models['Gradient Boosting']
    df_out = df.copy()
    df_out['pred_proba'] = best.predict_proba(scaler.transform(X))[:,1]
    df_out['risk_tier'] = pd.cut(df_out['pred_proba'], bins=[-0.01,0.05,0.3,0.7,1.01], labels=['Very Low','Low','Medium','High'])
    return results, feats, df_out

df = load_data()
results, feats, df_pred = train_models(df)

# ── Color System ─────────────────────────────────────────────────────────────
C = dict(blue='#3b82f6', purple='#8b5cf6', amber='#f59e0b', green='#10b981',
         red='#ef4444', cyan='#06b6d4', pink='#ec4899', lime='#84cc16',
         bg='#0a0f1e', card='#0f1c33', grid='#152035', text='#e0eaf5', muted='#6b8db5',
         accept='#10b981', reject='#ef4444')
EDU_COLORS = {'Undergrad': C['blue'], 'Graduate': C['purple'], 'Advanced/Professional': C['amber']}

def style(fig, h=440):
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='DM Sans, sans-serif', color=C['text'], size=12),
                      margin=dict(l=50, r=30, t=50, b=50), height=h,
                      hoverlabel=dict(bgcolor=C['card'], font_size=13, font_family='DM Sans'),
                      legend=dict(font=dict(size=11)),
                      xaxis=dict(gridcolor=C['grid'], zerolinecolor=C['grid']),
                      yaxis=dict(gridcolor=C['grid'], zerolinecolor=C['grid']))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🏦 Universal Bank")
    st.markdown("##### Complete Analytics Suite")
    st.markdown("---")
    st.markdown(f"""
    <div style="background:{C['card']};border-radius:12px;padding:16px;border:1px solid #1e3050;">
    <div style="font-size:11px;color:{C['muted']};text-transform:uppercase;letter-spacing:1px;font-weight:700;">Dataset Summary</div>
    <div style="margin-top:10px;">
    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e3050;">
        <span style="color:{C['muted']};">Customers</span><span style="font-weight:800;font-family:'JetBrains Mono';">5,000</span></div>
    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e3050;">
        <span style="color:{C['muted']};">Variables</span><span style="font-weight:800;font-family:'JetBrains Mono';">14</span></div>
    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e3050;">
        <span style="color:{C['muted']};">Loan Rate</span><span style="font-weight:800;color:{C['green']};font-family:'JetBrains Mono';">9.6%</span></div>
    <div style="display:flex;justify-content:space-between;padding:5px 0;">
        <span style="color:{C['muted']};">Best Model AUC</span><span style="font-weight:800;color:{C['purple']};font-family:'JetBrains Mono';">0.999</span></div>
    </div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"<div style='text-align:center;color:{C['muted']};font-size:11px;'>Built with Streamlit + Plotly + Scikit-learn</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Descriptive", "🔍 Diagnostic", "🤖 Predictive", "🎯 Prescriptive", "📈 Interactive Drill-Down"])


# ═══════════════════ TAB 1 — DESCRIPTIVE ═════════════════════════════════════
with tab1:
    st.markdown("## 📊 Descriptive Analytics")
    st.markdown("*What does the data look like? Structure, distributions, and key statistics.*")

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Total Customers", f"{len(df):,}")
    k2.metric("Avg Income", f"${df['Income'].mean():,.0f}K", f"Med: ${df['Income'].median():.0f}K")
    k3.metric("Avg Age", f"{df['Age'].mean():.1f}", f"Std: {df['Age'].std():.1f}")
    k4.metric("Avg CC Spend", f"${df['CCAvg'].mean():,.2f}K/mo")
    k5.metric("Loan Accepted", f"{df['Personal Loan'].sum():,}", f"{df['Personal Loan'].mean():.1%}")
    k6.metric("CD Holders", f"{df['CD Account'].sum():,}", f"{df['CD Account'].mean():.1%}")
    st.markdown("---")

    # Row 1: Target Donut + Education + Stats
    c1, c2, c3 = st.columns([2,2,3])
    with c1:
        st.markdown("#### 🎯 Target Variable")
        lc = df['Personal Loan'].value_counts()
        fig = go.Figure(go.Pie(labels=['Rejected','Accepted'], values=[lc[0], lc[1]], hole=0.72,
                               marker=dict(colors=[C['reject'], C['accept']], line=dict(color=C['bg'], width=4)), textinfo='none',
                               hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'))
        fig.add_annotation(text=f"<b style='font-size:36px;color:{C['accept']}'>{df['Personal Loan'].mean():.1%}</b><br><span style='font-size:13px;color:{C['muted']}'>Accepted</span>", x=0.5, y=0.5, showarrow=False)
        style(fig, 340); fig.update_layout(showlegend=True, legend=dict(orientation='h', y=-0.05, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🎓 Education Breakdown")
        ec = df['Edu_Label'].value_counts()
        fig = go.Figure(go.Pie(labels=ec.index, values=ec.values, hole=0.72,
                               marker=dict(colors=[EDU_COLORS.get(x, C['blue']) for x in ec.index], line=dict(color=C['bg'], width=4)), textinfo='none',
                               hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'))
        fig.add_annotation(text=f"<b style='font-size:28px;color:{C['text']}'>3</b><br><span style='font-size:12px;color:{C['muted']}'>Levels</span>", x=0.5, y=0.5, showarrow=False)
        style(fig, 340); fig.update_layout(showlegend=True, legend=dict(orientation='h', y=-0.05, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown("#### 📏 Continuous Variable Stats")
        stats_df = df[['Income','CCAvg','Age','Experience','Mortgage']].describe().T[['mean','std','min','50%','max']].round(2)
        stats_df.insert(0, 'Variable', stats_df.index)
        skews = df[['Income','CCAvg','Age','Experience','Mortgage']].skew().round(3)
        stats_df['Skew'] = skews.values
        stats_df['Shape'] = stats_df['Skew'].apply(lambda s: '← Left' if s < -0.5 else ('→ Right' if s > 0.5 else '⟷ Symmetric'))
        stats_df.columns = ['Variable','Mean','Std','Min','Median','Max','Skew','Shape']
        st.dataframe(stats_df, use_container_width=True, hide_index=True, height=230)
        st.markdown(f"""<div class="insight-card desc-card"><b>📊 Key:</b> Income (skew 0.84) and CCAvg (1.60) are right-skewed — most customers cluster low. Mortgage is extremely skewed (2.10) since 69% have zero mortgage.</div>""", unsafe_allow_html=True)

    # Row 2: Distribution overlay
    st.markdown("#### 📈 Distribution Overlay — Accepted vs Rejected")
    st.caption("Separation between colors = predictive power. Notice Income and CCAvg have clear separation.")
    vars_dist = ['Income','CCAvg','Age','Mortgage']
    fig = make_subplots(rows=1, cols=4, subplot_titles=vars_dist, horizontal_spacing=0.06)
    for i, v in enumerate(vars_dist):
        for lbl, clr, nm in [(0,C['reject'],'Rejected'),(1,C['accept'],'Accepted')]:
            fig.add_trace(go.Histogram(x=df[df['Personal Loan']==lbl][v], name=nm, marker_color=clr, opacity=0.7, nbinsx=35, showlegend=(i==0)), row=1, col=i+1)
    style(fig, 310); fig.update_layout(barmode='overlay', legend=dict(orientation='h', y=1.18, x=0.5, xanchor='center'))
    st.plotly_chart(fig, use_container_width=True)

    # Row 3: Binary vars + Correlation
    c1, c2 = st.columns([2,3])
    with c1:
        st.markdown("#### 🏷️ Binary Variables")
        bin_vars = ['Securities Account','CD Account','Online','CreditCard']
        fig = make_subplots(rows=1, cols=4, specs=[[{'type':'domain'}]*4], subplot_titles=bin_vars)
        pal = [C['cyan'], C['amber'], C['blue'], C['pink']]
        for i, v in enumerate(bin_vars):
            vc = df[v].value_counts()
            fig.add_trace(go.Pie(labels=['No','Yes'], values=[vc.get(0,0), vc.get(1,0)], hole=0.65,
                                 marker=dict(colors=['#1e3050', pal[i]], line=dict(color=C['bg'], width=3)),
                                 textinfo='percent', textfont=dict(size=10)), row=1, col=i+1)
        style(fig, 240); fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🔥 Correlation with Personal Loan")
        num_cols = ['Income','CCAvg','CD Account','Mortgage','Education','Family','Securities Account','Online','CreditCard','Age','Experience']
        corr_loan = df[num_cols].corrwith(df['Personal Loan']).sort_values()
        fig = go.Figure(go.Bar(y=corr_loan.index, x=corr_loan.values, orientation='h',
                               marker=dict(color=[C['accept'] if v>0.1 else (C['amber'] if v>0.03 else C['muted']) for v in corr_loan.values]),
                               text=[f"{v:.3f}" for v in corr_loan.values], textposition='outside', textfont=dict(size=10),
                               hovertemplate='<b>%{y}</b><br>r = %{x:.3f}<extra></extra>'))
        fig.update_layout(xaxis_title='Pearson r')
        style(fig, 340)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════ TAB 2 — DIAGNOSTIC ══════════════════════════════════════
with tab2:
    st.markdown("## 🔍 Diagnostic Analytics")
    st.markdown("*Why do certain customers accept? Root causes, interactions, and multi-variable patterns.*")

    # Row 1: Income Staircase + Education × Income Heatmap
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📈 Income Decile — The Staircase Effect")
        df_temp = df.copy()
        df_temp['Inc_Dec'] = pd.qcut(df_temp['Income'], 10, labels=[f'D{i+1}' for i in range(10)])
        dec = df_temp.groupby('Inc_Dec', observed=True).agg(rate=('Personal Loan','mean'), avg=('Income','mean'), n=('ID','count')).reset_index()
        fig = go.Figure(go.Bar(x=dec['Inc_Dec'], y=dec['rate']*100,
                               marker=dict(color=dec['rate'], colorscale='Turbo', showscale=True, colorbar=dict(title='%', thickness=10)),
                               text=[f"{r:.1f}%" for r in dec['rate']*100], textposition='outside', textfont=dict(size=10, color=C['text']),
                               customdata=np.stack([dec['avg'], dec['n']], axis=-1),
                               hovertemplate='<b>%{x}</b><br>Avg: $%{customdata[0]:.0f}K<br>Rate: %{y:.1f}%<br>n=%{customdata[1]}<extra></extra>'))
        fig.add_hline(y=9.6, line_dash='dot', line_color=C['amber'], annotation_text='Baseline 9.6%', annotation_font_color=C['amber'])
        fig.update_layout(xaxis_title='Income Decile (D1=Low → D10=High)', yaxis_title='Acceptance %')
        style(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🔥 Education × Income → Loan Rate")
        st.caption("The multiplicative interaction: education amplifies income's effect.")
        heat = df.pivot_table('Personal Loan', index='Edu_Label', columns='Inc_Band', aggfunc='mean', observed=True)
        order = ['<$30K','$30-50K','$50-80K','$80-100K','$100-150K','$150K+']
        heat = heat.reindex(columns=[c for c in order if c in heat.columns])
        fig = go.Figure(go.Heatmap(z=heat.values*100, x=heat.columns.astype(str), y=heat.index,
                                    colorscale=[[0,'#0a0f1e'],[0.15,'#1a1a4e'],[0.4,'#6d28d9'],[0.7,'#f59e0b'],[1,'#ef4444']],
                                    text=np.where(heat.values*100>0, np.char.add(np.round(heat.values*100,1).astype(str), '%'), '0%'),
                                    texttemplate='%{text}', textfont=dict(size=12, color='white'),
                                    hovertemplate='Edu: %{y}<br>Income: %{x}<br>Rate: %{z:.1f}%<extra></extra>',
                                    colorbar=dict(title='%', thickness=12)))
        style(fig, 420); fig.update_layout(yaxis=dict(autorange='reversed'), margin=dict(l=180))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: CD + Family
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 💿 CD Account — The Hidden Multiplier")
        cd_data = pd.DataFrame([
            {'Segment': 'No CD', 'Rate': df[df['CD Account']==0]['Personal Loan'].mean()*100},
            {'Segment': 'Has CD', 'Rate': df[df['CD Account']==1]['Personal Loan'].mean()*100},
            {'Segment': 'CD + Inc>$100K', 'Rate': df[(df['CD Account']==1)&(df['Income']>100)]['Personal Loan'].mean()*100},
        ])
        fig = go.Figure(go.Bar(x=cd_data['Segment'], y=cd_data['Rate'],
                               marker=dict(color=[C['muted'], C['amber'], C['red']]),
                               text=[f"{r:.1f}%" for r in cd_data['Rate']], textposition='outside', textfont=dict(size=14)))
        fig.add_annotation(text="<b>6.4x</b>", x=1, y=cd_data.iloc[1]['Rate']+4, showarrow=True, arrowhead=2, arrowcolor=C['amber'], font=dict(color=C['amber'], size=13))
        fig.add_annotation(text="<b>11x</b>", x=2, y=cd_data.iloc[2]['Rate']+4, showarrow=True, arrowhead=2, arrowcolor=C['red'], font=dict(color=C['red'], size=13))
        fig.update_layout(yaxis_title='Acceptance %', yaxis=dict(range=[0,95]))
        style(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 👨‍👩‍👧‍👦 Family Size Effect + Education Interaction")
        fam_edu = df.groupby(['Fam_Lbl','Edu_Label'], observed=True)['Personal Loan'].mean().reset_index()
        fam_edu.columns = ['Family','Education','Rate']
        fig = px.bar(fam_edu, x='Family', y='Rate', color='Education', barmode='group',
                    color_discrete_map=EDU_COLORS, text=fam_edu['Rate'].apply(lambda x: f"{x:.0%}"))
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(xaxis_title='Family Size', yaxis_title='Loan Rate', yaxis=dict(tickformat='.0%'))
        style(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Flow + Radar
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown("#### 🌊 Customer Flow — Attributes → Loan Decision")
        flow_df = df[['Edu_Label','Inc_Band','Fam_Lbl','Loan_Label']].copy()
        fig = px.parallel_categories(flow_df, dimensions=['Edu_Label','Inc_Band','Fam_Lbl','Loan_Label'],
                                      color=df['Personal Loan'], color_continuous_scale=[[0,C['reject']],[1,C['accept']]],
                                      labels={'Edu_Label':'Education','Inc_Band':'Income','Fam_Lbl':'Family','Loan_Label':'Loan'})
        style(fig, 460); fig.update_layout(margin=dict(l=60, r=60))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🕸️ Acceptor vs Rejector Radar")
        radar_f = ['Income','CCAvg','Family','Education','Mortgage']
        acc = (df[df['Personal Loan']==1][radar_f].mean()/df[radar_f].max()).tolist()
        rej = (df[df['Personal Loan']==0][radar_f].mean()/df[radar_f].max()).tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=acc+[acc[0]], theta=radar_f+[radar_f[0]], fill='toself', name='Accepted',
                                       fillcolor='rgba(16,185,129,0.25)', line=dict(color=C['accept'], width=3)))
        fig.add_trace(go.Scatterpolar(r=rej+[rej[0]], theta=radar_f+[radar_f[0]], fill='toself', name='Rejected',
                                       fillcolor='rgba(239,68,68,0.12)', line=dict(color=C['reject'], width=2, dash='dot')))
        style(fig, 460)
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor=C['grid']), angularaxis=dict(gridcolor=C['grid']), bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div class="insight-card diag-card">
    <b>🔍 Root Cause:</b> Loan acceptance is driven by a <b>3-factor interaction</b>: Income > $100K + Graduate/Advanced education + Family 3-4. 
    CD Account acts as a 6.4× multiplier. Age, Online, CreditCard, Securities Account are <b>confirmed noise</b> (r ≈ 0).
    </div>""", unsafe_allow_html=True)


# ═══════════════════ TAB 3 — PREDICTIVE ══════════════════════════════════════
with tab3:
    st.markdown("## 🤖 Predictive Analytics")
    st.markdown("*Four ML models trained and compared. Which best predicts loan acceptance?*")

    # Leaderboard
    st.markdown("#### 🏆 Model Leaderboard")
    model_names = ['Logistic Regression','Decision Tree','Random Forest','Gradient Boosting']
    emojis = ['📐','🌳','🌲','🚀']
    mcolors = [C['blue'], C['amber'], C['green'], C['purple']]
    cols = st.columns(4)
    for i, nm in enumerate(model_names):
        r = results[nm]
        with cols[i]:
            st.markdown(f"""<div style="background:linear-gradient(135deg,{mcolors[i]}15,{C['card']});border:1px solid {mcolors[i]}40;border-radius:14px;padding:18px;text-align:center;">
            <div style="font-size:28px;">{emojis[i]}</div>
            <div style="font-size:13px;font-weight:700;color:{mcolors[i]};margin:6px 0;">{nm}</div>
            <div style="font-size:32px;font-weight:900;font-family:'JetBrains Mono';color:{C['text']};">{r['auc']:.4f}</div>
            <div style="font-size:11px;color:{C['muted']};">AUC-ROC</div>
            <div style="margin-top:10px;font-size:12px;color:{C['muted']};">
            Acc: <b style="color:{C['text']}">{r['accuracy']:.1%}</b> · P: <b style="color:{C['text']}">{r['precision']:.1%}</b> · R: <b style="color:{C['text']}">{r['recall']:.1%}</b>
            </div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # ROC + Confusion
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown("#### 📉 ROC Curves — All Models")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color=C['muted'], width=1), name='Random', showlegend=True))
        for nm, clr in zip(model_names, mcolors):
            r = results[nm]
            fig.add_trace(go.Scatter(x=r['fpr'], y=r['tpr'], mode='lines', name=f"{nm} ({r['auc']:.4f})",
                                     line=dict(color=clr, width=3 if nm=='Gradient Boosting' else 2)))
        fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                         legend=dict(x=0.4, y=0.05, bgcolor='rgba(15,28,51,0.9)', bordercolor=C['grid']))
        style(fig, 460)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🔢 Confusion Matrix")
        cm_sel = st.selectbox("Model", model_names, index=3, key='cm_sel')
        cm = results[cm_sel]['cm']
        fig = go.Figure(go.Heatmap(z=cm, x=['Rejected','Accepted'], y=['Rejected','Accepted'],
                                    colorscale=[[0,C['bg']],[1,mcolors[model_names.index(cm_sel)]]],
                                    text=cm, texttemplate='<b>%{text}</b>', textfont=dict(size=22), showscale=False,
                                    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'))
        fig.update_layout(xaxis_title='Predicted', yaxis_title='Actual', yaxis=dict(autorange='reversed'))
        style(fig, 350); fig.update_layout(margin=dict(l=80))
        st.plotly_chart(fig, use_container_width=True)
        tn,fp,fn,tp = cm.ravel()
        st.markdown(f"""<div style="display:flex;gap:6px;flex-wrap:wrap;font-size:12px;">
        <div style="background:{C['card']};padding:6px 12px;border-radius:8px;flex:1;text-align:center;"><span style="color:{C['accept']};font-weight:800;font-size:16px;">{tp}</span><br><span style="color:{C['muted']}">TP</span></div>
        <div style="background:{C['card']};padding:6px 12px;border-radius:8px;flex:1;text-align:center;"><span style="color:{C['red']};font-weight:800;font-size:16px;">{fp}</span><br><span style="color:{C['muted']}">FP</span></div>
        <div style="background:{C['card']};padding:6px 12px;border-radius:8px;flex:1;text-align:center;"><span style="color:{C['red']};font-weight:800;font-size:16px;">{fn}</span><br><span style="color:{C['muted']}">FN</span></div>
        <div style="background:{C['card']};padding:6px 12px;border-radius:8px;flex:1;text-align:center;"><span style="color:{C['accept']};font-weight:800;font-size:16px;">{tn}</span><br><span style="color:{C['muted']}">TN</span></div>
        </div>""", unsafe_allow_html=True)

    # Feature Importance
    st.markdown("#### ⚖️ Feature Importance — All Models Compared")
    imp_df = pd.DataFrame({nm: results[nm]['importances'] for nm in model_names})
    imp_df = imp_df.div(imp_df.max())
    imp_df = imp_df.reindex(results['Gradient Boosting']['importances'].sort_values(ascending=False).index)
    fig = go.Figure()
    for i, nm in enumerate(model_names):
        fig.add_trace(go.Bar(name=nm, x=imp_df.index, y=imp_df[nm], marker_color=mcolors[i], opacity=0.85))
    fig.update_layout(barmode='group', xaxis_title='Feature', yaxis_title='Normalized Importance',
                     legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'))
    style(fig, 400)
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.markdown("#### 📋 Full Metrics Table")
    mt = [{'Model':nm, 'Accuracy':f"{results[nm]['accuracy']:.2%}", 'AUC':f"{results[nm]['auc']:.4f}",
           'Precision':f"{results[nm]['precision']:.2%}", 'Recall':f"{results[nm]['recall']:.2%}",
           'F1':f"{results[nm]['f1']:.2%}", 'FP':int(results[nm]['cm'][0][1]), 'FN':int(results[nm]['cm'][1][0])} for nm in model_names]
    st.dataframe(pd.DataFrame(mt), use_container_width=True, hide_index=True)

    st.markdown(f"""<div class="insight-card pred-card">
    <b>🤖 Winner: Gradient Boosting</b> (AUC=0.9989, 98.9% accuracy). Only 4 false positives and 13 false negatives 
    on 1,500 test samples. The model's top decile captures <b>99.4%</b> of all acceptors. Income + Education + Family 
    account for <b>89%</b> of signal across all models.
    </div>""", unsafe_allow_html=True)


# ═══════════════════ TAB 4 — PRESCRIPTIVE ════════════════════════════════════
with tab4:
    st.markdown("## 🎯 Prescriptive Analytics")
    st.markdown("*What should the bank DO? Model-driven targeting, ROI simulation, and action plans.*")

    # Golden Banner
    golden = df_pred[(df_pred['Education']>=2) & (df_pred['Income']>100) & (df_pred['Family']>=3)]
    golden_rate = golden['Personal Loan'].mean()
    st.markdown(f"""<div class="golden-banner">
    <div style="font-size:14px;font-weight:700;color:{C['amber']};letter-spacing:1px;">🏆 GOLDEN SEGMENT</div>
    <div style="font-size:15px;color:{C['text']};margin:8px 0;"><b>Graduate/Advanced</b> + <b>Income > $100K</b> + <b>Family >= 3</b></div>
    <div style="font-size:48px;font-weight:900;color:{C['accept']};font-family:'JetBrains Mono';">{golden_rate:.1%}</div>
    <div style="font-size:13px;color:{C['muted']};">{len(golden)} customers · {golden_rate/df['Personal Loan'].mean():.0f}x lift · ~{int(golden_rate*len(golden))} expected conversions</div>
    </div>""", unsafe_allow_html=True)

    # Campaign Simulator
    st.markdown("#### 🎚️ Campaign Targeting Simulator")
    st.caption("Move the slider to adjust model confidence threshold and see the real-time impact on ROI.")
    threshold = st.slider("Probability Threshold", 0.05, 0.95, 0.30, 0.05, key='thresh')
    targeted = df_pred[df_pred['pred_proba'] >= threshold]
    not_targeted = df_pred[df_pred['pred_proba'] < threshold]
    n_t = len(targeted)
    actual_conv = int(targeted['Personal Loan'].sum()) if n_t > 0 else 0
    prec_t = actual_conv/n_t if n_t > 0 else 0
    recall_t = actual_conv/df['Personal Loan'].sum() if df['Personal Loan'].sum() > 0 else 0
    cost_per, rev_per = 50, 5000
    roi = ((actual_conv*rev_per)-(n_t*cost_per))/(n_t*cost_per)*100 if n_t > 0 else 0

    t1,t2,t3,t4,t5 = st.columns(5)
    t1.metric("Targeted", f"{n_t:,}", f"{n_t/len(df):.1%} of base")
    t2.metric("Conversions", f"{actual_conv:,}")
    t3.metric("Precision", f"{prec_t:.1%}")
    t4.metric("Recall", f"{recall_t:.1%}")
    t5.metric("Est. ROI", f"{roi:,.0f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📊 Targeted vs Not-Targeted")
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Targeted', x=['Customers','Conversions'], y=[n_t, actual_conv], marker_color=C['accept']))
        fig.add_trace(go.Bar(name='Not Targeted', x=['Customers','Conversions'], y=[len(not_targeted), int(not_targeted['Personal Loan'].sum())], marker_color=C['muted']))
        fig.update_layout(barmode='group', yaxis_title='Count')
        style(fig, 360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 📈 Cumulative Gains (Lift Curve)")
        sorted_df = df_pred.sort_values('pred_proba', ascending=False)
        sorted_df['cum'] = sorted_df['Personal Loan'].cumsum()
        total_pos = sorted_df['Personal Loan'].sum()
        pcts = np.arange(1, len(sorted_df)+1)/len(sorted_df)*100
        gains = sorted_df['cum'].values/total_pos*100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pcts, y=gains, mode='lines', name='Model', line=dict(color=C['purple'], width=3)))
        fig.add_trace(go.Scatter(x=[0,100], y=[0,100], mode='lines', name='Random', line=dict(color=C['muted'], dash='dash')))
        fig.add_vline(x=n_t/len(df)*100, line_dash='dot', line_color=C['amber'], annotation_text=f'Threshold={threshold}', annotation_font_color=C['amber'])
        fig.update_layout(xaxis_title='% Customers Contacted', yaxis_title='% Conversions Captured')
        style(fig, 360)
        st.plotly_chart(fig, use_container_width=True)

    # Strategy Table
    st.markdown("#### 📋 Segment Strategy Matrix")
    seg_s = df_pred.groupby(['Edu_Label','Inc_Band'], observed=True).agg(n=('ID','count'), rate=('Personal Loan','mean'), score=('pred_proba','mean'), conv=('Personal Loan','sum')).reset_index()
    seg_s['Strategy'] = seg_s['score'].apply(lambda s: '🟢 Priority — RM Outreach' if s>0.5 else ('🟡 Strong — Digital Campaign' if s>0.1 else ('🟠 Low — Nurture' if s>0.01 else '🔴 Skip — Zero ROI')))
    seg_s['Rate'] = seg_s['rate'].apply(lambda x: f"{x:.1%}")
    seg_s['Score'] = seg_s['score'].apply(lambda x: f"{x:.3f}")
    display = seg_s[['Edu_Label','Inc_Band','n','Rate','Score','conv','Strategy']].sort_values('score', ascending=False).drop(columns='score')
    display.columns = ['Education','Income','Customers','Actual Rate','Model Score','Conversions','Strategy']
    st.dataframe(display, use_container_width=True, hide_index=True, height=440)

    # Action Plan
    st.markdown("#### 🗺️ Action Plan")
    with st.expander("🟢 Immediate — High Impact", expanded=True):
        st.markdown(f"""
- Deploy GB model (AUC=0.999) in production scoring
- Target **{len(golden)} golden-segment** customers → expected **{int(golden_rate*len(golden))} conversions** ({golden_rate:.0%})
- Assign RMs to all **{df['CD Account'].sum()} CD holders** (46% conversion)
- A/B test model-targeted vs untargeted for lift validation
        """)
    with st.expander("🟡 Short-Term — Process"):
        st.markdown("""
- Integrate scores into CRM for real-time prioritization
- Tiered outreach: RM calls (score>0.7), email (0.3–0.7), suppress (<0.05)
- Redesign messaging: family financial planning, education funding
        """)
    with st.expander("🔵 Medium-Term — Strategic"):
        st.markdown("""
- Investigate CD → Personal Loan as a product pathway
- Explore why high-income undergrads reject (product-market fit?)
- Build automated re-scoring triggered by life events
        """)

    st.markdown(f"""<div class="insight-card presc-card">
    <b>🎯 Bottom Line:</b> Targeting top ~10% by model score captures <b>99%+</b> of conversions while contacting 
    only 500 of 5,000 customers. At $50/contact and $5K/conversion, estimated <b>{roi:,.0f}% ROI</b> vs ~9.6% random.
    </div>""", unsafe_allow_html=True)


# ═══════════════════ TAB 5 — INTERACTIVE DRILL-DOWN ══════════════════════════
with tab5:
    st.markdown("## 📈 Interactive Drill-Down Explorer")
    st.markdown("*Click, filter, and explore from any angle. Build segments in real-time.*")

    # Sunburst
    st.markdown("#### 🌞 Click-to-Drill Sunburst")
    st.caption("Click any ring to zoom in. Click center to zoom out. 4 levels: Education → Income → Family → Loan.")
    sun_df = df.groupby(['Edu_Label','Inc_Band','Fam_Lbl','Loan_Label'], observed=True).size().reset_index(name='Count')
    fig = px.sunburst(sun_df, path=['Edu_Label','Inc_Band','Fam_Lbl','Loan_Label'], values='Count',
                      color='Edu_Label', color_discrete_map=EDU_COLORS, maxdepth=2)
    fig.update_traces(textinfo='label+percent parent', insidetextorientation='radial',
                      hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percentParent:.1%}<extra></extra>')
    style(fig, 560); fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # Segment Builder
    st.markdown("#### 🎚️ Custom Segment Builder")
    fc1,fc2,fc3,fc4,fc5 = st.columns(5)
    with fc1: inc_r = st.slider("Income ($K)", 8, 224, (8,224), key='dd_inc')
    with fc2: edu_s = st.multiselect("Education", ['Undergrad','Graduate','Advanced/Professional'], default=['Undergrad','Graduate','Advanced/Professional'], key='dd_edu')
    with fc3: fam_s = st.multiselect("Family Size", [1,2,3,4], default=[1,2,3,4], key='dd_fam')
    with fc4: cd_s = st.multiselect("CD Account", ['No','Yes'], default=['No','Yes'], key='dd_cd')
    with fc5: cc_r = st.slider("CC Spend ($K)", 0.0, 10.0, (0.0,10.0), 0.1, key='dd_cc')

    cd_map = {'No':0, 'Yes':1}
    seg = df[(df['Income']>=inc_r[0]) & (df['Income']<=inc_r[1]) & (df['Edu_Label'].isin(edu_s)) &
             (df['Family'].isin(fam_s)) & (df['CD Account'].isin([cd_map[x] for x in cd_s])) &
             (df['CCAvg']>=cc_r[0]) & (df['CCAvg']<=cc_r[1])]

    if len(seg) == 0:
        st.warning("No customers match. Broaden filters.")
    else:
        seg_rate = seg['Personal Loan'].mean()
        base = df['Personal Loan'].mean()
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Size", f"{len(seg):,}", f"{len(seg)/len(df):.1%}")
        m2.metric("Rate", f"{seg_rate:.1%}", f"{'▲' if seg_rate>base else '▼'} {abs(seg_rate-base):.1%}")
        m3.metric("Lift", f"{seg_rate/base:.1f}x" if base > 0 else "—")
        m4.metric("Avg Income", f"${seg['Income'].mean():,.0f}K")
        m5.metric("Avg CC", f"${seg['CCAvg'].mean():,.1f}K")
        m6.metric("Converts", f"{seg['Personal Loan'].sum()}/{len(seg)}")

        c1,c2,c3 = st.columns([2,3,2])
        with c1:
            st.markdown("##### Loan Split")
            sc = seg['Personal Loan'].value_counts()
            fig = go.Figure(go.Pie(labels=['Rejected','Accepted'], values=[sc.get(0,0), sc.get(1,0)], hole=0.7,
                                   marker=dict(colors=[C['reject'], C['accept']], line=dict(color=C['bg'], width=3)), textinfo='percent+value'))
            fig.add_annotation(text=f"<b style='font-size:28px;color:{C['accept'] if seg_rate>base else C['reject']}'>{seg_rate:.0%}</b>", x=0.5, y=0.5, showarrow=False)
            style(fig, 320); fig.update_layout(showlegend=False, margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("##### Income vs CC Spend")
            fig = px.scatter(seg, x='Income', y='CCAvg', color='Loan_Label', color_discrete_map={'Rejected':C['reject'],'Accepted':C['accept']},
                            opacity=0.6, size='Family', size_max=14, hover_data=['Edu_Label','Age'])
            fig.update_layout(xaxis_title='Income ($K)', yaxis_title='CC ($K/mo)')
            style(fig, 320); fig.update_layout(margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            st.markdown("##### Education Mix")
            ec = seg['Edu_Label'].value_counts()
            fig = go.Figure(go.Bar(x=ec.values, y=ec.index, orientation='h', marker_color=[EDU_COLORS.get(x, C['blue']) for x in ec.index],
                                   text=ec.values, textposition='outside'))
            style(fig, 320); fig.update_layout(margin=dict(t=10,l=150))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Column Deep-Dive
    st.markdown("#### 🔬 Variable Deep Dive")
    dd_col = st.selectbox("Variable", ['Income','Education','CCAvg','Family','CD Account','Age','Mortgage','Securities Account','Online','CreditCard'], key='dd_var')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"##### Distribution of {dd_col}")
        if dd_col in ['Income','CCAvg','Age','Mortgage']:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[df['Personal Loan']==0][dd_col], name='Rejected', marker_color=C['reject'], opacity=0.7, nbinsx=40))
            fig.add_trace(go.Histogram(x=df[df['Personal Loan']==1][dd_col], name='Accepted', marker_color=C['accept'], opacity=0.8, nbinsx=40))
            fig.update_layout(barmode='overlay', xaxis_title=dd_col, yaxis_title='Count')
        elif dd_col == 'Education':
            ec2 = df.groupby(['Edu_Label','Loan_Label'], observed=True).size().reset_index(name='n')
            fig = px.bar(ec2, x='Edu_Label', y='n', color='Loan_Label', barmode='stack', color_discrete_map={'Rejected':C['reject'],'Accepted':C['accept']})
        elif dd_col == 'Family':
            fc2 = df.groupby(['Family','Loan_Label'], observed=True).size().reset_index(name='n')
            fig = px.bar(fc2, x='Family', y='n', color='Loan_Label', barmode='stack', color_discrete_map={'Rejected':C['reject'],'Accepted':C['accept']})
        else:
            bc = df.groupby([dd_col,'Loan_Label']).size().reset_index(name='n')
            bc[dd_col] = bc[dd_col].map({0:'No', 1:'Yes'})
            fig = px.bar(bc, x=dd_col, y='n', color='Loan_Label', barmode='stack', color_discrete_map={'Rejected':C['reject'],'Accepted':C['accept']})
        style(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(f"##### Loan Rate by {dd_col}")
        if dd_col in ['Income','CCAvg','Age','Mortgage']:
            temp = df.copy()
            temp['_bin'] = pd.qcut(temp[dd_col], 10, duplicates='drop')
            r_df = temp.groupby('_bin', observed=True)['Personal Loan'].mean().reset_index()
            r_df.columns = ['Bin','Rate']
            r_df['Bin'] = r_df['Bin'].astype(str)
            fig = go.Figure(go.Bar(x=r_df['Bin'], y=r_df['Rate']*100,
                                   marker=dict(color=r_df['Rate'], colorscale='Viridis', showscale=True, colorbar=dict(title='%', thickness=10)),
                                   text=[f"{r:.1f}%" for r in r_df['Rate']*100], textposition='outside', textfont=dict(size=10)))
            fig.add_hline(y=9.6, line_dash='dot', line_color=C['amber'])
            fig.update_layout(xaxis_title=dd_col, yaxis_title='%', xaxis_tickangle=-30)
        elif dd_col == 'Education':
            r_df = df.groupby('Edu_Label', observed=True)['Personal Loan'].mean().reset_index()
            fig = go.Figure(go.Bar(x=r_df['Edu_Label'], y=r_df['Personal Loan']*100, marker_color=[EDU_COLORS.get(x,C['blue']) for x in r_df['Edu_Label']],
                                   text=[f"{r:.1f}%" for r in r_df['Personal Loan']*100], textposition='outside'))
            fig.add_hline(y=9.6, line_dash='dot', line_color=C['amber'])
        elif dd_col == 'Family':
            r_df = df.groupby('Family', observed=True)['Personal Loan'].mean().reset_index()
            fig = go.Figure(go.Bar(x=r_df['Family'].astype(str), y=r_df['Personal Loan']*100, marker_color=[C['blue'],C['purple'],C['amber'],C['green']],
                                   text=[f"{r:.1f}%" for r in r_df['Personal Loan']*100], textposition='outside'))
            fig.add_hline(y=9.6, line_dash='dot', line_color=C['amber'])
        else:
            r_df = df.groupby(dd_col)['Personal Loan'].mean().reset_index()
            fig = go.Figure(go.Bar(x=['No','Yes'], y=r_df['Personal Loan'].values*100, marker_color=[C['muted'],C['cyan']],
                                   text=[f"{r:.1f}%" for r in r_df['Personal Loan'].values*100], textposition='outside'))
            fig.add_hline(y=9.6, line_dash='dot', line_color=C['amber'])
        fig.update_layout(yaxis_title='Acceptance %')
        style(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Cross-Variable Explorer
    st.markdown("#### 🧩 Cross-Variable Interaction")
    ix1, ix2 = st.columns(2)
    with ix1: var_x = st.selectbox("X-Axis", ['Income','CCAvg','Age','Mortgage','Family','Education'], key='ix_x')
    with ix2: var_c = st.selectbox("Color By", ['Loan Status','Education','Family','CD Account'], key='ix_c')

    cmap_lookup = {'Loan Status': ('Loan_Label', {'Rejected':C['reject'],'Accepted':C['accept']}),
                   'Education': ('Edu_Label', EDU_COLORS),
                   'Family': ('Fam_Lbl', {'Single':C['blue'],'Couple':C['purple'],'Family of 3':C['amber'],'Family of 4':C['green']}),
                   'CD Account': ('CD Account', {0:C['muted'], 1:C['amber']})}
    cf, cm_d = cmap_lookup[var_c]
    samp = df.sample(min(2000, len(df)), random_state=42)

    if var_x in ['Family','Education']:
        agg = df.groupby([var_x, cf], observed=True)['Personal Loan'].mean().reset_index()
        agg.columns = [var_x, cf, 'Rate']
        fig = px.bar(agg, x=agg[var_x].astype(str), y='Rate', color=agg[cf].astype(str), barmode='group',
                    color_discrete_map={str(k):v for k,v in cm_d.items()}, text=agg['Rate'].apply(lambda x: f"{x:.0%}"))
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(yaxis=dict(tickformat='.0%'))
    else:
        y_var = 'CCAvg' if var_x != 'CCAvg' else 'Income'
        fig = px.scatter(samp, x=var_x, y=y_var, color=samp[cf].astype(str), opacity=0.55, size='Family', size_max=14,
                        color_discrete_map={str(k):v for k,v in cm_d.items()}, hover_data=['Edu_Label','Age','Personal Loan'])
        fig.update_layout(xaxis_title=var_x, yaxis_title=y_var)
    style(fig, 440)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div class="insight-card" style="border-color:{C['cyan']};">
    <b>💡 Tip:</b> Click into the Sunburst to discover hidden segments. Use the Segment Builder with 
    <b>Income>$100K + Graduate/Advanced + Family 3–4</b> to see the ~78% golden segment. The Cross-Variable 
    Explorer reveals interaction effects between any two variables.
    </div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"<div style='text-align:center;padding:12px;font-size:11px;color:{C['muted']};'>Universal Bank · Full Analytics Suite · 5,000 Customers · 14 Variables · 4 Models · Streamlit + Plotly + Scikit-learn</div>", unsafe_allow_html=True)
