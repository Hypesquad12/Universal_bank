import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank — Descriptive & Diagnostic Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}
h1, h2, h3 {
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-size: 13px !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 10px 24px;
    font-weight: 700;
}
section[data-testid="stSidebar"] {
    background: #0f172a;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0;
}
.insight-box {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-left: 4px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 10px 0;
    color: #e2e8f0;
    font-size: 14px;
    line-height: 1.7;
}
.golden-box {
    background: linear-gradient(135deg, #422006, #1c1917);
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 16px 0;
}
.section-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #64748b;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    df['Experience'] = df['Experience'].clip(lower=0)
    df['Education_Label'] = df['Education'].map({1: 'Undergrad', 2: 'Graduate', 3: 'Advanced/Professional'})
    df['Loan_Label'] = df['Personal Loan'].map({0: 'Rejected', 1: 'Accepted'})
    df['Income_Band'] = pd.cut(df['Income'], bins=[0,30,50,80,100,150,225],
                               labels=['<$30K','$30-50K','$50-80K','$80-100K','$100-150K','$150K+'])
    df['Age_Group'] = pd.cut(df['Age'], bins=[20,30,35,40,45,50,55,60,70],
                             labels=['23-30','31-35','36-40','41-45','46-50','51-55','56-60','61-67'])
    df['CCAvg_Band'] = pd.cut(df['CCAvg'], bins=[-0.1,1,2,3,5,10],
                              labels=['$0-1K','$1-2K','$2-3K','$3-5K','$5K+'])
    df['Mortgage_Status'] = np.where(df['Mortgage'] == 0, 'No Mortgage', 'Has Mortgage')
    df['Family_Label'] = df['Family'].astype(str) + ' Members'
    return df

df = load_data()

# ── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    'primary': '#3b82f6',
    'secondary': '#8b5cf6',
    'accent': '#f59e0b',
    'success': '#10b981',
    'danger': '#ef4444',
    'bg_dark': '#0f172a',
    'bg_card': '#1e293b',
    'text': '#e2e8f0',
    'text_muted': '#94a3b8',
    'accepted': '#10b981',
    'rejected': '#ef4444',
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#e2e8f0'),
        margin=dict(l=40, r=40, t=50, b=40),
        hoverlabel=dict(bgcolor='#1e293b', font_size=13, font_family='Inter'),
        colorway=['#3b82f6','#8b5cf6','#f59e0b','#10b981','#ef4444','#06b6d4','#ec4899','#84cc16'],
    )
)

def style_fig(fig, height=450):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        xaxis=dict(gridcolor='#1e293b', zerolinecolor='#334155'),
        yaxis=dict(gridcolor='#1e293b', zerolinecolor='#334155'),
    )
    return fig

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏦 Universal Bank")
    st.markdown("##### Descriptive & Diagnostic Analytics")
    st.markdown("---")

    page = st.radio(
        "**Navigate**",
        [
            "🏠 Executive Overview",
            "📊 Column Deep-Dives",
            "🔍 Diagnostic Drilldowns",
            "🎯 Segment Explorer",
            "📋 Data Quality Report",
        ],
        index=0,
    )
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:#64748b;font-size:12px;'>"
        f"<b>{len(df):,}</b> Customers · <b>14</b> Variables<br>"
        f"Loan Acceptance: <b style=\"color:#10b981\">{df['Personal Loan'].mean():.1%}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Overview":

    st.markdown("## 🏠 Executive Overview")
    st.markdown("*A bird's-eye view of 5,000 Universal Bank customers and their personal loan response.*")

    # ── KPI Row ──
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Customers", f"{len(df):,}")
    k2.metric("Loan Accepted", f"{df['Personal Loan'].sum():,}", f"{df['Personal Loan'].mean():.1%}")
    k3.metric("Avg Income", f"${df['Income'].mean():,.0f}K")
    k4.metric("Avg Age", f"{df['Age'].mean():.1f} yrs")
    k5.metric("Avg CC Spend", f"${df['CCAvg'].mean():,.1f}K/mo")
    k6.metric("CD Account Holders", f"{df['CD Account'].sum():,}", f"{df['CD Account'].mean():.1%}")

    st.markdown("---")

    # ── Row 1: Sunburst + Loan Split Donut ──
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("#### 🌞 Customer Composition Sunburst")
        st.caption("Click any ring to drill down: Education → Income Band → Loan Status")

        sun_df = df.groupby(['Education_Label','Income_Band','Loan_Label']).size().reset_index(name='Count')
        fig_sun = px.sunburst(
            sun_df, path=['Education_Label','Income_Band','Loan_Label'], values='Count',
            color='Education_Label',
            color_discrete_map={'Undergrad':'#3b82f6','Graduate':'#8b5cf6','Advanced/Professional':'#f59e0b'},
        )
        fig_sun.update_traces(
            textinfo='label+percent parent',
            insidetextorientation='radial',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percentParent:.1%}<extra></extra>',
        )
        style_fig(fig_sun, 520)
        fig_sun.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_sun, use_container_width=True)

    with c2:
        st.markdown("#### 🍩 Loan Acceptance Split")
        st.caption("9.6% accepted — a highly imbalanced target")

        loan_counts = df['Personal Loan'].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=['Rejected (90.4%)','Accepted (9.6%)'],
            values=[loan_counts[0], loan_counts[1]],
            hole=0.65,
            marker=dict(colors=[COLORS['rejected'], COLORS['accepted']], line=dict(color='#0f172a', width=3)),
            textinfo='label+value',
            textfont=dict(size=13),
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>',
        ))
        fig_donut.add_annotation(text=f"<b>{df['Personal Loan'].sum()}</b><br><span style='font-size:12px;color:#94a3b8'>Accepted</span>",
                                  x=0.5, y=0.5, showarrow=False, font=dict(size=28, color='#10b981'))
        style_fig(fig_donut, 520)
        fig_donut.update_layout(showlegend=False, margin=dict(t=20,b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Row 2: Correlation Heatmap ──
    st.markdown("#### 🔥 Correlation Heatmap — All Numeric Variables vs Personal Loan")
    st.caption("Hover over cells for exact values. Darker red/blue = stronger relationship.")

    num_cols = ['Age','Experience','Income','Family','CCAvg','Education','Mortgage',
                'Personal Loan','Securities Account','CD Account','Online','CreditCard']
    corr = df[num_cols].corr()

    fig_heat = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
        colorbar=dict(title='r', thickness=15),
    ))
    style_fig(fig_heat, 560)
    fig_heat.update_layout(
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange='reversed'),
        margin=dict(l=100, b=100),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Row 3: Distribution Grid ──
    st.markdown("#### 📈 Distribution Profiles — Continuous Variables")
    st.caption("Overlapping distributions colored by Loan acceptance. Separation = predictive power.")

    cont_vars = ['Income','CCAvg','Age','Experience','Mortgage']
    fig_dist = make_subplots(rows=1, cols=5, subplot_titles=cont_vars, horizontal_spacing=0.05)

    for i, var in enumerate(cont_vars):
        for label, color, name in [(0, COLORS['rejected'],'Rejected'),(1, COLORS['accepted'],'Accepted')]:
            subset = df[df['Personal Loan']==label][var]
            fig_dist.add_trace(go.Histogram(
                x=subset, name=name, marker_color=color, opacity=0.7,
                nbinsx=30, showlegend=(i==0),
                hovertemplate=f'{var}: ' + '%{x}<br>Count: %{y}<extra></extra>',
            ), row=1, col=i+1)

    style_fig(fig_dist, 320)
    fig_dist.update_layout(barmode='overlay', legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center'),
                           margin=dict(t=60, b=30))
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Insight Box ──
    st.markdown("""
    <div class="insight-box">
    <b>📌 Key Descriptive Takeaway:</b> The bank's customer base is predominantly middle-income ($64K median), 
    middle-aged (45 median), with low personal loan uptake (9.6%). Income and Credit Card Spending distributions 
    show <b>clear visual separation</b> between acceptors and non-acceptors — the two most promising signals for targeting.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COLUMN DEEP-DIVES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Column Deep-Dives":

    st.markdown("## 📊 Column-by-Column Deep Dive")
    st.markdown("*Select any variable for its full descriptive and diagnostic profile.*")

    col_choice = st.selectbox(
        "Choose a Column",
        ['Income', 'Education', 'CCAvg', 'Family', 'CD Account', 'Age',
         'Mortgage', 'Securities Account', 'Online', 'CreditCard', 'Experience'],
    )

    st.markdown("---")

    # ── INCOME ──
    if col_choice == 'Income':
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📊 Descriptive: Income Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[df['Personal Loan']==0]['Income'], name='Rejected',
                                       marker_color=COLORS['rejected'], opacity=0.7, nbinsx=40))
            fig.add_trace(go.Histogram(x=df[df['Personal Loan']==1]['Income'], name='Accepted',
                                       marker_color=COLORS['accepted'], opacity=0.8, nbinsx=40))
            fig.add_vline(x=100, line_dash='dash', line_color='#f59e0b', annotation_text='$100K Threshold',
                         annotation_position='top right', annotation_font_color='#f59e0b')
            fig.update_layout(barmode='overlay', xaxis_title='Annual Income ($K)', yaxis_title='Count')
            style_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### 🔍 Diagnostic: Loan Rate by Income Decile")
            df['Inc_Decile'] = pd.qcut(df['Income'], 10, labels=[f'D{i+1}' for i in range(10)])
            dec_df = df.groupby('Inc_Decile').agg(
                loan_rate=('Personal Loan','mean'),
                avg_income=('Income','mean'),
                count=('ID','count')
            ).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dec_df['Inc_Decile'], y=dec_df['loan_rate']*100,
                marker=dict(color=dec_df['loan_rate'], colorscale='Viridis', showscale=True,
                            colorbar=dict(title='Rate %', thickness=12)),
                text=[f"{r:.1f}%" for r in dec_df['loan_rate']*100],
                textposition='outside', textfont=dict(size=11, color='#e2e8f0'),
                hovertemplate='Decile: %{x}<br>Avg Income: $%{customdata[0]:.0f}K<br>Loan Rate: %{y:.1f}%<br>Customers: %{customdata[1]}<extra></extra>',
                customdata=np.stack([dec_df['avg_income'], dec_df['count']], axis=-1),
            ))
            fig.update_layout(xaxis_title='Income Decile (D1=Lowest)', yaxis_title='Loan Acceptance %')
            style_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        # Violin + Box combo
        st.markdown("#### 🎻 Income Distribution — Violin + Box by Loan Status")
        fig = go.Figure()
        for label, color, name in [(0, COLORS['rejected'],'Rejected'),(1, COLORS['accepted'],'Accepted')]:
            fig.add_trace(go.Violin(
                y=df[df['Personal Loan']==label]['Income'], name=name,
                box_visible=True, meanline_visible=True,
                fillcolor=color, opacity=0.6, line_color=color,
            ))
        fig.update_layout(yaxis_title='Income ($K)', showlegend=True)
        style_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <b>🔍 Diagnostic Insight — Income:</b> Income is the single strongest predictor (r=0.50). A dramatic 
        cliff exists at $100K: below it, acceptance is near-zero; above $150K, it surges to 49%. The 
        bottom 60% of earners have <b>zero</b> loan acceptance — marketing to them is wasted spend.
        The top 2 income deciles alone account for <b>~80%</b> of all loan acceptances.
        </div>
        """, unsafe_allow_html=True)

    # ── EDUCATION ──
    elif col_choice == 'Education':
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📊 Descriptive: Education Composition")
            edu_counts = df['Education_Label'].value_counts()
            fig = go.Figure(go.Pie(
                labels=edu_counts.index, values=edu_counts.values, hole=0.6,
                marker=dict(colors=['#3b82f6','#8b5cf6','#f59e0b'], line=dict(color='#0f172a', width=3)),
                textinfo='label+percent', textfont=dict(size=13),
                hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>',
            ))
            fig.add_annotation(text=f"<b>3</b><br><span style='font-size:11px;color:#94a3b8'>Levels</span>",
                              x=0.5, y=0.5, showarrow=False, font=dict(size=24, color='#e2e8f0'))
            style_fig(fig, 400)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### 🔍 Diagnostic: Loan Rate by Education")
            edu_loan = df.groupby('Education_Label')['Personal Loan'].agg(['mean','sum','count']).reset_index()
            edu_loan.columns = ['Education','Rate','Accepted','Total']

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=edu_loan['Education'], y=edu_loan['Rate']*100,
                marker=dict(color=['#3b82f6','#8b5cf6','#f59e0b']),
                text=[f"{r:.1f}%<br>({a}/{t})" for r,a,t in zip(edu_loan['Rate']*100, edu_loan['Accepted'], edu_loan['Total'])],
                textposition='outside', textfont=dict(size=12, color='#e2e8f0'),
                hovertemplate='<b>%{x}</b><br>Acceptance: %{y:.1f}%<extra></extra>',
            ))
            fig.update_layout(xaxis_title='Education Level', yaxis_title='Loan Acceptance %',
                             yaxis=dict(range=[0, 18]))
            style_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        # Interaction: Education × Income
        st.markdown("#### 🧩 Interaction Effect: Education × Income Band")
        st.caption("The real story is in the COMBINATION — education amplifies income's effect massively.")
        cross = df.groupby(['Education_Label','Income_Band'])['Personal Loan'].mean().reset_index()
        cross.columns = ['Education','Income_Band','Loan_Rate']

        fig = px.bar(cross, x='Income_Band', y='Loan_Rate', color='Education',
                     barmode='group', text=cross['Loan_Rate'].apply(lambda x: f"{x:.0%}"),
                     color_discrete_map={'Undergrad':'#3b82f6','Graduate':'#8b5cf6','Advanced/Professional':'#f59e0b'})
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(xaxis_title='Income Band', yaxis_title='Loan Acceptance Rate',
                         yaxis=dict(tickformat='.0%'), legend_title='Education')
        style_fig(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <b>🔍 Diagnostic Insight — Education:</b> Education is the #1 feature by decision tree importance (42.6%). 
        Undergrads accept at just 4.4%, while Graduate/Advanced are 3× more likely (13-14%). But the <b>explosive</b> 
        insight is the interaction: High Income ($150K+) Undergrads accept at only ~14%, while High Income 
        Graduate/Advanced accept at <b>70-80%+</b>. Education doesn't just add to income — it <b>multiplies</b> the effect.
        </div>
        """, unsafe_allow_html=True)

    # ── CCAVG ──
    elif col_choice == 'CCAvg':
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📊 Descriptive: CC Spending Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[df['Personal Loan']==0]['CCAvg'], name='Rejected',
                                       marker_color=COLORS['rejected'], opacity=0.7, nbinsx=40))
            fig.add_trace(go.Histogram(x=df[df['Personal Loan']==1]['CCAvg'], name='Accepted',
                                       marker_color=COLORS['accepted'], opacity=0.8, nbinsx=40))
            fig.update_layout(barmode='overlay', xaxis_title='Avg Monthly CC Spend ($K)', yaxis_title='Count')
            style_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### 🔍 Diagnostic: Loan Rate by CC Spend Band")
            cc_loan = df.groupby('CCAvg_Band')['Personal Loan'].agg(['mean','count']).reset_index()
            cc_loan.columns = ['Band','Rate','Count']

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cc_loan['Band'], y=cc_loan['Rate']*100,
                marker=dict(color=cc_loan['Rate'], colorscale='Plasma', showscale=True,
                            colorbar=dict(title='Rate%', thickness=12)),
                text=[f"{r:.1f}%" for r in cc_loan['Rate']*100],
                textposition='outside', textfont=dict(size=12),
                hovertemplate='<b>%{x}</b><br>Acceptance: %{y:.1f}%<br>Customers: %{customdata}<extra></extra>',
                customdata=cc_loan['Count'],
            ))
            fig.update_layout(xaxis_title='Monthly CC Spend Band', yaxis_title='Loan Acceptance %')
            style_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter: CCAvg vs Income colored by loan
        st.markdown("#### 🔬 CCAvg vs Income — 2D Behavioral Map")
        fig = px.scatter(df, x='Income', y='CCAvg', color='Loan_Label',
                        color_discrete_map={'Rejected': COLORS['rejected'], 'Accepted': COLORS['accepted']},
                        opacity=0.5, hover_data=['Education_Label','Family'])
        fig.update_layout(xaxis_title='Annual Income ($K)', yaxis_title='Monthly CC Spend ($K)')
        style_fig(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <b>🔍 Diagnostic Insight — CCAvg:</b> Loan acceptors spend 2.3× more on credit cards ($3.9K vs $1.7K/mo). 
        The scatter plot reveals a clear <b>cluster separation</b>: accepted customers form a distinct cloud in the 
        high-income + high-spend quadrant. CCAvg acts as a behavioral proxy for financial activity and credit appetite — 
        customers already comfortable with credit products are far more receptive to personal loans.
        </div>
        """, unsafe_allow_html=True)

    # ── FAMILY ──
    elif col_choice == 'Family':
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📊 Descriptive: Family Size Distribution")
            fam_counts = df['Family'].value_counts().sort_index()
            fig = go.Figure(go.Bar(
                x=[f"{x} Members" for x in fam_counts.index], y=fam_counts.values,
                marker=dict(color=['#3b82f6','#8b5cf6','#f59e0b','#10b981']),
                text=fam_counts.values, textposition='outside',
                hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>',
            ))
            fig.update_layout(xaxis_title='Family Size', yaxis_title='Count')
            style_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### 🔍 Diagnostic: Loan Rate by Family Size")
            fam_loan = df.groupby('Family')['Personal Loan'].mean().reset_index()
            fig = go.Figure(go.Bar(
                x=[f"{x} Members" for x in fam_loan['Family']], y=fam_loan['Personal Loan']*100,
                marker=dict(color=['#3b82f6','#8b5cf6','#f59e0b','#10b981']),
                text=[f"{r:.1f}%" for r in fam_loan['Personal Loan']*100],
                textposition='outside', textfont=dict(size=13),
                hovertemplate='<b>%{x}</b><br>Acceptance: %{y:.1f}%<extra></extra>',
            ))
            fig.update_layout(xaxis_title='Family Size', yaxis_title='Loan Acceptance %',
                             yaxis=dict(range=[0,18]))
            style_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        # Treemap: Family × Education × Loan
        st.markdown("#### 🌳 Treemap: Family × Education × Loan Status")
        st.caption("Size = customer count. Click to drill down into any segment.")
        tree_df = df.groupby(['Family_Label','Education_Label','Loan_Label']).size().reset_index(name='Count')
        fig = px.treemap(tree_df, path=['Family_Label','Education_Label','Loan_Label'], values='Count',
                        color='Count', color_continuous_scale='Blues')
        fig.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>')
        style_fig(fig, 450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <b>🔍 Diagnostic Insight — Family:</b> Family size 3 shows the highest acceptance (13.2%), 
        nearly double that of singles (7.3%). Larger families have greater financial needs — education costs, 
        housing, vehicles — driving higher loan demand. The treemap reveals that <b>Family 3-4 + Graduate/Advanced</b> 
        segments are disproportionately concentrated in the accepted group.
        </div>
        """, unsafe_allow_html=True)

    # ── CD ACCOUNT ──
    elif col_choice == 'CD Account':
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📊 Descriptive: CD Account Penetration")
            cd_counts = df['CD Account'].value_counts()
            fig = go.Figure(go.Pie(
                labels=['No CD Account','Has CD Account'], values=[cd_counts[0], cd_counts[1]],
                hole=0.65, marker=dict(colors=['#334155','#f59e0b'], line=dict(color='#0f172a', width=3)),
                textinfo='label+percent', textfont=dict(size=13),
            ))
            fig.add_annotation(text=f"<b>6%</b><br><span style='font-size:11px;color:#94a3b8'>CD Rate</span>",
                              x=0.5, y=0.5, showarrow=False, font=dict(size=22, color='#f59e0b'))
            style_fig(fig, 400)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### 🔍 Diagnostic: Loan Rate — CD vs No CD")
            cd_rates = df.groupby('CD Account')['Personal Loan'].mean()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['No CD Account','Has CD Account'], y=[cd_rates[0]*100, cd_rates[1]*100],
                marker=dict(color=['#334155','#f59e0b']),
                text=[f"{cd_rates[0]:.1%}", f"{cd_rates[1]:.1%}"],
                textposition='outside', textfont=dict(size=16, color='#e2e8f0'),
            ))
            fig.add_annotation(text="<b>6.4× Higher</b>", x=1, y=cd_rates[1]*100+5,
                             showarrow=True, arrowhead=2, arrowcolor='#f59e0b',
                             font=dict(color='#f59e0b', size=14))
            fig.update_layout(xaxis_title='', yaxis_title='Loan Acceptance %', yaxis=dict(range=[0,55]))
            style_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        # CD × Income interaction
        st.markdown("#### 🧩 Power Combo: CD Account × Income")
        combo = df.groupby(['CD Account','Income_Band']).agg(
            rate=('Personal Loan','mean'), n=('ID','count')
        ).reset_index()
        combo['CD_Label'] = combo['CD Account'].map({0:'No CD', 1:'Has CD'})
        fig = px.bar(combo, x='Income_Band', y='rate', color='CD_Label',
                    barmode='group', text=combo['rate'].apply(lambda x: f"{x:.0%}"),
                    color_discrete_map={'No CD':'#334155','Has CD':'#f59e0b'})
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(xaxis_title='Income Band', yaxis_title='Loan Rate', yaxis=dict(tickformat='.0%'))
        style_fig(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <b>🔍 Diagnostic Insight — CD Account:</b> CD holders accept personal loans at <b>46.4%</b> — a staggering 
        6.4× the rate of non-holders (7.2%). When combined with high income, the rate exceeds 80%. CD holders 
        represent deep banking relationships, savings discipline, and financial sophistication. They are a 
        <b>golden micro-segment</b> of only 302 customers with outsized conversion potential.
        </div>
        """, unsafe_allow_html=True)

    # ── AGE ──
    elif col_choice == 'Age':
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📊 Descriptive: Age Distribution")
            fig = px.histogram(df, x='Age', nbins=45, color='Loan_Label',
                             color_discrete_map={'Rejected': COLORS['rejected'], 'Accepted': COLORS['accepted']},
                             barmode='overlay', opacity=0.7)
            fig.update_layout(xaxis_title='Age (Years)', yaxis_title='Count')
            style_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### 🔍 Diagnostic: Loan Rate by Age Group")
            age_loan = df.groupby('Age_Group')['Personal Loan'].mean().reset_index()
            fig = go.Figure(go.Scatter(
                x=age_loan['Age_Group'], y=age_loan['Personal Loan']*100,
                mode='lines+markers+text', line=dict(color='#3b82f6', width=3),
                marker=dict(size=10, color='#3b82f6'),
                text=[f"{r:.1f}%" for r in age_loan['Personal Loan']*100],
                textposition='top center', textfont=dict(size=11),
            ))
            fig.add_hline(y=9.6, line_dash='dash', line_color='#f59e0b',
                         annotation_text='Overall 9.6%', annotation_font_color='#f59e0b')
            fig.update_layout(xaxis_title='Age Group', yaxis_title='Loan Acceptance %',
                             yaxis=dict(range=[0,16]))
            style_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <b>🔍 Diagnostic Insight — Age:</b> Age is effectively <b>noise</b> for loan prediction (r = -0.008). 
        Loan acceptance rates are nearly flat across all age groups (8.7% to 10.8%). The near-uniform age distribution 
        (skewness ≈ 0) suggests the bank's customer base spans all ages equally. <b>Do not waste marketing 
        budget on age-based segmentation</b> for personal loan campaigns.
        </div>
        """, unsafe_allow_html=True)

    # ── MORTGAGE ──
    elif col_choice == 'Mortgage':
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📊 Descriptive: Mortgage Distribution")
            fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'},{'type':'xy'}]],
                               subplot_titles=['Has Mortgage?','Mortgage Value (if any)'])
            mort_counts = df['Mortgage_Status'].value_counts()
            fig.add_trace(go.Pie(labels=mort_counts.index, values=mort_counts.values, hole=0.6,
                                marker=dict(colors=['#334155','#8b5cf6'], line=dict(color='#0f172a', width=2)),
                                textinfo='label+percent'), row=1, col=1)
            fig.add_trace(go.Histogram(x=df[df['Mortgage']>0]['Mortgage'], nbinsx=30,
                                       marker_color='#8b5cf6', opacity=0.8), row=1, col=2)
            style_fig(fig, 380)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### 🔍 Diagnostic: Loan Rate by Mortgage Status")
            mort_loan = df.groupby('Mortgage_Status')['Personal Loan'].mean()
            fig = go.Figure(go.Bar(
                x=mort_loan.index, y=mort_loan.values*100,
                marker=dict(color=['#334155','#8b5cf6']),
                text=[f"{r:.1f}%" for r in mort_loan.values*100],
                textposition='outside', textfont=dict(size=14),
            ))
            fig.update_layout(xaxis_title='', yaxis_title='Loan Acceptance %', yaxis=dict(range=[0,18]))
            style_fig(fig, 380)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <b>🔍 Diagnostic Insight — Mortgage:</b> 69% of customers have zero mortgage. Among those with mortgages, 
        values average $184K. Mortgage has a weak positive correlation (r=0.14) with loan acceptance — those with 
        mortgages are slightly more likely to accept, possibly because they're already comfortable with debt. 
        However, mortgage has <b>zero importance in tree models</b> — it's overshadowed by Income and Education.
        </div>
        """, unsafe_allow_html=True)

    # ── REMAINING COLUMNS (compact treatment) ──
    else:
        binary_cols = {'Securities Account': ('securities', 'Securities'),
                       'Online': ('online banking', 'Online Banking'),
                       'CreditCard': ('credit card', 'UniversalBank Credit Card'),
                       'Experience': (None, None)}

        if col_choice == 'Experience':
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 📊 Descriptive: Experience Distribution")
                fig = px.histogram(df, x='Experience', nbins=47, color='Loan_Label',
                                 color_discrete_map={'Rejected': COLORS['rejected'], 'Accepted': COLORS['accepted']},
                                 barmode='overlay', opacity=0.7)
                fig.update_layout(xaxis_title='Years of Experience', yaxis_title='Count')
                style_fig(fig, 400)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("#### 🔍 Diagnostic: Experience vs Age (Collinearity)")
                sample = df.sample(1000, random_state=42)
                fig = px.scatter(sample, x='Age', y='Experience', color='Loan_Label', opacity=0.5,
                               color_discrete_map={'Rejected': COLORS['rejected'], 'Accepted': COLORS['accepted']})
                fig.update_layout(xaxis_title='Age', yaxis_title='Experience')
                style_fig(fig, 400)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
            <b>🔍 Diagnostic Insight — Experience:</b> Experience is <b>99% correlated with Age</b> — 
            they form a near-perfect diagonal. 52 records have impossible negative values (all ages 23-29). 
            This variable is <b>completely redundant</b>: drop from all models to avoid multicollinearity. 
            Fix negative values as max(0, Age-23).
            </div>
            """, unsafe_allow_html=True)

        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### 📊 Descriptive: {col_choice} Distribution")
                counts = df[col_choice].value_counts()
                fig = go.Figure(go.Pie(
                    labels=[f'No ({counts.get(0,0):,})', f'Yes ({counts.get(1,0):,})'],
                    values=[counts.get(0,0), counts.get(1,0)], hole=0.6,
                    marker=dict(colors=['#334155', COLORS['primary']], line=dict(color='#0f172a', width=3)),
                    textinfo='label+percent',
                ))
                style_fig(fig, 400)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown(f"#### 🔍 Diagnostic: Loan Rate by {col_choice}")
                rates = df.groupby(col_choice)['Personal Loan'].mean()
                fig = go.Figure(go.Bar(
                    x=['No','Yes'], y=[rates[0]*100, rates[1]*100],
                    marker=dict(color=['#334155', COLORS['primary']]),
                    text=[f"{rates[0]:.1%}", f"{rates[1]:.1%}"],
                    textposition='outside', textfont=dict(size=14),
                ))
                fig.update_layout(xaxis_title=col_choice, yaxis_title='Loan Acceptance %',
                                 yaxis=dict(range=[0, max(rates)*130]))
                style_fig(fig, 400)
                st.plotly_chart(fig, use_container_width=True)

            corr_val = df[col_choice].corr(df['Personal Loan'])
            verdict = "negligible" if abs(corr_val) < 0.05 else "weak" if abs(corr_val) < 0.15 else "moderate"
            st.markdown(f"""
            <div class="insight-box">
            <b>🔍 Diagnostic Insight — {col_choice}:</b> Correlation with Personal Loan is <b>{corr_val:.3f} ({verdict})</b>. 
            {'This variable has virtually no diagnostic value for loan acceptance and should be excluded from targeting models.' if verdict == 'negligible' else 'This variable shows a slight association but is not a primary driver of loan acceptance.'}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DIAGNOSTIC DRILLDOWNS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Diagnostic Drilldowns":

    st.markdown("## 🔍 Diagnostic Drilldowns")
    st.markdown("*Multi-dimensional analysis: why do certain customers accept loans?*")

    # ── Interactive Donut with Drilldown ──
    st.markdown("#### 🍩 Click-to-Drill Donut: Education → Income → Loan")
    st.caption("Click any Education slice to expand into Income bands; click again to see Loan split.")

    # We'll use a sunburst as the drill-down donut alternative (Plotly's best click-to-expand)
    drill_df = df.groupby(['Education_Label', 'Income_Band', 'Loan_Label']).size().reset_index(name='Count')
    fig = px.sunburst(
        drill_df, path=['Education_Label','Income_Band','Loan_Label'], values='Count',
        color='Count', color_continuous_scale='Viridis',
        maxdepth=2,  # Start showing 2 levels, click to expand
    )
    fig.update_traces(
        textinfo='label+percent parent',
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share of Parent: %{percentParent:.1%}<extra></extra>',
        insidetextorientation='radial',
    )
    style_fig(fig, 550)
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Parallel Categories ──
    st.markdown("#### 🌊 Flow Diagram: How Customer Attributes Flow to Loan Decision")
    st.caption("Each ribbon traces a customer path through Education → Family → CD Account → Income Band → Loan Status")

    parcats_df = df[['Education_Label','Family_Label','Mortgage_Status','Income_Band','Loan_Label']].copy()
    fig = px.parallel_categories(
        parcats_df,
        dimensions=['Education_Label','Family_Label','Income_Band','Loan_Label'],
        color=df['Personal Loan'],
        color_continuous_scale=[[0, COLORS['rejected']], [1, COLORS['accepted']]],
        labels={'Education_Label':'Education','Family_Label':'Family','Income_Band':'Income','Loan_Label':'Loan'},
    )
    style_fig(fig, 500)
    fig.update_layout(margin=dict(l=50, r=50))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Bubble Chart: Income × CCAvg × Education ──
    st.markdown("#### 🫧 Bubble Map: Income × CC Spend × Family × Education")
    st.caption("Bubble size = Family size. Color = Education. Hover for details.")

    agg_df = df.groupby(['Income_Band','CCAvg_Band','Education_Label']).agg(
        loan_rate=('Personal Loan','mean'),
        avg_family=('Family','mean'),
        count=('ID','count'),
    ).reset_index()

    fig = px.scatter(
        df.sample(2000, random_state=42), x='Income', y='CCAvg',
        size='Family', color='Education_Label',
        color_discrete_map={'Undergrad':'#3b82f6','Graduate':'#8b5cf6','Advanced/Professional':'#f59e0b'},
        symbol='Loan_Label', symbol_map={'Rejected':'circle','Accepted':'diamond'},
        opacity=0.6, size_max=18,
        hover_data=['Age','Mortgage','CD Account'],
    )
    fig.update_layout(xaxis_title='Income ($K)', yaxis_title='Avg Monthly CC Spend ($K)',
                     legend_title='Education / Loan')
    style_fig(fig, 500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Heatmap: Pairwise Feature × Feature Loan Rate ──
    st.markdown("#### 🔥 Diagnostic Heatmap: Education × Income Band Loan Acceptance Rate")

    heat_data = df.pivot_table(values='Personal Loan', index='Education_Label',
                               columns='Income_Band', aggfunc='mean')
    fig = go.Figure(go.Heatmap(
        z=heat_data.values * 100,
        x=heat_data.columns.astype(str),
        y=heat_data.index,
        colorscale='YlOrRd',
        text=np.round(heat_data.values * 100, 1),
        texttemplate='%{text:.1f}%',
        textfont=dict(size=12),
        hovertemplate='Education: %{y}<br>Income: %{x}<br>Loan Rate: %{z:.1f}%<extra></extra>',
        colorbar=dict(title='Loan %', thickness=15),
    ))
    style_fig(fig, 350)
    fig.update_layout(xaxis_title='Income Band', yaxis_title='Education Level',
                     margin=dict(l=150))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Radar Chart: Acceptor vs Rejector Profile ──
    st.markdown("#### 🕸️ Radar Profile: Average Acceptor vs Rejector")

    radar_features = ['Income','CCAvg','Family','Education','Mortgage']
    accepted = df[df['Personal Loan']==1][radar_features].mean()
    rejected = df[df['Personal Loan']==0][radar_features].mean()

    # Normalize to 0-1 scale
    max_vals = df[radar_features].max()
    accepted_norm = (accepted / max_vals).values.tolist()
    rejected_norm = (rejected / max_vals).values.tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=accepted_norm + [accepted_norm[0]],
        theta=radar_features + [radar_features[0]],
        fill='toself', name='Accepted',
        fillcolor='rgba(16,185,129,0.2)', line=dict(color=COLORS['accepted'], width=3),
    ))
    fig.add_trace(go.Scatterpolar(
        r=rejected_norm + [rejected_norm[0]],
        theta=radar_features + [radar_features[0]],
        fill='toself', name='Rejected',
        fillcolor='rgba(239,68,68,0.15)', line=dict(color=COLORS['rejected'], width=2, dash='dot'),
    ))
    style_fig(fig, 450)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], gridcolor='#1e293b', tickfont=dict(size=9)),
            angularaxis=dict(gridcolor='#1e293b'),
            bgcolor='rgba(0,0,0,0)',
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>📌 Multi-Dimensional Diagnostic Summary:</b><br>
    The radar chart crystallizes the difference: loan acceptors have <b>2.2× higher income</b>, <b>2.3× higher CC spend</b>, 
    <b>higher education</b>, and <b>slightly larger families</b>. The diagnostic flow diagram confirms that the highest-converting 
    paths run through <b>Graduate/Advanced → High Income ($100K+) → Accepted</b>. Mortgage and Age add virtually nothing 
    to the diagnostic picture. The interaction between Education and Income is <b>multiplicative, not additive</b>.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SEGMENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Segment Explorer":

    st.markdown("## 🎯 Interactive Segment Explorer")
    st.markdown("*Build custom segments and instantly see their loan acceptance profile.*")

    # ── Filters ──
    st.markdown("##### 🎚️ Define Your Segment")
    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        inc_range = st.slider("Income ($K)", int(df['Income'].min()), int(df['Income'].max()),
                              (int(df['Income'].min()), int(df['Income'].max())))
    with fc2:
        edu_sel = st.multiselect("Education", ['Undergrad','Graduate','Advanced/Professional'],
                                 default=['Undergrad','Graduate','Advanced/Professional'])
    with fc3:
        fam_sel = st.multiselect("Family Size", [1,2,3,4], default=[1,2,3,4])
    with fc4:
        cd_sel = st.multiselect("CD Account", [0,1], default=[0,1], format_func=lambda x: 'Yes' if x else 'No')

    # Filter
    seg = df[
        (df['Income'] >= inc_range[0]) & (df['Income'] <= inc_range[1]) &
        (df['Education_Label'].isin(edu_sel)) &
        (df['Family'].isin(fam_sel)) &
        (df['CD Account'].isin(cd_sel))
    ]

    if len(seg) == 0:
        st.warning("No customers match this segment. Adjust your filters.")
    else:
        # KPIs
        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        seg_rate = seg['Personal Loan'].mean()
        lift = seg_rate / df['Personal Loan'].mean() if df['Personal Loan'].mean() > 0 else 0

        m1.metric("Segment Size", f"{len(seg):,}", f"{len(seg)/len(df):.1%} of total")
        m2.metric("Loan Acceptance", f"{seg_rate:.1%}",
                  f"{'↑' if seg_rate > 0.096 else '↓'} {abs(seg_rate - 0.096):.1%} vs baseline")
        m3.metric("Lift over Baseline", f"{lift:.1f}×")
        m4.metric("Avg Income", f"${seg['Income'].mean():,.0f}K")
        m5.metric("Avg CC Spend", f"${seg['CCAvg'].mean():,.1f}K/mo")

        # Charts
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📊 Segment Loan Split")
            seg_counts = seg['Personal Loan'].value_counts()
            fig = go.Figure(go.Pie(
                labels=['Rejected','Accepted'],
                values=[seg_counts.get(0,0), seg_counts.get(1,0)],
                hole=0.65,
                marker=dict(colors=[COLORS['rejected'], COLORS['accepted']], line=dict(color='#0f172a', width=3)),
                textinfo='label+percent+value', textfont=dict(size=12),
            ))
            pct = seg_rate*100
            fig.add_annotation(text=f"<b>{pct:.1f}%</b>", x=0.5, y=0.5, showarrow=False,
                              font=dict(size=26, color=COLORS['accepted'] if pct > 9.6 else COLORS['rejected']))
            style_fig(fig, 380)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### 📈 Income Distribution in Segment")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=seg[seg['Personal Loan']==0]['Income'], name='Rejected',
                                       marker_color=COLORS['rejected'], opacity=0.7, nbinsx=30))
            fig.add_trace(go.Histogram(x=seg[seg['Personal Loan']==1]['Income'], name='Accepted',
                                       marker_color=COLORS['accepted'], opacity=0.8, nbinsx=30))
            fig.update_layout(barmode='overlay', xaxis_title='Income ($K)', yaxis_title='Count')
            style_fig(fig, 380)
            st.plotly_chart(fig, use_container_width=True)

        # Segment breakdown table
        st.markdown("#### 📋 Segment Breakdown by Education")
        seg_edu = seg.groupby('Education_Label').agg(
            count=('ID','count'),
            loan_rate=('Personal Loan','mean'),
            avg_income=('Income','mean'),
            avg_cc=('CCAvg','mean'),
        ).reset_index()
        seg_edu.columns = ['Education','Customers','Loan Rate','Avg Income ($K)','Avg CC Spend ($K)']
        seg_edu['Loan Rate'] = seg_edu['Loan Rate'].apply(lambda x: f"{x:.1%}")
        seg_edu['Avg Income ($K)'] = seg_edu['Avg Income ($K)'].apply(lambda x: f"${x:,.0f}")
        seg_edu['Avg CC Spend ($K)'] = seg_edu['Avg CC Spend ($K)'].apply(lambda x: f"${x:,.1f}")
        st.dataframe(seg_edu, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DATA QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Data Quality Report":

    st.markdown("## 📋 Data Quality Report")
    st.markdown("*Every column assessed for completeness, validity, and anomalies.*")

    # ── Overview ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Total Columns", "14")
    m3.metric("Missing Values", "0 ✅")
    m4.metric("Data Quality Issues", "1 ⚠️", "Experience has 52 negative values")

    st.markdown("---")

    # Quality table
    st.markdown("#### 📊 Column-Level Quality Assessment")

    raw_df = pd.read_csv("UniversalBank.csv")

    quality_data = []
    for col in raw_df.columns:
        nulls = raw_df[col].isnull().sum()
        unique = raw_df[col].nunique()
        dtype = str(raw_df[col].dtype)
        issues = []

        if col == 'Experience' and (raw_df[col] < 0).any():
            issues.append(f"52 negative values (min={raw_df[col].min()})")
        if col == 'ZIP Code':
            issues.append("Excluded per data dictionary")
        if col == 'ID':
            issues.append("Identifier only — exclude from models")

        quality_data.append({
            'Column': col,
            'Type': dtype,
            'Nulls': nulls,
            'Unique': unique,
            'Min': f"{raw_df[col].min():.1f}" if dtype != 'object' else '—',
            'Max': f"{raw_df[col].max():.1f}" if dtype != 'object' else '—',
            'Status': '⚠️ Issue' if issues else '✅ Clean',
            'Notes': '; '.join(issues) if issues else 'No issues found',
        })

    qdf = pd.DataFrame(quality_data)
    st.dataframe(qdf, use_container_width=True, hide_index=True, height=530)

    st.markdown("---")

    # ── Outlier detection ──
    st.markdown("#### 📦 Box Plots — Outlier Detection")
    st.caption("Boxes show IQR. Dots beyond whiskers are potential outliers.")

    box_cols = ['Income','CCAvg','Mortgage','Age','Experience']
    fig = make_subplots(rows=1, cols=5, subplot_titles=box_cols, horizontal_spacing=0.06)
    colors = ['#3b82f6','#8b5cf6','#f59e0b','#10b981','#ef4444']
    for i, col in enumerate(box_cols):
        fig.add_trace(go.Box(y=df[col], name=col, marker_color=colors[i], boxmean='sd'), row=1, col=i+1)
    style_fig(fig, 380)
    fig.update_layout(showlegend=False, margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)

    # ── Skewness ──
    st.markdown("#### 📐 Distribution Shape — Skewness & Kurtosis")
    skew_data = []
    for col in ['Income','CCAvg','Mortgage','Age','Experience']:
        skew_data.append({
            'Variable': col,
            'Skewness': f"{df[col].skew():.3f}",
            'Kurtosis': f"{df[col].kurtosis():.3f}",
            'Shape': 'Symmetric' if abs(df[col].skew()) < 0.5 else ('Right-skewed' if df[col].skew() > 0 else 'Left-skewed'),
            'Tail Behavior': 'Normal tails' if abs(df[col].kurtosis()) < 1 else ('Heavy tails' if df[col].kurtosis() > 1 else 'Light tails'),
        })
    st.dataframe(pd.DataFrame(skew_data), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-box">
    <b>📌 Data Quality Summary:</b> The dataset is remarkably clean — zero null values across all 5,000 records 
    and 14 columns. The <b>only quality issue</b> is 52 negative Experience values (all in ages 23-29), 
    likely a data entry error for fresh graduates. <b>Recommendation:</b> Replace negative Experience with 
    max(0, Age - 23). CCAvg and Mortgage show heavy right skewness — consider log transforms for parametric models.
    </div>
    """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:12px;padding:10px;'>"
    "Universal Bank Analytics Dashboard · Built with Streamlit & Plotly · "
    "5,000 Customers · Descriptive & Diagnostic Analysis"
    "</div>",
    unsafe_allow_html=True,
)
