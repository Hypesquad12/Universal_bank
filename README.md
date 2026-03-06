# 🏦 Universal Bank — Complete Analytics Dashboard

A 5-tab interactive Streamlit dashboard covering **Descriptive**, **Diagnostic**, **Predictive**, **Prescriptive**, and **Interactive Drill-Down** analytics on the Universal Bank Personal Loan dataset.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## 📊 Five Analytics Tabs

| Tab | Type | What's Inside |
|-----|------|---------------|
| 📊 **Descriptive** | What happened? | KPIs, donut charts, distribution overlays, correlation matrix, binary variable composition |
| 🔍 **Diagnostic** | Why? | Income staircase, Education×Income heatmap, CD multiplier effect, parallel categories flow, radar profiles |
| 🤖 **Predictive** | What will happen? | 4-model comparison (LR, DT, RF, GB), ROC curves, confusion matrices, feature importance |
| 🎯 **Prescriptive** | What to do? | Golden segment banner, campaign simulator with ROI slider, lift curve, strategy matrix, action plans |
| 📈 **Drill-Down** | Explore freely | Click-to-drill sunburst, custom segment builder, variable deep-dive, cross-variable interaction explorer |

## 🔑 Key Findings

- **9.6%** loan acceptance (highly imbalanced)
- **Gradient Boosting** achieves **AUC = 0.9989** (98.9% accuracy)
- **Golden Segment**: Graduate/Advanced + Income >$100K + Family 3+ → **~78% conversion** (8× lift)
- **CD Account holders** convert at **46.4%** (6.4× multiplier)
- Top 10% by model score captures **99%+** of all conversions

## 🛠️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/universal-bank-dashboard.git
cd universal-bank-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → Set main file to `app.py` → Deploy

## 📁 Structure

```
├── app.py                  # Main dashboard (all 5 tabs)
├── UniversalBank.csv       # Dataset (5,000 records × 14 columns)
├── requirements.txt        # Dependencies
├── .streamlit/config.toml  # Dark theme
└── README.md
```

---
Built with Streamlit + Plotly + Scikit-learn
