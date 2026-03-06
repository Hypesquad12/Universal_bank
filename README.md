# 🏦 Universal Bank — Descriptive & Diagnostic Analytics Dashboard

An interactive Streamlit dashboard performing comprehensive **Descriptive** and **Diagnostic** analysis on the Universal Bank Personal Loan dataset (5,000 customers, 14 variables).

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)
![Plotly](https://img.shields.io/badge/Plotly-5.24-purple)

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **🏠 Executive Overview** | KPIs, Sunburst drill-down, Correlation Heatmap, Distribution Profiles |
| **📊 Column Deep-Dives** | Per-variable descriptive stats + diagnostic insights with interactive charts |
| **🔍 Diagnostic Drilldowns** | Multi-dimensional analysis: Sunburst, Parallel Categories, Bubble Maps, Radar |
| **🎯 Segment Explorer** | Build custom segments with sliders/filters, instant conversion rate feedback |
| **📋 Data Quality Report** | Completeness, outliers, skewness, and anomaly detection |

## 🔑 Key Findings

- **9.6%** loan acceptance rate (highly imbalanced target)
- **Income** (r=0.50) and **Education** (42.6% feature importance) are the top two drivers
- **Golden Segment**: Graduate/Advanced + Income >$100K → **~78% conversion** (8× lift)
- **CD Account holders** convert at **46.4%** — a 6.5× multiplier
- Age, Online, CreditCard, and Securities Account have **zero predictive value**

## 🛠️ Setup & Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/universal-bank-dashboard.git
cd universal-bank-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

## 📁 Project Structure

```
universal-bank-dashboard/
├── app.py                    # Main Streamlit application
├── UniversalBank.csv         # Dataset (5,000 records)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Dark theme configuration
└── README.md                 # This file
```

## 📈 Charts & Visualizations

- **Sunburst Chart** — Click-to-drill: Education → Income → Loan Status
- **Parallel Categories** — Flow diagram tracing customer paths to loan decisions
- **Radar Chart** — Acceptor vs Rejector normalized profiles
- **Correlation Heatmap** — All 12 numeric variables
- **Violin + Box Plots** — Distribution comparison with statistical overlays
- **Bubble Maps** — 4D scatter: Income × CCAvg × Family × Education
- **Interactive Segment Builder** — Real-time filtering with KPI feedback

## 📋 Dataset

The Universal Bank dataset contains 5,000 customer records with 14 variables:

| Variable | Type | Description |
|----------|------|-------------|
| Income | Continuous | Annual income ($K) — **Top predictor** |
| Education | Categorical | 1: Undergrad, 2: Graduate, 3: Advanced — **#1 tree split** |
| CCAvg | Continuous | Monthly credit card spending ($K) |
| Family | Categorical | Family size (1–4) |
| CD Account | Binary | Certificate of deposit — **46% conversion if yes** |
| Personal Loan | Binary | **Target variable** (9.6% acceptance) |
| Age, Experience, Mortgage, Securities Account, Online, CreditCard | Various | Supporting variables |

---

Built with ❤️ using Streamlit & Plotly
