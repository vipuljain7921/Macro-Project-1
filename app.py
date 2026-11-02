"""
ChurnGuard AI – Premium Streamlit Dashboard  (Final Build)
===========================================================
Pages: Home · Predict · Analytics · Segmentation · Model Performance
       Dataset Explorer · Prediction History · AI Chatbot
Run :  streamlit run app.py
"""

from __future__ import annotations
import os, sys, json, pickle, datetime, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils.data_utils       import preprocess_single_customer, get_feature_display_names
from utils.model_utils      import (models_exist, load_artifacts, load_training_results,
                                     predict_churn, get_feature_importances, generate_sample_customer)
from utils.retention_engine import get_risk_category, generate_retention_strategies, get_action_plan
from utils.auth             import verify_credentials, get_demo_hint
from utils.report_generator import generate_pdf_report
from utils.segmentation     import (segmentation_exists, train_segmentation, load_segmentation,
                                     predict_segment, get_segment_summary, SEGMENT_META, SEG_FEATURES)

try:
    import anthropic as _anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")


def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');
:root{--bg:#080d1a;--bg2:#0e1628;--card:rgba(14,22,44,.88);--bdr:rgba(0,212,255,.18);
 --bdr2:rgba(0,212,255,.38);--a:#00d4ff;--a2:#7c3aed;--a3:#00ffaa;
 --danger:#ef4444;--warn:#f59e0b;--ok:#10b981;--fg:#e8f0ff;--muted:#8899b8;
 --fd:'Syne',sans-serif;--fb:'DM Sans',sans-serif;}
html,body,.stApp{background:var(--bg)!important;font-family:var(--fb)!important;color:var(--fg)!important;}
.main .block-container{padding:1.5rem 2rem 3rem;max-width:1440px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#070c18,#0b1224)!important;border-right:1px solid var(--bdr);}
[data-testid="stSidebar"] .stRadio>div{gap:.12rem;}
[data-testid="stSidebar"] .stRadio label{font-family:var(--fb)!important;font-size:.87rem;color:var(--muted)!important;padding:.43rem .8rem;border-radius:8px;transition:all .2s;cursor:pointer;}
[data-testid="stSidebar"] .stRadio label:hover{color:var(--a)!important;background:rgba(0,212,255,.06);}
h1,h2,h3{font-family:var(--fd)!important;}
h1{font-size:2.5rem!important;font-weight:800!important;background:linear-gradient(135deg,var(--a),var(--a2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.15!important;}
h2{font-size:1.4rem!important;font-weight:700!important;color:var(--fg)!important;}
h3{font-size:1.05rem!important;font-weight:600!important;color:var(--fg)!important;}
.cg-card{background:var(--card);backdrop-filter:blur(16px);border:1px solid var(--bdr);border-radius:16px;padding:1.4rem 1.6rem;margin-bottom:.9rem;box-shadow:0 4px 32px rgba(0,0,0,.4);transition:border-color .25s,box-shadow .25s;}
.cg-card:hover{border-color:var(--bdr2);box-shadow:0 4px 48px rgba(0,212,255,.07);}
.cg-card-title{font-family:var(--fd);font-size:.7rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;}
.cg-card-value{font-family:var(--fd);font-size:2rem;font-weight:800;line-height:1;}
.cg-card-delta{font-size:.79rem;color:var(--muted);margin-top:.3rem;}
.risk-badge{display:inline-flex;align-items:center;gap:.4rem;padding:.4rem 1.1rem;border-radius:99px;font-family:var(--fd);font-size:.86rem;font-weight:700;letter-spacing:.04em;}
.risk-high{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.5);color:#ef4444;}
.risk-medium{background:rgba(245,158,11,.15);border:1px solid rgba(245,158,11,.5);color:#f59e0b;}
.risk-low{background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.5);color:#10b981;}
.strat-card{background:rgba(14,22,44,.9);border-left:3px solid var(--a);border-radius:0 12px 12px 0;padding:.9rem 1.1rem;margin-bottom:.65rem;}
.strat-title{font-family:var(--fd);font-weight:700;font-size:.93rem;margin-bottom:.2rem;}
.strat-desc{font-size:.81rem;color:var(--muted);line-height:1.55;}
.prio-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.67rem;font-weight:700;letter-spacing:.05em;margin-left:.4rem;}
.mp{background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.2);border-radius:8px;padding:.5rem .8rem;text-align:center;}
.mp .mv{font-family:var(--fd);font-size:1.25rem;font-weight:700;color:var(--a);}
.mp .ml{font-size:.68rem;color:var(--muted);}
.seg-card{background:var(--card);border:1px solid var(--bdr);border-radius:14px;padding:1.1rem 1.3rem;height:100%;}
.seg-icon{font-size:1.7rem;margin-bottom:.3rem;}
.seg-name{font-family:var(--fd);font-size:.95rem;font-weight:700;margin-bottom:.3rem;}
.seg-desc{font-size:.79rem;color:var(--muted);line-height:1.5;margin-bottom:.5rem;}
.seg-action{font-size:.76rem;color:var(--a3);font-style:italic;}
.chat-user{background:rgba(124,58,237,.18);border:1px solid rgba(124,58,237,.3);border-radius:16px 16px 4px 16px;padding:.72rem .95rem;margin:.35rem 0;max-width:80%;margin-left:auto;font-size:.87rem;}
.chat-ai{background:rgba(0,212,255,.07);border:1px solid rgba(0,212,255,.14);border-radius:16px 16px 16px 4px;padding:.72rem .95rem;margin:.35rem 0;max-width:85%;font-size:.87rem;line-height:1.6;}
.chat-lbl{font-size:.68rem;color:var(--muted);margin-bottom:.15rem;}
.stButton>button{background:linear-gradient(135deg,var(--a),var(--a2))!important;color:#fff!important;border:none!important;border-radius:10px!important;font-family:var(--fd)!important;font-weight:600!important;font-size:.87rem!important;padding:.5rem 1.3rem!important;transition:opacity .2s,transform .15s!important;}
.stButton>button:hover{opacity:.85!important;transform:translateY(-1px)!important;}
.stSelectbox label,.stSlider label,.stNumberInput label,.stTextInput label,.stTextArea label{color:var(--muted)!important;font-size:.81rem!important;}
.stTextInput input,.stNumberInput input,.stTextArea textarea{background:var(--bg2)!important;border:1px solid var(--bdr)!important;border-radius:8px!important;color:var(--fg)!important;}
.stTextInput input:focus,.stTextArea textarea:focus{border-color:var(--a)!important;box-shadow:0 0 0 2px rgba(0,212,255,.13)!important;}
.cg-div{border:none;border-top:1px solid var(--bdr);margin:1rem 0;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.live-dot{display:inline-block;width:7px;height:7px;background:var(--ok);border-radius:50%;animation:pulse 1.6s ease-in-out infinite;margin-right:4px;}
.cg-footer{text-align:center;font-size:.72rem;color:var(--muted);padding:2rem 0 .5rem;border-top:1px solid var(--bdr);margin-top:2rem;}
.sh{display:flex;align-items:center;gap:.55rem;margin-bottom:.75rem;}
.sh-ico{width:30px;height:30px;background:linear-gradient(135deg,rgba(0,212,255,.14),rgba(124,58,237,.14));border:1px solid var(--bdr);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:.9rem;flex-shrink:0;}
.sh-ttl{font-family:var(--fd);font-size:1.1rem;font-weight:700;color:var(--fg);}
.login-wrap{max-width:440px;margin:3rem auto;background:var(--card);border:1px solid var(--bdr);border-radius:20px;padding:2.6rem 2.2rem;text-align:center;}
</style>""", unsafe_allow_html=True)


def init_state():
    for k,v in {"authenticated":False,"user_role":None,
                "prediction_history":[],"chat_history":[],
                "last_prediction":None}.items():
        if k not in st.session_state: st.session_state[k]=v


@st.cache_resource(show_spinner=False)
def _load_model_artifacts(artifact_signature):
    return load_artifacts()

def boot_models():
    if not models_exist():
        from train_model import run_training
        run_training(n_samples=10_000)
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    artifact_names = ("best_model.pkl", "scaler.pkl", "feature_names.pkl", "best_model_name.pkl")
    artifact_signature = tuple(
        (name, os.stat(os.path.join(model_dir, name)).st_mtime_ns,
         os.path.getsize(os.path.join(model_dir, name)))
        for name in artifact_names
    )
    return _load_model_artifacts(artifact_signature)

@st.cache_data(show_spinner=False)
def _read_dataset(path, modified_ns, file_size):
    return pd.read_csv(path)

def get_dataset():
    path = os.path.join(os.path.dirname(__file__), "data", "banking_churn_data.csv")
    if not os.path.exists(path):
        from data.generate_data import generate_banking_dataset
        df = generate_banking_dataset(10_000)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    stat = os.stat(path)
    return _read_dataset(path, stat.st_mtime_ns, stat.st_size)

@st.cache_data(show_spinner=False)
def _read_training_results(path, modified_ns, file_size):
    return load_training_results()

def get_training_results():
    path = os.path.join(os.path.dirname(__file__), "models", "training_results.pkl")
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    return _read_training_results(path, stat.st_mtime_ns, stat.st_size)

@st.cache_data(show_spinner=False)
def get_seg_data(df):
    if not segmentation_exists(): train_segmentation(df,n_clusters=5)
    km,scaler,pca=load_segmentation()
    X=df[SEG_FEATURES].copy().fillna(0)
    Xs=scaler.transform(X)
    labels=km.predict(Xs); X_pca=pca.transform(Xs)
    summary=get_segment_summary(df,labels)
    return labels,X_pca,summary


# ── Chart helpers ─────────────────────────────────────────────────────────────
_BL=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
         font=dict(family="DM Sans,sans-serif",color="#8899b8",size=12),
         margin=dict(l=16,r=16,t=44,b=16),
         legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor="rgba(0,0,0,0)"),
         xaxis=dict(gridcolor="rgba(255,255,255,.05)",zerolinecolor="rgba(255,255,255,.07)"),
         yaxis=dict(gridcolor="rgba(255,255,255,.05)",zerolinecolor="rgba(255,255,255,.07)"))
PAL=["#00d4ff","#7c3aed","#10b981","#f59e0b","#ef4444","#f97316","#06b6d4","#8b5cf6","#ec4899","#14b8a6"]

def sf(fig,title="",h=370):
    fig.update_layout(**_BL,height=h,title=dict(text=title,font=dict(family="Syne,sans-serif",size=14,color="#e8f0ff")))
    return fig

def sh(icon,title):
    st.markdown(f"<div class='sh'><div class='sh-ico'>{icon}</div><div class='sh-ttl'>{title}</div></div>",unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
def page_login():
    st.markdown("""<div class="login-wrap">
    <div style="font-size:3.2rem;margin-bottom:.4rem">🛡️</div>
    <div style="font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;
         background:linear-gradient(135deg,#00d4ff,#7c3aed);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent">ChurnGuard AI</div>
    <div style="color:#8899b8;font-size:.87rem;margin:.3rem 0 1.8rem">
         Banking Intelligence Platform · Secure Login</div></div>""",unsafe_allow_html=True)
    _,col,_=st.columns([1,2,1])
    with col:
        u=st.text_input("Username",placeholder="Username")
        p=st.text_input("Password",type="password",placeholder="Password")
        if st.button("🔐  Sign In",use_container_width=True):
            ok,result=verify_credentials(u,p)
            if ok:
                st.session_state.authenticated=True; st.session_state.user_role=result; st.rerun()
            else: st.error(result)
        st.caption(f"ℹ️  {get_demo_hint()}")


# ═══════════════════════════════════════════════════════════════════════════════
def page_home(df):
    st.markdown("""<div style="text-align:center;padding:1.2rem 0 1.8rem">
    <div style="font-size:3.4rem;margin-bottom:.25rem">🛡️</div>
    <h1 style="font-size:2.9rem!important;margin-bottom:.35rem">ChurnGuard AI</h1>
    <p style="color:#8899b8;font-size:1rem;max-width:620px;margin:0 auto 1.2rem;line-height:1.6">
      AI-Powered Banking Churn Prediction &amp; Customer Retention Intelligence Platform</p>
    <span class="live-dot"></span>
    <span style="font-size:.79rem;color:#10b981">Live System · All Models Operational</span></div>""",
    unsafe_allow_html=True)

    cr=df["Churn"].mean()
    for col,(ic,val,lbl,clr) in zip(st.columns(4),[
        ("🧑‍🤝‍🧑",f"{len(df):,}","Total Customers","#00d4ff"),
        ("⚠️",f"{cr:.1%}","Overall Churn Rate","#ef4444"),
        ("✅",f"{df['IsActiveMember'].mean():.1%}","Active Members","#10b981"),
        ("💰",f"${df['Balance'].mean():,.0f}","Avg Balance","#f59e0b")]):
        col.markdown(f"<div class='cg-card' style='text-align:center'><div style='font-size:1.5rem'>{ic}</div>"
                     f"<div class='cg-card-value' style='color:{clr}'>{val}</div>"
                     f"<div class='cg-card-delta'>{lbl}</div></div>",unsafe_allow_html=True)

    st.markdown("<hr class='cg-div'>",unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        vals=[int((df["Churn"]==0).sum()),int((df["Churn"]==1).sum())]
        fig=go.Figure(go.Pie(labels=["Retained","Churned"],values=vals,hole=.58,
                              marker_colors=["#10b981","#ef4444"]))
        fig.add_annotation(text=f"{cr:.1%}<br>Churn",x=.5,y=.5,showarrow=False,font_size=17,font_color="#e8f0ff")
        st.plotly_chart(sf(fig,"Churn Distribution",310),use_container_width=True)
    with c2:
        gc=df.groupby("Geography")["Churn"].mean().reset_index(); gc.columns=["Geo","Rate"]
        fig=px.bar(gc,x="Geo",y="Rate",color="Rate",color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                   text=gc["Rate"].map("{:.1%}".format))
        fig.update_traces(textposition="outside",textfont_color="#e8f0ff"); fig.update_coloraxes(showscale=False)
        st.plotly_chart(sf(fig,"Churn by Geography",310),use_container_width=True)
    with c3:
        d2=df.copy()
        d2["AG"]=pd.cut(d2["Age"],[18,25,35,45,55,65,100],labels=["18-24","25-34","35-44","45-54","55-64","65+"],right=False)
        ag=d2.groupby("AG",observed=True)["Churn"].mean().reset_index(); ag.columns=["Group","Rate"]
        fig=px.line(ag,x="Group",y="Rate",markers=True,color_discrete_sequence=["#00d4ff"])
        fig.update_traces(line_width=2.5,marker_size=8)
        fig.add_hline(y=cr,line_dash="dot",line_color="#f59e0b",annotation_text="Avg",annotation_font_color="#f59e0b")
        st.plotly_chart(sf(fig,"Churn by Age Group",310),use_container_width=True)




# ═══════════════════════════════════════════════════════════════════════════════
def page_predict(model, scaler, feature_names, best_name):
    sh("🔮","Customer Churn Prediction")
    st.caption("Complete the customer profile and click **Predict** for instant AI-powered churn risk assessment.")
    if st.button("🎲  Load Random Sample Customer"):
        st.session_state["_sample"]=generate_sample_customer(); st.rerun()
    s=st.session_state.get("_sample",{})

    with st.form("pf"):
        st.markdown("##### 👤 Demographics")
        r1,r2,r3=st.columns(3)
        with r1:
            age=st.slider("Age",18,80,int(s.get("Age",40)))
            gender=st.selectbox("Gender",["Male","Female"],index=0 if s.get("Gender","Male")=="Male" else 1)
        with r2:
            geography=st.selectbox("Geography",["France","Germany","Spain"],
                index=["France","Germany","Spain"].index(s.get("Geography","France")))
            card_type=st.selectbox("Card Type",["Silver","Gold","Platinum","Diamond"],
                index=["Silver","Gold","Platinum","Diamond"].index(s.get("CardType","Gold")))
        with r3:
            tenure=st.slider("Tenure (Years)",0,10,int(s.get("Tenure",3)))
            num_products=st.selectbox("Number of Products",[1,2,3,4],index=int(s.get("NumOfProducts",2))-1)

        st.markdown("##### 💰 Financial Profile")
        f1,f2,f3=st.columns(3)
        with f1: credit_score=st.slider("Credit Score",300,850,int(s.get("CreditScore",650)))
        with f2: balance=st.number_input("Account Balance ($)",0.0,300_000.0,float(s.get("Balance",80_000.0)),step=1_000.0)
        with f3: salary=st.number_input("Estimated Salary ($)",10_000.0,200_000.0,float(s.get("EstimatedSalary",100_000.0)),step=1_000.0)

        st.markdown("##### 📱 Behaviour & Engagement")
        b1,b2,b3=st.columns(3)
        with b1:
            is_active=st.selectbox("Active Member",["Yes","No"],index=0 if s.get("IsActiveMember",1)==1 else 1)
            online_banking=st.selectbox("Online Banking",["Yes","No"],index=0 if s.get("OnlineBankingUsage",1)==1 else 1)
        with b2:
            tx_freq=st.slider("Monthly Transactions",1,30,int(s.get("TransactionFrequency",12)))
            account_activity=st.selectbox("Account Activity",["High","Medium","Low"],
                index=["High","Medium","Low"].index(s.get("AccountActivity","Medium")))
        with b3:
            has_loan=st.selectbox("Has Active Loan",["Yes","No"],index=0 if s.get("HasLoan",0)==1 else 1)

        submitted=st.form_submit_button("⚡  Predict Churn Risk",use_container_width=True)

    if submitted:
        customer={"Age":age,"Gender":gender,"Geography":geography,"CreditScore":credit_score,
                  "Balance":balance,"EstimatedSalary":salary,"NumOfProducts":num_products,
                  "Tenure":tenure,"IsActiveMember":1 if is_active=="Yes" else 0,
                  "TransactionFrequency":tx_freq,"AccountActivity":account_activity,
                  "CardType":card_type,"OnlineBankingUsage":1 if online_banking=="Yes" else 0,
                  "HasLoan":1 if has_loan=="Yes" else 0}

        X=preprocess_single_customer(customer,scaler,feature_names)
        label,prob=predict_churn(model,X)
        risk_cat,risk_color,risk_emoji=get_risk_category(prob)
        strategies=generate_retention_strategies(customer,prob)
        action_plan=get_action_plan(prob)
        seg_id,seg_meta=predict_segment(customer)

        entry={"customer":customer,"probability":prob,"label":label,"risk_category":risk_cat,
               "segment":seg_meta["name"],"timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        st.session_state.last_prediction=entry
        st.session_state.prediction_history.append(entry.copy())

        st.markdown("---")
        sh("📊","Prediction Results")
        left,right=st.columns([1,1.5])

        with left:
            fig=go.Figure(go.Indicator(mode="gauge+number",value=prob*100,
                number={"suffix":"%","font":{"size":42,"family":"Syne,sans-serif","color":"#e8f0ff"}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#8899b8"},
                       "bar":{"color":risk_color,"thickness":.22},"bgcolor":"rgba(0,0,0,0)",
                       "bordercolor":"rgba(255,255,255,.1)",
                       "steps":[{"range":[0,30],"color":"rgba(16,185,129,.12)"},
                                 {"range":[30,60],"color":"rgba(245,158,11,.12)"},
                                 {"range":[60,100],"color":"rgba(239,68,68,.12)"}],
                       "threshold":{"line":{"color":risk_color,"width":3},"thickness":.8,"value":prob*100}}))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=265,margin=dict(l=26,r=26,t=14,b=6),font=dict(color="#8899b8"))
            st.plotly_chart(fig,use_container_width=True)

            rc={"High Risk":"risk-high","Medium Risk":"risk-medium","Low Risk":"risk-low"}[risk_cat]
            st.markdown(f"<div style='text-align:center;margin-top:-.4rem'>"
                        f"<span class='risk-badge {rc}'>{risk_emoji} {risk_cat}</span></div>",unsafe_allow_html=True)
            st.markdown("<div style='height:.6rem'></div>",unsafe_allow_html=True)
            m1,m2,m3=st.columns(3)
            m1.markdown(f"<div class='mp'><div class='mv'>{prob:.0%}</div><div class='ml'>Churn Prob</div></div>",unsafe_allow_html=True)
            m2.markdown(f"<div class='mp'><div class='mv'>{'YES' if label else 'NO'}</div><div class='ml'>Prediction</div></div>",unsafe_allow_html=True)
            m3.markdown(f"<div class='mp'><div class='mv'>{(1-prob):.0%}</div><div class='ml'>Retention</div></div>",unsafe_allow_html=True)
            st.markdown("<div style='height:.5rem'></div>",unsafe_allow_html=True)
            st.markdown(f"""<div class="cg-card" style="text-align:center;padding:.9rem">
            <div style="font-size:1.4rem">{seg_meta['icon']}</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.88rem;
                 color:{seg_meta['color']};margin:.2rem 0">{seg_meta['name']}</div>
            <div style="font-size:.73rem;color:#8899b8">Customer Segment</div></div>""",unsafe_allow_html=True)

        with right:
            st.markdown("#### 🎯 Retention Strategies")
            pc_map={"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#f59e0b","LOW":"#10b981"}
            for s in strategies[:3]:   # show top 3 for clarity
                pc=pc_map.get(s["priority"],"#8899b8")
                st.markdown(f"""<div class="strat-card" style="border-left-color:{s['badge_color']}">
                <div class="strat-title">{s['icon']} {s['title']}
                    <span class="prio-badge" style="background:rgba(0,0,0,.3);color:{pc};border:1px solid {pc}">{s['priority']}</span>
                </div>
                <div class="strat-desc">{s['description']}</div>
                <div style="font-size:.72rem;color:#00d4aa;margin-top:.3rem">Est. retention improvement: {s['estimated_retention']}</div>
                </div>""",unsafe_allow_html=True)

        # Concise next-step callout
        ap_color={"High Risk":"rgba(239,68,68,.12)","Medium Risk":"rgba(245,158,11,.10)","Low Risk":"rgba(16,185,129,.10)"}[risk_cat]
        ap_border={"High Risk":"rgba(239,68,68,.5)","Medium Risk":"rgba(245,158,11,.4)","Low Risk":"rgba(16,185,129,.4)"}[risk_cat]
        st.markdown(f"""<div style="background:{ap_color};border:1px solid {ap_border};border-radius:12px;
            padding:.85rem 1.1rem;margin-top:.6rem">
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.82rem;margin-bottom:.35rem">
                ⚡ Recommended Next Step</div>
            <div style="font-size:.82rem;color:#b0bfd8">{action_plan['immediate']}</div>
            <div style="font-size:.76rem;color:#8899b8;margin-top:.3rem">
                Short-term: {action_plan['short_term']}</div>
        </div>""",unsafe_allow_html=True)

        # Downloads
        report_json=json.dumps({"generated":datetime.datetime.now().isoformat(),"model":best_name,
            "customer":customer,"churn_probability":f"{prob:.4f}","risk_category":risk_cat,
            "predicted_churn":bool(label),"segment":seg_meta["name"],
            "strategies":[{"title":s["title"],"description":s["description"]} for s in strategies[:3]],
            "action_plan":{k:v for k,v in action_plan.items() if k not in("color","emoji")}},indent=2)
        st.markdown("<div style='height:.4rem'></div>",unsafe_allow_html=True)
        dl1,dl2=st.columns(2)
        with dl1:
            st.download_button("⬇️  JSON Report",data=report_json,
                file_name=f"churnguard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",use_container_width=True)
        with dl2:
            try:
                pdf=generate_pdf_report(customer,prob,risk_cat,strategies,action_plan,best_name)
                st.download_button("📄  PDF Report",data=pdf,
                    file_name=f"churnguard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",use_container_width=True)
            except Exception:
                st.caption("📄 PDF export not available in this environment.")


# ═══════════════════════════════════════════════════════════════════════════════
def page_analytics(df):
    sh("📊","Customer Analytics")
    st.caption("Key churn drivers across products, activity, card type, and customer tenure.")

    # Row 1 ─ Products & Activity
    c1,c2=st.columns(2)
    with c1:
        pc=df.groupby("NumOfProducts")["Churn"].mean().reset_index(); pc.columns=["Products","Rate"]
        fig=px.bar(pc,x="Products",y="Rate",color="Rate",color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                   text=pc["Rate"].map("{:.1%}".format))
        fig.update_traces(textposition="outside",textfont_color="#e8f0ff"); fig.update_coloraxes(showscale=False)
        st.plotly_chart(sf(fig,"Churn Rate by Number of Products"),use_container_width=True)
    with c2:
        ac=df.groupby("IsActiveMember")["Churn"].mean().reset_index()
        ac["Label"]=ac["IsActiveMember"].map({0:"Inactive",1:"Active"})
        fig=px.bar(ac,x="Label",y="Churn",color="Label",
                   color_discrete_map={"Active":"#10b981","Inactive":"#ef4444"},
                   text=ac["Churn"].map("{:.1%}".format))
        fig.update_traces(textposition="outside",textfont_color="#e8f0ff"); fig.update_layout(showlegend=False)
        st.plotly_chart(sf(fig,"Active vs Inactive Member Churn Rate"),use_container_width=True)

    # Row 2 ─ Card Type & Tenure
    c1,c2=st.columns(2)
    with c1:
        cc=df.groupby("CardType")["Churn"].mean().reset_index().sort_values("Churn"); cc.columns=["Card","Rate"]
        fig=px.bar(cc,x="Card",y="Rate",color_discrete_sequence=["#7c3aed"],text=cc["Rate"].map("{:.1%}".format))
        fig.update_traces(textposition="outside",textfont_color="#e8f0ff")
        st.plotly_chart(sf(fig,"Churn Rate by Card Type"),use_container_width=True)
    with c2:
        tc=df.groupby("Tenure")["Churn"].mean().reset_index(); tc.columns=["Tenure","Rate"]
        fig=px.bar(tc,x="Tenure",y="Rate",color="Rate",color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                   text=tc["Rate"].map("{:.1%}".format))
        fig.update_traces(textposition="outside",textfont_color="#e8f0ff"); fig.update_coloraxes(showscale=False)
        st.plotly_chart(sf(fig,"Churn Rate by Tenure (Years)"),use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
def page_segmentation(df):
    sh("🧩","Customer Segmentation")
    st.caption("K-Means (k=5) clusters customers into 5 behavioural archetypes. Each card shows segment size and churn risk.")
    with st.spinner("Loading segmentation…"):
        labels,_,summary=get_seg_data(df)

    # ── Segment Summary Cards ─────────────────────────────────────────────────
    for col,(seg_id,meta) in zip(st.columns(5),SEGMENT_META.items()):
        row=summary[summary["Segment"]==seg_id]
        cnt=int(row["Count"].values[0]) if not row.empty else 0
        rate=f"{row['ChurnRate'].values[0]:.1%}" if not row.empty else "–"
        col.markdown(f"""<div class="cg-card" style="text-align:center;padding:1rem">
        <div style="font-size:1.6rem">{meta['icon']}</div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.8rem;color:{meta['color']};margin:.2rem 0">{meta['name']}</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800">{cnt:,}</div>
        <div style="font-size:.7rem;color:#8899b8">customers</div>
        <div style="font-size:.77rem;color:#ef4444;margin-top:.2rem">{rate} churn</div></div>""",unsafe_allow_html=True)

    st.markdown("<div style='height:.4rem'></div>",unsafe_allow_html=True)

    # ── Two-chart row: Segment churn + Age-group churn ────────────────────────
    c1,c2=st.columns(2)
    with c1:
        sc=summary[["SegmentName","ChurnRate","Count"]].sort_values("ChurnRate",ascending=False)
        clr_map={m["name"]:m["color"] for m in SEGMENT_META.values()}
        fig=px.bar(sc,x="SegmentName",y="ChurnRate",color="SegmentName",color_discrete_map=clr_map,
                   text=sc["ChurnRate"].map("{:.1%}".format))
        fig.update_traces(textposition="outside",textfont_color="#e8f0ff"); fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=-15)
        st.plotly_chart(sf(fig,"Churn Rate by Customer Segment",400),use_container_width=True)
    with c2:
        d2=df.copy()
        d2["AgeGroup"]=pd.cut(d2["Age"],[18,25,35,45,55,65,100],
            labels=["18–24","25–34","35–44","45–54","55–64","65+"],right=False)
        ag=d2.groupby("AgeGroup",observed=True)["Churn"].mean().reset_index(); ag.columns=["Age Group","Churn Rate"]
        fig=px.bar(ag,x="Age Group",y="Churn Rate",color="Churn Rate",
                   color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                   text=ag["Churn Rate"].map("{:.1%}".format))
        fig.update_traces(textposition="outside",textfont_color="#e8f0ff"); fig.update_coloraxes(showscale=False)
        st.plotly_chart(sf(fig,"Churn Rate by Age Group",400),use_container_width=True)

    # ── Segment Descriptions ──────────────────────────────────────────────────
    st.markdown("##### Segment Profiles")
    for col,(seg_id,meta) in zip(st.columns(5),SEGMENT_META.items()):
        col.markdown(f"""<div class="seg-card" style="border-top:3px solid {meta['color']}">
        <div class="seg-icon">{meta['icon']}</div>
        <div class="seg-name" style="color:{meta['color']}">{meta['name']}</div>
        <div class="seg-desc">{meta['description']}</div>
        <div class="seg-action">→ {meta['action']}</div></div>""",unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
def page_model_performance(results, best_name):
    sh("🏆","Model Performance")
    if results is None:
        st.warning("Run `python train_model.py` first."); return
    mres={k:v for k,v in results.items() if not k.startswith("_")}
    br=mres.get(best_name, list(mres.values())[0])

    # ── Active Model Banner ───────────────────────────────────────────────────
    st.markdown(f"""<div style="background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.22);
        border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:1.1rem;display:flex;align-items:center;gap:1rem">
        <div style="font-size:2rem">🤖</div>
        <div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.15rem;color:#00d4ff">{best_name}</div>
            <div style="font-size:.8rem;color:#8899b8;margin-top:.2rem">
                Active prediction model &nbsp;·&nbsp; Trained on banking churn dataset &nbsp;·&nbsp;
                Selected for highest AUC-ROC on held-out test set
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── 4 Key Metric Cards ────────────────────────────────────────────────────
    metrics=[
        ("Accuracy",  br["accuracy"],  "#00d4ff",
         "Of all customers tested, the model correctly predicted churn or retention for this share."),
        ("Precision", br["precision"], "#7c3aed",
         "When the model flags a customer as likely to churn, it is correct this proportion of the time."),
        ("Recall",    br["recall"],    "#f59e0b",
         "Share of actual churners the model successfully identifies — critical for retention targeting."),
        ("F1 Score",  br["f1"],        "#10b981",
         "Harmonic mean of Precision and Recall — the primary balanced performance metric."),
    ]
    for col,(label,val,clr,desc) in zip(st.columns(4),metrics):
        col.markdown(f"""<div class="cg-card" style="text-align:center">
            <div class="cg-card-title">{label}</div>
            <div class="cg-card-value" style="color:{clr}">{val:.2%}</div>
            <div style="font-size:.72rem;color:#8899b8;margin-top:.45rem;line-height:1.45">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # ── Plain-Language Model Description ─────────────────────────────────────
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""<div style="background:rgba(14,22,44,.85);border:1px solid rgba(0,212,255,.15);
        border-radius:14px;padding:1.1rem 1.4rem">
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.85rem;
             color:#e8f0ff;margin-bottom:.55rem">How this model works</div>
        <div style="font-size:.82rem;color:#b0bfd8;line-height:1.65">
            <b style="color:#00d4ff">{best_name}</b> is a supervised classification algorithm trained on
            historical banking records. Given a customer's demographics, financial profile, and behavioural
            signals, it outputs a <b style="color:#e8f0ff">churn probability between 0 % and 100 %</b>.
            Customers above the threshold are flagged as at-risk, enabling the retention team to intervene
            proactively. The model was evaluated on a held-out test set it has never seen during training,
            ensuring the reported accuracy reflects real-world performance rather than memorisation.
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
def page_dataset(df):
    sh("📂","Dataset Explorer")
    with st.expander("🔍  Filters & Search",expanded=False):
        fc1,fc2,fc3=st.columns(3)
        with fc1: geo_f=st.multiselect("Geography",df["Geography"].unique().tolist(),default=df["Geography"].unique().tolist())
        with fc2: ch_f=st.multiselect("Churn",[0,1],default=[0,1],format_func=lambda x:"Churned" if x else "Retained")
        with fc3: age_r=st.slider("Age Range",int(df["Age"].min()),int(df["Age"].max()),(int(df["Age"].min()),int(df["Age"].max())))
        sid=st.text_input("Search Customer ID",placeholder="e.g. CG0000123")

    mask=df["Geography"].isin(geo_f)&df["Churn"].isin(ch_f)&df["Age"].between(*age_r)
    if sid: mask&=df["CustomerID"].str.contains(sid.upper(),na=False)
    filtered=df[mask]
    churn_rate = f"{filtered['Churn'].mean():.2%}" if not filtered.empty else "—"
    for col,(lbl,val,clr) in zip(st.columns(4),[
        ("Filtered Rows",f"{len(filtered):,}","#00d4ff"),("Columns",f"{len(df.columns)}","#7c3aed"),
        ("Filtered Churn Rate",churn_rate,"#ef4444"),
        ("Missing Values",f"{filtered.isnull().sum().sum()}","#10b981")]):
        col.markdown(f"<div class='cg-card' style='text-align:center'>"
                     f"<div class='cg-card-value' style='color:{clr}'>{val}</div>"
                     f"<div class='cg-card-delta'>{lbl}</div></div>",unsafe_allow_html=True)

    st.caption(
        f"Source: {len(df):,} customers. Filters match {len(filtered):,}; "
        f"displaying {min(len(filtered), 200):,} (maximum 200)."
    )
    st.dataframe(filtered.head(200),use_container_width=True,height=420)
    with st.expander("📈  Descriptive Statistics (filtered rows)"):
        st.dataframe(filtered.describe().T.round(3),use_container_width=True)
    st.download_button("⬇️  Download Filtered Data (CSV)",data=filtered.to_csv(index=False).encode(),
        file_name=f"churnguard_export_{datetime.datetime.now().strftime('%Y%m%d')}.csv",mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
def page_history():
    sh("📋","Prediction History")
    history=st.session_state.prediction_history
    if not history:
        st.info("No predictions yet. Go to **🔮  Predict** to get started."); return

    st.caption(f"{len(history)} prediction(s) this session")
    rows=[{"Timestamp":h["timestamp"],"Age":h["customer"]["Age"],"Gender":h["customer"]["Gender"],
           "Geography":h["customer"]["Geography"],"Credit Score":h["customer"]["CreditScore"],
           "Balance":f"${h['customer']['Balance']:,.0f}","Products":h["customer"]["NumOfProducts"],
           "Segment":h.get("segment","–"),"Churn Prob":f"{h['probability']:.1%}",
           "Risk":h["risk_category"],"Predicted":"YES ⚠️" if h["label"] else "NO ✅"}
          for h in reversed(history)]
    hist_df=pd.DataFrame(rows)
    st.dataframe(hist_df,use_container_width=True,height=360)

    if len(history)>=3:
        rc=hist_df["Risk"].value_counts().reset_index(); rc.columns=["Risk","Count"]
        fig=px.pie(rc,values="Count",names="Risk",color="Risk",hole=.52,
                   color_discrete_map={"Low Risk":"#10b981","Medium Risk":"#f59e0b","High Risk":"#ef4444"})
        st.plotly_chart(sf(fig,"Session Risk Distribution",290),use_container_width=True)

    c1,c2=st.columns(2)
    with c1: st.download_button("⬇️  Export CSV",data=hist_df.to_csv(index=False).encode(),
        file_name=f"history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",mime="text/csv")
    with c2:
        if st.button("🗑️  Clear History"): st.session_state.prediction_history=[]; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
_SYS="""You are ChurnGuard AI Assistant — an expert banking analytics advisor.
You specialise in: churn prediction models, banking customer behaviour,
retention strategy design, and interpreting ML metrics (AUC-ROC, F1, precision/recall).
Be concise, professional, and helpful. Use bullet points where appropriate.
Never invent statistics. If asked about a customer's prediction, use context provided."""

def page_chatbot():
    sh("🤖","AI Chatbot — Churn Insights Assistant")
    st.caption("Ask anything about churn drivers, retention tactics, model interpretation, or banking analytics.")

    with st.expander("⚙️  API Key",expanded=not bool(st.session_state.get("_ckey",""))):
        st.caption("Get your key at [console.anthropic.com](https://console.anthropic.com)")
        ak=st.text_input("Anthropic API Key",type="password",value=st.session_state.get("_ckey",""),placeholder="sk-ant-…")
        if st.button("Save Key"): st.session_state["_ckey"]=ak; st.success("Saved.")

    ak=st.session_state.get("_ckey","")
    if not ANTHROPIC_AVAILABLE:
        st.warning("Run `pip install anthropic` to enable the chatbot."); return

    st.markdown("**💡 Quick prompts:**")
    q1,q2,q3,q4=st.columns(4)
    for col,prompt in[(q1,"What factors most predict churn?"),(q2,"Explain AUC-ROC in simple terms."),
                      (q3,"Best retention for high-risk seniors?"),(q4,"How does K-Means segment customers?")]:
        if col.button(prompt[:32]+"…",use_container_width=True):
            st.session_state.chat_history.append({"role":"user","content":prompt})

    st.markdown("<hr class='cg-div'>",unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"]=="user":
            st.markdown(f"<div class='chat-lbl'>👤 You</div><div class='chat-user'>{msg['content']}</div>",unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-lbl'>🛡️ ChurnGuard AI</div><div class='chat-ai'>{msg['content']}</div>",unsafe_allow_html=True)

    ui=st.text_input("Message",placeholder="Ask about churn, customers, or retention…",label_visibility="collapsed",key="ci")
    sc,cc=st.columns([5,1])
    with sc: send=st.button("Send ➤",use_container_width=True)
    with cc:
        if st.button("Clear",use_container_width=True): st.session_state.chat_history=[]; st.rerun()

    if send and ui.strip():
        if not ak:
            st.error("Enter your Anthropic API key above."); return
        st.session_state.chat_history.append({"role":"user","content":ui.strip()})
        msgs=list(st.session_state.chat_history)
        if st.session_state.last_prediction:
            lp=st.session_state.last_prediction
            ctx=(f"[Context: Last prediction — Age {lp['customer']['Age']}, "
                 f"Geography {lp['customer']['Geography']}, "
                 f"Prob {lp['probability']:.1%}, Risk {lp['risk_category']}, "
                 f"Segment {lp.get('segment','–')}] ")
            msgs[-1]={"role":"user","content":ctx+msgs[-1]["content"]}
        with st.spinner("Thinking…"):
            try:
                client=_anthropic.Anthropic(api_key=ak)
                resp=client.messages.create(model="claude-sonnet-4-20250514",max_tokens=700,
                    system=_SYS,messages=msgs)
                reply=resp.content[0].text
                st.session_state.chat_history[-1]={"role":"user","content":ui.strip()}
                st.session_state.chat_history.append({"role":"assistant","content":reply})
                st.rerun()
            except Exception as e:
                st.error(f"API error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
def build_sidebar(best_name):
    with st.sidebar:
        st.markdown("""<div style="text-align:center;padding:1rem 0 .5rem">
        <div style="font-size:2.1rem">🛡️</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.08rem;font-weight:800;
             background:linear-gradient(135deg,#00d4ff,#7c3aed);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent">ChurnGuard AI</div>
        <div style="font-size:.68rem;color:#8899b8;margin-top:2px">Banking Intelligence Platform</div></div>
        <hr style="border:none;border-top:1px solid rgba(0,212,255,.15);margin:.65rem 0">""",unsafe_allow_html=True)

        page=st.radio("nav",["🏠  Home","🔮  Predict","📊  Analytics","🧩  Segmentation",
                              "🏆  Model Performance","📂  Dataset","📋  History","🤖  AI Chatbot"],
                       label_visibility="collapsed")

        st.markdown("<hr style='border:none;border-top:1px solid rgba(0,212,255,.15);margin:.65rem 0'>",unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.15);border-radius:10px;padding:.72rem;font-size:.77rem;margin-bottom:.55rem">
            <div style="color:#8899b8;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px">Active Model</div>
            <div style="color:#00d4ff;font-weight:600">{best_name}</div>
            <div style="color:#8899b8;font-size:.68rem;margin-top:2px">⭐ Best AUC on Test Set</div>
        </div>
        <div style="background:rgba(124,58,237,.07);border:1px solid rgba(124,58,237,.2);border-radius:10px;padding:.72rem;font-size:.77rem">
            <div style="color:#8899b8;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px">Session</div>
            <div style="color:#e8f0ff">{len(st.session_state.prediction_history)} prediction(s)</div>
            <div style="color:#8899b8;font-size:.68rem;margin-top:2px">Role: {st.session_state.get('user_role','Guest')}</div>
        </div>""",unsafe_allow_html=True)
        st.markdown("<div style='height:.4rem'></div>",unsafe_allow_html=True)
        if st.button("🚪  Logout",use_container_width=True):
            st.session_state.authenticated=False; st.rerun()
    return page.strip()


def render_footer():
    st.markdown("""<div class="cg-footer">
    🛡️ ChurnGuard AI &nbsp;·&nbsp; AI-Powered Banking Intelligence &nbsp;·&nbsp;
    Streamlit · Scikit-learn · XGBoost · Plotly · ReportLab
    <br><span style="font-size:.67rem">© 2024 · Hackathon Edition · All rights reserved</span></div>""",unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    init_state(); inject_css()
    if not st.session_state.authenticated:
        page_login(); return
    with st.spinner("⚙️ Loading AI models…"):
        try: model,scaler,feature_names,best_name=boot_models()
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.info("Run `python train_model.py` first."); return
    df=get_dataset(); results=get_training_results()
    page=build_sidebar(best_name)
    if   page.startswith("🏠"): page_home(df)
    elif page.startswith("🔮"): page_predict(model,scaler,feature_names,best_name)
    elif page.startswith("📊"): page_analytics(df)
    elif page.startswith("🧩"): page_segmentation(df)
    elif page.startswith("🏆"): page_model_performance(results,best_name)
    elif page.startswith("📂"): page_dataset(df)
    elif page.startswith("📋"): page_history()
    elif page.startswith("🤖"): page_chatbot()
    render_footer()

if __name__=="__main__":
    main()
