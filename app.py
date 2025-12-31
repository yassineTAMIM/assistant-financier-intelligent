import streamlit as st
import os
from pathlib import Path

# Chemins (robustes quel que soit le répertoire de lancement)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

from src.rag import FinancialRAG
from src.agent import FinancialAgent
from src.llm import FinancialLLM

from langchain_community.document_loaders import PyPDFLoader

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


# Configuration de la page
st.set_page_config(
    page_title="Assistant Financier Intelligent",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé amélioré
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 0.5rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    /* Bulles de chat */
    .user-message {
        background-color: #e6f4ff;
        border-radius: 15px;
        padding: 12px;
        margin: 10px 0;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #f0f2f6;
        border-radius: 15px;
        padding: 12px;
        margin: 10px 0;
        text-align: left;
        max-width: 80%;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Initialisation des états de session
if 'rag' not in st.session_state:
    st.session_state.rag = None
if 'agent' not in st.session_state:
    st.session_state.agent = FinancialAgent()
if 'llm' not in st.session_state:
    st.session_state.llm = FinancialLLM(model_name="mistral")  # ou "phi3:mini" si tu préfères
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore_loaded' not in st.session_state:
    st.session_state.vectorstore_loaded = False

# Header
st.markdown('<h1 class="main-header">Assistant Financier Intelligent</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar 
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📚 Base de Connaissances RAG")
    
    if st.button("🔄 Charger le Vectorstore"):
        with st.spinner("Chargement du vectorstore..."):
            try:
                if st.session_state.rag is None:
                    st.session_state.rag = FinancialRAG()
                
                if os.path.exists("vectorstore"):
                    st.session_state.rag.load_vectorstore("vectorstore")
                    st.session_state.vectorstore_loaded = True
                    st.success("✅ Vectorstore chargé!")
                else:
                    st.error("❌ Vectorstore introuvable. Exécutez d'abord: python rag_test.py")
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
    
    if st.button("🗃️ Reconstruire l'index"):
        with st.spinner("Construction de l'index..."):
            try:
                if st.session_state.rag is None:
                    st.session_state.rag = FinancialRAG()
                st.session_state.rag.build_index("data/", save=True)
                st.session_state.vectorstore_loaded = True
                st.success("✅ Index construit!")
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
    
    if st.session_state.vectorstore_loaded:
        st.success("🟢 RAG Opérationnel")
    else:
        st.warning("🟡 RAG non chargé")
    
    st.markdown("---")
    
    st.subheader("🤖 Agent Financier")
    st.success("🟢 Agent Opérationnel")
    
    st.markdown("---")
    
    st.subheader("🧠 LLM")
    if st.session_state.llm.is_available():
        st.success(f"🟢 Ollama ({st.session_state.llm.model_name})")
    else:
        st.warning("🟡 Mode Fallback (sans Ollama)")
        st.info("💡 Installez Ollama pour des réponses plus intelligentes")
    
    st.markdown("---")
    
    st.subheader("ℹ️ À propos")
    st.info("""
    **Assistant Financier Intelligent**
    
    Combine:
    - 🔍 RAG pour l'analyse de documents
    - 🤖 Agent pour les calculs financiers
    - 💬 Interface conversationnelle
    
    Technologies: LangChain, FAISS, YFinance, Streamlit
    """)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Intelligent", "📊 Analyse d'Entreprise", "📈 Comparaison", "📄 Documents"])

# TAB 1: Chat Intelligent
with tab1:
    st.header("💬 Chat avec l'Assistant")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Conversation")
        
        if st.session_state.chat_history:
            chat_container = st.container(height=600)
            with chat_container:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f'<div class="user-message"><strong>👤 Vous :</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="assistant-message"><strong>🤖 Assistant :</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.info("💡 Commencez la conversation en posant une question ci-dessous !")
        
        user_input = st.text_input("Posez votre question sur les documents financiers :", key="chat_input")
        
        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            send_button = st.button("Envoyer 📤", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Effacer l'historique", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        if send_button and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            
            with st.spinner("Recherche dans les documents et génération de la réponse..."):
                try:
                    context = ""
                    if st.session_state.vectorstore_loaded and st.session_state.rag:
                        results = st.session_state.rag.search(user_input, k=5)
                        context_parts = []
                        for doc in results:
                            source = doc.metadata.get('source_file', 'Document inconnu')
                            page = doc.metadata.get('page', '?')
                            content = doc.page_content.strip()
                            context_parts.append(f"[Source : {source} - Page {page}]\n{content}")
                        context = "\n\n".join(context_parts)
                    else:
                        st.warning("Vectorstore non chargé.")
                    
                    response = st.session_state.llm.generate_response(user_input.strip(), context)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Erreur: {e}")
                    st.session_state.chat_history.append({"role": "assistant", "content": "Désolé, une erreur est survenue."})
            
            st.rerun()
    
    with col2:
        st.subheader("📚 Documents Chargés")
        if os.path.exists("data"):
            pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
            if pdf_files:
                st.success(f"✅ {len(pdf_files)} rapports chargés")
                st.markdown("**Détails :**")
                total_pages = 0
                for f in pdf_files:
                    try:
                        loader = PyPDFLoader(os.path.join("data", f))
                        pages = loader.load()
                        total_pages += len(pages)
                    except:
                        pass
                st.markdown(f"- Total pages indexées : {total_pages}")
                st.markdown(f"- Exemples : {', '.join(pdf_files[:3])}...")
            else:
                st.warning("Aucun PDF dans 'data/'")
        else:
            st.error("Dossier 'data/' manquant")

# TAB 2: Analyse d'Entreprise 
with tab2:
    st.header("📊 Analyse Détaillée d'une Entreprise")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        ticker = st.text_input("Symbole boursier (ex: AAPL, MSFT, GOOGL)", value="AAPL")
        analyze_button = st.button("🔍 Analyser", key="analyze_single")
    
    if analyze_button and ticker:
        with st.spinner(f"Analyse de {ticker}..."):
            try:
                info = st.session_state.agent.get_company_info(ticker)
                
                if info:
                    st.markdown(f"## 🏢 {info['name']}")
                    st.markdown(f"**Secteur:** {info['sector']} | **Industrie:** {info['industry']}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("💰 Cap. Boursière", st.session_state.agent._format_number(info['market_cap']))
                    
                    with col2:
                        st.metric("📈 Revenus", st.session_state.agent._format_number(info['revenue']))
                    
                    with col3:
                        st.metric("💵 Bénéfice Net", st.session_state.agent._format_number(info['net_income']))
                    
                    with col4:
                        st.metric("💳 Prix Actuel", f"${info['current_price']:.2f}")
                    
                    st.markdown("---")
                    
                    st.subheader("📊 Ratios Financiers")
                    
                    ratios = st.session_state.agent.get_all_ratios(ticker)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        roe = ratios['roe'] if ratios['roe'] else "N/A"
                        st.metric("ROE", f"{roe:.2f}%" if roe != "N/A" else roe)
                    
                    with col2:
                        roa = ratios['roa'] if ratios['roa'] else "N/A"
                        st.metric("ROA", f"{roa:.2f}%" if roa != "N/A" else roa)
                    
                    with col3:
                        de = ratios['debt_to_equity'] if ratios['debt_to_equity'] else "N/A"
                        st.metric("Dette/Equity", f"{de:.2f}" if de != "N/A" else de)
                    
                    with col4:
                        margin = ratios['profit_margin'] if ratios['profit_margin'] else "N/A"
                        st.metric("Marge Nette", f"{margin:.2f}%" if margin != "N/A" else margin)
                    
                    st.markdown("---")
                    
                    st.subheader("📈 Performance Historique")
                    
                    period = st.selectbox("Période", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
                    
                    hist = st.session_state.agent.get_stock_history(ticker, period)
                    
                    if not hist.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            mode='lines',
                            name='Prix de clôture',
                            line=dict(color='#1f77b4', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"Évolution du prix de {ticker}",
                            xaxis_title="Date",
                            yaxis_title="Prix ($)",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        perf = st.session_state.agent.analyze_performance(ticker, period)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("📊 Rendement", f"{perf['return_pct']}%")
                        
                        with col2:
                            st.metric("📉 Volatilité", f"{perf['volatility']}%")
                        
                        with col3:
                            st.metric("📈 Min / Max", f"${perf['min_price']:.2f} / ${perf['max_price']:.2f}")

            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {e}")

# TAB 3: Comparaison 
with tab3:
    st.header("🔍 Comparaison d'Entreprises")
    
    st.markdown("Comparez les performances de plusieurs entreprises")
    
    tickers_input = st.text_input(
        "Symboles boursiers (séparés par des virgules)",
        value="AAPL,MSFT,GOOGL",
        help="Exemple: AAPL,MSFT,GOOGL,AMZN"
    )
    
    compare_button = st.button("📊 Comparer", key="compare_btn")
    
    if compare_button and tickers_input:
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        
        with st.spinner(f"Comparaison de {len(tickers)} entreprises..."):
            try:
                df_comparison = st.session_state.agent.compare_companies(tickers)
                
                st.subheader("📋 Tableau Comparatif")
                st.dataframe(df_comparison, use_container_width=True)
                
                # Export comparison
                csv_comp = df_comparison.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger Comparaison CSV",
                    data=csv_comp,
                    file_name=f"comparaison_{len(tickers)}_entreprises.csv",
                    mime="text/csv"
                )
                
                st.markdown("---")
                
                st.subheader("📊 Visualisations Comparatives")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_margin = px.bar(
                        df_comparison,
                        x='Ticker',
                        y='Marge Nette (%)',
                        title='Comparaison des Marges Nettes',
                        color='Marge Nette (%)',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_margin, use_container_width=True)
                
                with col2:
                    # ROE vs ROA Comparison
                    fig_roe_roa = go.Figure()
                    fig_roe_roa.add_trace(go.Bar(
                        x=df_comparison['Ticker'],
                        y=df_comparison['ROE (%)'],
                        name='ROE (%)',
                        marker_color='lightblue'
                    ))
                    fig_roe_roa.add_trace(go.Bar(
                        x=df_comparison['Ticker'],
                        y=df_comparison['ROA (%)'],
                        name='ROA (%)',
                        marker_color='lightcoral'
                    ))
                    fig_roe_roa.update_layout(
                        title='Comparaison ROE vs ROA',
                        barmode='group',
                        yaxis_title='Pourcentage (%)'
                    )
                    st.plotly_chart(fig_roe_roa, use_container_width=True)
                
                # Second row
                col3, col4 = st.columns(2)
                
                with col3:
                    if not df_comparison['Dette/Equity'].isna().all():
                        fig_de = px.bar(
                            df_comparison.dropna(subset=['Dette/Equity']),
                            x='Ticker',
                            y='Dette/Equity',
                            title='Comparaison Dette/Equity',
                            color='Dette/Equity',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_de, use_container_width=True)
                    else:
                        st.info("Données Dette/Equity non disponibles")
                
                with col4:
                    # Fixed Radar Chart - properly get values and handle NaN
                    fig_radar = go.Figure()
                    for _, row in df_comparison.iterrows():
                        # Get values and replace NaN with 0
                        roe_val = row.get('ROE (%)', 0)
                        roa_val = row.get('ROA (%)', 0)
                        de_val = row.get('Dette/Equity', 0)
                        margin_val = row.get('Marge Nette (%)', 0)
                        
                        # Replace NaN with 0
                        roe_val = 0 if pd.isna(roe_val) else roe_val
                        roa_val = 0 if pd.isna(roa_val) else roa_val
                        de_val = 0 if pd.isna(de_val) else de_val
                        margin_val = 0 if pd.isna(margin_val) else margin_val
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[roe_val, roa_val, de_val, margin_val],
                            theta=['ROE (%)', 'ROA (%)', 'Dette/Equity', 'Marge Nette (%)'],
                            fill='toself',
                            name=row['Ticker']
                        ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max(df_comparison[['ROE (%)', 'ROA (%)', 'Dette/Equity', 'Marge Nette (%)']].max()) * 1.1]
                            )
                        ),
                        title="Comparaison des Ratios (Radar)",
                        showlegend=True
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                st.subheader("📈 Performance Comparative (6 mois)")
                
                fig_perf = go.Figure()
                for ticker in tickers:
                    try:
                        hist = st.session_state.agent.get_stock_history(ticker, "6mo")
                        if not hist.empty:
                            hist['Normalized'] = (hist['Close'] / hist['Close'].iloc[0]) * 100
                            fig_perf.add_trace(go.Scatter(x=hist.index, y=hist['Normalized'], name=ticker, mode='lines'))
                    except:
                        pass
                
                if len(fig_perf.data) > 0:
                    fig_perf.update_layout(title="Performance normalisée (6 mois)", yaxis_title="Prix normalisé (base 100)")
                    st.plotly_chart(fig_perf, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la comparaison: {e}")

# TAB 4: Documents (amélioré légèrement)
with tab4:
    st.header("📄 Gestion des Documents")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Documents Disponibles")
        
        if os.path.exists("data"):
            pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
            
            if pdf_files:
                st.success(f"✅ {len(pdf_files)} documents trouvés")
                
                for pdf in pdf_files:
                    st.markdown(f"- 📄 {pdf}")
            else:
                st.warning("Aucun document trouvé dans le dossier 'data/'")
        else:
            st.error("Le dossier 'data/' n'existe pas")
    
    with col2:
        st.subheader("🔎 Recherche dans les Documents")
        
        if st.session_state.vectorstore_loaded:
            search_query = st.text_input("Rechercher dans les documents:")
            
            if st.button("🔎 Rechercher"):
                if search_query:
                    with st.spinner("Recherche en cours..."):
                        try:
                            results = st.session_state.rag.search_with_scores(search_query, k=5)
                            
                            st.subheader("📊 Résultats")
                            
                            for i, (doc, score) in enumerate(results, 1):
                                with st.expander(f"Résultat {i} - Score: {score:.4f}"):
                                    st.markdown(f"**Source:** {doc.metadata.get('source_file', 'Unknown')}")
                                    st.markdown(f"**Page:** {doc.metadata.get('page', 'Unknown')}")
                                    st.markdown(f"**Contenu:**")
                                    st.text(doc.page_content[:500] + "...")
                        except Exception as e:
                            st.error(f"Erreur: {e}")
        else:
            st.warning("Veuillez d'abord charger le vectorstore")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💼 Assistant Financier Intelligent | Powered by LangChain, FAISS & YFinance</p>
    <p>Développé par Yassine TAMIM & Zakaria LIMI | Décembre 2025</p>
</div>
""", unsafe_allow_html=True)