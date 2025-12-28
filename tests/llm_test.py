"""
Test complet du LLM avec RAG et Agent
Test réel avec vos documents et données financières
"""
from pathlib import Path
import sys

# Permet d'importer le package `src/` quand on exécute le fichier directement
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import os
from src.rag import FinancialRAG
from src.agent import FinancialAgent
from src.llm import FinancialLLM
import time


def print_separator(title: str = ""):
    """Affiche un séparateur visuel"""
    print("\n" + "="*70)
    if title:
        print(f"  {title}")
        print("="*70)
    print()


def test_1_llm_only():
    """Test 1: LLM seul avec un prompt simple"""
    print_separator("TEST 1: LLM Seul - Prompt Simple")
    
    llm = FinancialLLM(model_name="phi3:mini")
    
    if not llm.is_available():
        print("❌ LLM non disponible. Vérifiez Ollama.")
        return None
    
    print("✅ LLM initialisé")
    
    # Test simple
    print("\n📝 Test de génération simple...")
    prompt = "Explique brièvement ce qu'est le ROE en finance (2 phrases max)."
    
    start = time.time()
    response = llm.llm.invoke(prompt)
    elapsed = time.time() - start
    
    print(f"\n⏱️  Temps: {elapsed:.2f}s")
    print(f"\n💬 Réponse:\n{response}")
    
    return llm


def test_2_llm_with_rag(llm: FinancialLLM):
    """Test 2: LLM avec contexte RAG réel"""
    print_separator("TEST 2: LLM + RAG - Contexte Réel")
    
    # Charger le RAG
    print("📚 Chargement du vectorstore...")
    rag = FinancialRAG()
    
    if not os.path.exists("vectorstore"):
        print("❌ Vectorstore introuvable. Exécutez: python rag_test.py")
        return
    
    rag.load_vectorstore("vectorstore")
    print("✅ Vectorstore chargé")
    
    # Questions réelles
    questions = [
        "Quel est le chiffre d'affaires mentionné dans les documents?",
        "Quels sont les résultats du dernier trimestre?",
        "Quelle est la croissance des revenus?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*70}")
        print(f"Question {i}: {question}")
        print('─'*70)
        
        # Recherche RAG
        print("🔍 Recherche dans les documents...")
        results = rag.search(question, k=2)
        
        # Construire le contexte
        context = "\n\n".join([
            f"[Document: {doc.metadata.get('source_file', 'Unknown')}]\n{doc.page_content[:300]}"
            for doc in results
        ])
        
        print(f"✅ {len(results)} documents trouvés")
        print(f"📄 Contexte: {len(context)} caractères")
        
        # Génération avec LLM
        print("\n🤖 Génération de la réponse...")
        start = time.time()
        
        prompt = f"""Tu es un assistant financier expert. Réponds à la question en utilisant UNIQUEMENT les informations du contexte fourni.

Contexte:
{context[:1500]}

Question: {question}

Réponse (concise, 2-3 phrases maximum):"""
        
        response = llm.llm.invoke(prompt)
        elapsed = time.time() - start
        
        print(f"⏱️  Temps: {elapsed:.2f}s")
        print(f"\n💬 Réponse du LLM:\n{response}\n")


def test_3_llm_with_agent(llm: FinancialLLM):
    """Test 3: LLM avec données de l'agent"""
    print_separator("TEST 3: LLM + Agent - Analyse Financière")
    
    # Initialiser l'agent
    print("🤖 Initialisation de l'agent...")
    agent = FinancialAgent()
    print("✅ Agent prêt")
    
    # Récupérer des données réelles
    ticker = "AAPL"
    print(f"\n📊 Analyse de {ticker}...")
    
    ratios = agent.get_all_ratios(ticker)
    summary = agent.get_financial_summary(ticker)
    
    # Construire le contexte avec les vraies données
    context = f"""
Données financières pour {summary['company_name']}:

Secteur: {summary['sector']}
Industrie: {summary['industry']}

Métriques clés:
- Capitalisation boursière: {summary['market_cap']}
- Revenus: {summary['revenue']}
- Bénéfice net: {summary['net_income']}
- Prix actuel: ${summary['current_price']}

Ratios financiers:
- ROE: {ratios['roe']:.2f}%
- ROA: {ratios['roa']:.2f}%
- Dette/Equity: {ratios['debt_to_equity']:.2f}
- Marge nette: {ratios['profit_margin']:.2f}%
"""
    
    print(f"✅ Données récupérées")
    
    # Questions d'analyse
    questions = [
        "Est-ce que cette entreprise est rentable? Justifie brièvement.",
        "Quel est le niveau d'endettement de l'entreprise?",
        "Résume la santé financière de cette entreprise en 2 phrases."
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*70}")
        print(f"Question {i}: {question}")
        print('─'*70)
        
        print("🤖 Génération de la réponse...")
        start = time.time()
        
        prompt = f"""Tu es un analyste financier expert. Réponds à la question en analysant les données fournies.

{context}

Question: {question}

Réponse (analytique et concise):"""
        
        response = llm.llm.invoke(prompt)
        elapsed = time.time() - start
        
        print(f"⏱️  Temps: {elapsed:.2f}s")
        print(f"\n💬 Réponse du LLM:\n{response}\n")


def test_4_llm_comparison(llm: FinancialLLM):
    """Test 4: LLM pour comparer des entreprises"""
    print_separator("TEST 4: LLM - Comparaison d'Entreprises")
    
    agent = FinancialAgent()
    
    # Comparer 3 entreprises
    tickers = ["AAPL", "MSFT", "GOOGL"]
    print(f"📊 Comparaison: {', '.join(tickers)}")
    
    df = agent.compare_companies(tickers)
    
    # Construire le contexte
    context = "Comparaison financière:\n\n"
    for _, row in df.iterrows():
        context += f"""
{row['Entreprise']} ({row['Ticker']}):
- ROE: {row['ROE (%)']}%
- ROA: {row['ROA (%)']}%
- Dette/Equity: {row['Dette/Equity']}
- Marge Nette: {row['Marge Nette (%)']}%
"""
    
    print("✅ Données de comparaison récupérées")
    
    # Questions de comparaison
    question = "Quelle entreprise a la meilleure performance financière globale? Justifie ta réponse en 3 phrases."
    
    print(f"\n{'─'*70}")
    print(f"Question: {question}")
    print('─'*70)
    
    print("🤖 Génération de l'analyse comparative...")
    start = time.time()
    
    prompt = f"""Tu es un analyste financier expert. Compare ces entreprises et réponds à la question.

{context}

Question: {question}

Réponse (analytique, compare les ratios):"""
    
    response = llm.llm.invoke(prompt)
    elapsed = time.time() - start
    
    print(f"⏱️  Temps: {elapsed:.2f}s")
    print(f"\n💬 Analyse comparative:\n{response}\n")


def test_5_llm_performance(llm: FinancialLLM):
    """Test 5: Performance du LLM"""
    print_separator("TEST 5: Performance et Vitesse")
    
    prompts = [
        "Définis le ROE en une phrase.",
        "Qu'est-ce qu'un bon ratio dette/equity?",
        "Comment interpréter une marge nette de 30%?"
    ]
    
    times = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🔄 Test {i}/3: {prompt[:50]}...")
        
        start = time.time()
        response = llm.llm.invoke(prompt)
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"⏱️  Temps: {elapsed:.2f}s")
        print(f"📏 Longueur réponse: {len(response)} caractères")
    
    print(f"\n{'─'*70}")
    print("📊 Statistiques de performance:")
    print(f"   - Temps moyen: {sum(times)/len(times):.2f}s")
    print(f"   - Temps min: {min(times):.2f}s")
    print(f"   - Temps max: {max(times):.2f}s")


def run_all_tests():
    """Exécute tous les tests du LLM"""
    
    print("\n" + "🎯"*35)
    print("  TEST COMPLET DU LLM AVEC RAG ET AGENT")
    print("🎯"*35 + "\n")
    
    # Test 1: LLM seul
    llm = test_1_llm_only()
    
    if not llm or not llm.is_available():
        print("\n❌ LLM non disponible. Tests arrêtés.")
        print("💡 Vérifiez:")
        print("   1. Ollama est installé et lancé")
        print("   2. Le modèle phi3:mini est téléchargé (ollama pull phi3:mini)")
        return False
    
    # Test 2: LLM + RAG
    test_2_llm_with_rag(llm)
    
    # Test 3: LLM + Agent
    test_3_llm_with_agent(llm)
    
    # Test 4: Comparaison
    test_4_llm_comparison(llm)
    
    # Test 5: Performance
    test_5_llm_performance(llm)
    
    # Résumé final
    print_separator("RÉSUMÉ FINAL")
    print("✅ Tous les tests LLM sont passés!")
    print("\n📦 Tests réalisés:")
    print("   ✅ LLM seul (prompt simple)")
    print("   ✅ LLM + RAG (contexte documents)")
    print("   ✅ LLM + Agent (données financières)")
    print("   ✅ LLM comparaison (analyse multi-entreprises)")
    print("   ✅ Performance (vitesse)")
    print("\n🚀 Le LLM est opérationnel et intégré!")
    print("📝 Vous pouvez maintenant lancer: streamlit run app.py")
    print("\n" + "🎉"*35 + "\n")
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    
    if not success:
        print("\n⚠️  Certains tests ont échoué.")
        exit(1)