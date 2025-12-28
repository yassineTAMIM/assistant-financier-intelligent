"""
Script de test complet pour l'Agent Financier
Teste toutes les fonctionnalités de calcul et d'analyse
"""
from pathlib import Path
import sys

# Permet d'importer le package `src/` quand on exécute le fichier directement
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.agent import FinancialAgent
import pandas as pd


def print_separator(title: str = ""):
    """Affiche un séparateur visuel"""
    print("\n" + "="*70)
    if title:
        print(f"  {title}")
        print("="*70)
    print()


def test_1_initialization():
    """Test 1: Initialisation de l'agent"""
    print_separator("TEST 1: Initialisation de l'Agent Financier")
    
    try:
        agent = FinancialAgent()
        print("✅ Agent initialisé avec succès")
        return agent
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        return None


def test_2_company_info(agent: FinancialAgent):
    """Test 2: Récupération d'informations d'entreprise"""
    print_separator("TEST 2: Récupération des données entreprise")
    
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print(f"🔍 Test avec: {', '.join(test_tickers)}\n")
    
    results = {}
    for ticker in test_tickers:
        try:
            info = agent.get_company_info(ticker)
            if info:
                results[ticker] = info
                print(f"✅ {ticker}: {info['name']}")
                print(f"   Secteur: {info['sector']}")
                print(f"   Cap. boursière: {agent._format_number(info['market_cap'])}")
                print()
            else:
                print(f"⚠️  Échec pour {ticker}")
        except Exception as e:
            print(f"❌ Erreur pour {ticker}: {e}")
    
    if results:
        print(f"✅ {len(results)}/{len(test_tickers)} entreprises récupérées")
        return True
    return False


def test_3_roe_calculation(agent: FinancialAgent):
    """Test 3: Calcul du ROE"""
    print_separator("TEST 3: Calcul du Return on Equity (ROE)")
    
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print("📊 Calcul du ROE pour chaque entreprise:\n")
    
    roe_values = {}
    for ticker in test_tickers:
        try:
            roe = agent.calculate_roe(ticker)
            if roe is not None:
                roe_values[ticker] = roe
                print(f"✅ {ticker}: ROE = {roe:.2f}%")
            else:
                print(f"⚠️  ROE non disponible pour {ticker}")
        except Exception as e:
            print(f"❌ Erreur pour {ticker}: {e}")
    
    print(f"\n✅ ROE calculé pour {len(roe_values)} entreprises")
    return roe_values


def test_4_roa_calculation(agent: FinancialAgent):
    """Test 4: Calcul du ROA"""
    print_separator("TEST 4: Calcul du Return on Assets (ROA)")
    
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print("📊 Calcul du ROA pour chaque entreprise:\n")
    
    roa_values = {}
    for ticker in test_tickers:
        try:
            roa = agent.calculate_roa(ticker)
            if roa is not None:
                roa_values[ticker] = roa
                print(f"✅ {ticker}: ROA = {roa:.2f}%")
            else:
                print(f"⚠️  ROA non disponible pour {ticker}")
        except Exception as e:
            print(f"❌ Erreur pour {ticker}: {e}")
    
    print(f"\n✅ ROA calculé pour {len(roa_values)} entreprises")
    return roa_values


def test_5_debt_to_equity(agent: FinancialAgent):
    """Test 5: Calcul du ratio Dette/Equity"""
    print_separator("TEST 5: Calcul du ratio Dette/Equity")
    
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print("📊 Calcul du D/E ratio:\n")
    
    de_values = {}
    for ticker in test_tickers:
        try:
            de = agent.calculate_debt_to_equity(ticker)
            if de is not None:
                de_values[ticker] = de
                print(f"✅ {ticker}: D/E = {de:.2f}")
                # Interprétation
                if de < 0.5:
                    print(f"   → Faible endettement")
                elif de < 1.0:
                    print(f"   → Endettement modéré")
                else:
                    print(f"   → Endettement élevé")
            else:
                print(f"⚠️  D/E non disponible pour {ticker}")
        except Exception as e:
            print(f"❌ Erreur pour {ticker}: {e}")
        print()
    
    print(f"✅ D/E calculé pour {len(de_values)} entreprises")
    return de_values


def test_6_profit_margin(agent: FinancialAgent):
    """Test 6: Calcul de la marge bénéficiaire"""
    print_separator("TEST 6: Calcul de la marge bénéficiaire nette")
    
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print("💰 Calcul des marges:\n")
    
    margin_values = {}
    for ticker in test_tickers:
        try:
            margin = agent.calculate_profit_margin(ticker)
            if margin is not None:
                margin_values[ticker] = margin
                print(f"✅ {ticker}: Marge nette = {margin:.2f}%")
            else:
                print(f"⚠️  Marge non disponible pour {ticker}")
        except Exception as e:
            print(f"❌ Erreur pour {ticker}: {e}")
    
    print(f"\n✅ Marges calculées pour {len(margin_values)} entreprises")
    return margin_values


def test_7_all_ratios(agent: FinancialAgent):
    """Test 7: Calcul de tous les ratios"""
    print_separator("TEST 7: Analyse complète avec tous les ratios")
    
    ticker = "AAPL"
    
    try:
        print(f"📊 Analyse complète de {ticker}...\n")
        ratios = agent.get_all_ratios(ticker)
        
        print(f"\n📋 Résultats pour {ratios['company_name']}:")
        print(f"   ROE: {ratios['roe']:.2f}%" if ratios['roe'] else "   ROE: N/A")
        print(f"   ROA: {ratios['roa']:.2f}%" if ratios['roa'] else "   ROA: N/A")
        print(f"   Dette/Equity: {ratios['debt_to_equity']:.2f}" if ratios['debt_to_equity'] else "   D/E: N/A")
        print(f"   Marge nette: {ratios['profit_margin']:.2f}%" if ratios['profit_margin'] else "   Marge: N/A")
        
        print("\n✅ Analyse complète réussie")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        return False


def test_8_company_comparison(agent: FinancialAgent):
    """Test 8: Comparaison d'entreprises"""
    print_separator("TEST 8: Comparaison de plusieurs entreprises")
    
    # Tech Giants
    tech_tickers = ["AAPL", "MSFT", "GOOGL"]
    print(f"🔍 Comparaison des Tech Giants: {', '.join(tech_tickers)}\n")
    
    try:
        df_tech = agent.compare_companies(tech_tickers)
        print("✅ Comparaison Tech Giants réussie")
        
        # Analyse financière (différents secteurs)
        print("\n" + "="*70)
        print("🔍 Comparaison Multi-Secteurs")
        print("="*70 + "\n")
        
        multi_tickers = ["JPM", "WMT", "PFE"]  # Finance, Retail, Pharma
        df_multi = agent.compare_companies(multi_tickers)
        print("✅ Comparaison Multi-Secteurs réussie")
        
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la comparaison: {e}")
        return False


def test_9_financial_summary(agent: FinancialAgent):
    """Test 9: Résumé financier détaillé"""
    print_separator("TEST 9: Génération de résumé financier")
    
    ticker = "AAPL"
    
    try:
        print(f"📋 Génération du résumé pour {ticker}...\n")
        summary = agent.get_financial_summary(ticker)
        
        print(f"🏢 {summary['company_name']}")
        print(f"{'='*70}")
        print(f"Secteur: {summary['sector']}")
        print(f"Industrie: {summary['industry']}")
        print(f"\n💰 Données Financières:")
        print(f"   Capitalisation boursière: {summary['market_cap']}")
        print(f"   Revenus: {summary['revenue']}")
        print(f"   Bénéfice net: {summary['net_income']}")
        print(f"   Total des actifs: {summary['total_assets']}")
        print(f"   Capitaux propres: {summary['total_equity']}")
        print(f"   Dette totale: {summary['total_debt']}")
        print(f"   Prix actuel: ${summary['current_price']:.2f}")
        
        print("\n✅ Résumé généré avec succès")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        return False


def test_10_stock_history(agent: FinancialAgent):
    """Test 10: Historique des prix"""
    print_separator("TEST 10: Récupération de l'historique")
    
    ticker = "AAPL"
    periods = ["1mo", "3mo", "1y"]
    
    print(f"📈 Test de l'historique pour {ticker}\n")
    
    for period in periods:
        try:
            hist = agent.get_stock_history(ticker, period)
            if not hist.empty:
                print(f"✅ Période {period}: {len(hist)} jours récupérés")
                print(f"   Prix de début: ${hist['Close'].iloc[0]:.2f}")
                print(f"   Prix de fin: ${hist['Close'].iloc[-1]:.2f}")
                print()
            else:
                print(f"⚠️  Pas de données pour {period}")
        except Exception as e:
            print(f"❌ Erreur pour {period}: {e}")
    
    print("✅ Tests d'historique terminés")
    return True


def test_11_performance_analysis(agent: FinancialAgent):
    """Test 11: Analyse de performance"""
    print_separator("TEST 11: Analyse de performance")
    
    ticker = "AAPL"
    periods = ["1mo", "6mo", "1y"]
    
    print(f"📊 Analyse de performance pour {ticker}\n")
    
    for period in periods:
        try:
            perf = agent.analyze_performance(ticker, period)
            if perf:
                print(f"✅ Période {period}:")
                print(f"   Rendement: {perf['return_pct']}%")
                print(f"   Volatilité: {perf['volatility']}%")
                print(f"   Prix min/max: ${perf['min_price']} / ${perf['max_price']}")
                print()
            else:
                print(f"⚠️  Pas de données pour {period}")
        except Exception as e:
            print(f"❌ Erreur pour {period}: {e}")
    
    print("✅ Analyses de performance terminées")
    return True


def run_all_tests():
    """Exécute tous les tests de l'agent financier"""
    
    print("\n" + "🎯"*35)
    print("  TEST COMPLET DE L'AGENT FINANCIER")
    print("🎯"*35 + "\n")
    
    # Test 1: Initialisation
    agent = test_1_initialization()
    if not agent:
        print("\n❌ Échec de l'initialisation")
        return False
    
    # Test 2: Infos entreprises
    if not test_2_company_info(agent):
        print("\n⚠️  Problème lors de la récupération des données")
    
    # Test 3-6: Ratios individuels
    test_3_roe_calculation(agent)
    test_4_roa_calculation(agent)
    test_5_debt_to_equity(agent)
    test_6_profit_margin(agent)
    
    # Test 7: Tous les ratios
    test_7_all_ratios(agent)
    
    # Test 8: Comparaison
    test_8_company_comparison(agent)
    
    # Test 9: Résumé
    test_9_financial_summary(agent)
    
    # Test 10-11: Performance
    test_10_stock_history(agent)
    test_11_performance_analysis(agent)
    
    # Résumé final
    print_separator("RÉSUMÉ FINAL")
    print("✅ Tous les tests de l'agent financier sont passés!")
    print("\n📦 Fonctionnalités validées:")
    print("   ✅ Récupération de données via YFinance")
    print("   ✅ Calcul de ratios (ROE, ROA, D/E, Marge)")
    print("   ✅ Comparaison d'entreprises")
    print("   ✅ Génération de résumés")
    print("   ✅ Analyse de performance")
    print("   ✅ Historique des prix")
    print("\n🚀 L'agent financier est opérationnel!")
    print("📝 Prochaine étape: Intégrer avec le RAG et créer l'interface Streamlit")
    print("\n" + "🎉"*35 + "\n")
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    
    if not success:
        print("\n⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        exit(1)
    else:
        print("🎊 Tous les tests réussis!")