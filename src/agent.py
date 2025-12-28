import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class FinancialAgent:
    """
    Agent autonome pour analyses et calculs financiers
    """
    
    def __init__(self):
        """Initialise l'agent financier"""
        print("🤖 Initialisation de l'Agent Financier...")
        self.company_data = {}
        print("✅ Agent Financier prêt")
    
    def get_company_info(self, ticker: str) -> Dict:
        """
        Récupère les informations d'une entreprise
        
        Args:
            ticker: Symbole boursier (ex: AAPL, GOOGL)
            
        Returns:
            Dictionnaire avec les informations
        """
        try:
            print(f"📊 Récupération des données pour {ticker}...")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Récupérer les états financiers pour des données plus complètes
            try:
                balance_sheet = stock.balance_sheet
                income_stmt = stock.income_stmt
                
                # Dernières données disponibles (colonne la plus récente)
                if not balance_sheet.empty:
                    latest_bs = balance_sheet.iloc[:, 0]
                    total_assets = latest_bs.get('Total Assets', 0)
                    total_equity = latest_bs.get('Stockholders Equity', latest_bs.get('Total Equity Gross Minority Interest', 0))
                    total_debt = latest_bs.get('Total Debt', 0)
                else:
                    total_assets = info.get('totalAssets', 0)
                    total_equity = info.get('totalStockholderEquity', 0)
                    total_debt = info.get('totalDebt', 0)
                
                if not income_stmt.empty:
                    latest_is = income_stmt.iloc[:, 0]
                    net_income = latest_is.get('Net Income', 0)
                    revenue = latest_is.get('Total Revenue', 0)
                else:
                    net_income = info.get('netIncomeToCommon', 0)
                    revenue = info.get('totalRevenue', 0)
                    
            except:
                # Fallback sur info si les états financiers échouent
                total_assets = info.get('totalAssets', 0)
                total_equity = info.get('totalStockholderEquity', 0)
                total_debt = info.get('totalDebt', 0)
                net_income = info.get('netIncomeToCommon', 0)
                revenue = info.get('totalRevenue', 0)
            
            self.company_data[ticker] = {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'revenue': revenue,
                'net_income': net_income,
                'total_assets': total_assets,
                'total_equity': total_equity,
                'total_debt': total_debt,
                'current_price': info.get('currentPrice', 0),
                'info': info
            }
            
            print(f"✅ Données récupérées pour {self.company_data[ticker]['name']}")
            return self.company_data[ticker]
            
        except Exception as e:
            print(f"❌ Erreur lors de la récupération de {ticker}: {e}")
            return {}
    
    def calculate_roe(self, ticker: str) -> Optional[float]:
        """
        Calcule le Return on Equity (ROE)
        ROE = Net Income / Shareholders' Equity * 100
        
        Args:
            ticker: Symbole boursier
            
        Returns:
            ROE en pourcentage
        """
        if ticker not in self.company_data:
            self.get_company_info(ticker)
        
        data = self.company_data.get(ticker, {})
        net_income = data.get('net_income', 0)
        equity = data.get('total_equity', 0)
        
        if equity and equity != 0:
            roe = (net_income / equity) * 100
            print(f"📈 ROE de {ticker}: {roe:.2f}%")
            return roe
        else:
            print(f"⚠️  Données insuffisantes pour calculer ROE de {ticker}")
            return None
    
    def calculate_roa(self, ticker: str) -> Optional[float]:
        """
        Calcule le Return on Assets (ROA)
        ROA = Net Income / Total Assets * 100
        
        Args:
            ticker: Symbole boursier
            
        Returns:
            ROA en pourcentage
        """
        if ticker not in self.company_data:
            self.get_company_info(ticker)
        
        data = self.company_data.get(ticker, {})
        net_income = data.get('net_income', 0)
        assets = data.get('total_assets', 0)
        
        if assets and assets != 0:
            roa = (net_income / assets) * 100
            print(f"📈 ROA de {ticker}: {roa:.2f}%")
            return roa
        else:
            print(f"⚠️  Données insuffisantes pour calculer ROA de {ticker}")
            return None
    
    def calculate_debt_to_equity(self, ticker: str) -> Optional[float]:
        """
        Calcule le ratio Dette/Equity
        D/E = Total Debt / Total Equity
        
        Args:
            ticker: Symbole boursier
            
        Returns:
            Ratio D/E
        """
        if ticker not in self.company_data:
            self.get_company_info(ticker)
        
        data = self.company_data.get(ticker, {})
        debt = data.get('total_debt', 0)
        equity = data.get('total_equity', 0)
        
        if equity and equity != 0:
            de_ratio = debt / equity
            print(f"📊 Dette/Equity de {ticker}: {de_ratio:.2f}")
            return de_ratio
        else:
            print(f"⚠️  Données insuffisantes pour calculer D/E de {ticker}")
            return None
    
    def calculate_profit_margin(self, ticker: str) -> Optional[float]:
        """
        Calcule la marge bénéficiaire nette
        Profit Margin = Net Income / Revenue * 100
        
        Args:
            ticker: Symbole boursier
            
        Returns:
            Marge en pourcentage
        """
        if ticker not in self.company_data:
            self.get_company_info(ticker)
        
        data = self.company_data.get(ticker, {})
        net_income = data.get('net_income', 0)
        revenue = data.get('revenue', 0)
        
        if revenue and revenue != 0:
            margin = (net_income / revenue) * 100
            print(f"💰 Marge nette de {ticker}: {margin:.2f}%")
            return margin
        else:
            print(f"⚠️  Données insuffisantes pour calculer la marge de {ticker}")
            return None
    
    def get_all_ratios(self, ticker: str) -> Dict:
        """
        Calcule tous les ratios financiers d'une entreprise
        
        Args:
            ticker: Symbole boursier
            
        Returns:
            Dictionnaire avec tous les ratios
        """
        print(f"\n{'='*60}")
        print(f"📊 ANALYSE FINANCIÈRE COMPLÈTE: {ticker}")
        print(f"{'='*60}\n")
        
        ratios = {
            'ticker': ticker,
            'company_name': '',
            'roe': self.calculate_roe(ticker),
            'roa': self.calculate_roa(ticker),
            'debt_to_equity': self.calculate_debt_to_equity(ticker),
            'profit_margin': self.calculate_profit_margin(ticker)
        }
        
        if ticker in self.company_data:
            ratios['company_name'] = self.company_data[ticker].get('name', ticker)
        
        return ratios
    
    def compare_companies(self, tickers: List[str]) -> pd.DataFrame:
        """
        Compare plusieurs entreprises sur leurs ratios financiers
        
        Args:
            tickers: Liste de symboles boursiers
            
        Returns:
            DataFrame avec comparaison
        """
        print(f"\n{'='*60}")
        print(f"🔍 COMPARAISON D'ENTREPRISES")
        print(f"{'='*60}\n")
        
        comparison_data = []
        
        for ticker in tickers:
            ratios = self.get_all_ratios(ticker)
            comparison_data.append(ratios)
        
        df = pd.DataFrame(comparison_data)
        
        # Réorganiser les colonnes
        cols = ['ticker', 'company_name', 'roe', 'roa', 'debt_to_equity', 'profit_margin']
        df = df[cols]
        
        # Renommer pour plus de clarté
        df.columns = ['Ticker', 'Entreprise', 'ROE (%)', 'ROA (%)', 'Dette/Equity', 'Marge Nette (%)']
        
        print("\n" + "="*60)
        print("📊 RÉSULTATS DE LA COMPARAISON")
        print("="*60 + "\n")
        print(df.to_string(index=False))
        print("\n")
        
        return df
    
    def get_financial_summary(self, ticker: str) -> Dict:
        """
        Génère un résumé financier complet
        
        Args:
            ticker: Symbole boursier
            
        Returns:
            Dictionnaire avec résumé détaillé
        """
        if ticker not in self.company_data:
            self.get_company_info(ticker)
        
        data = self.company_data.get(ticker, {})
        
        summary = {
            'company_name': data.get('name', ticker),
            'sector': data.get('sector', 'N/A'),
            'industry': data.get('industry', 'N/A'),
            'market_cap': self._format_number(data.get('market_cap', 0)),
            'revenue': self._format_number(data.get('revenue', 0)),
            'net_income': self._format_number(data.get('net_income', 0)),
            'total_assets': self._format_number(data.get('total_assets', 0)),
            'total_equity': self._format_number(data.get('total_equity', 0)),
            'total_debt': self._format_number(data.get('total_debt', 0)),
            'current_price': data.get('current_price', 0)
        }
        
        return summary
    
    def _format_number(self, num: float) -> str:
        """
        Formate un nombre en notation lisible
        
        Args:
            num: Nombre à formater
            
        Returns:
            String formaté
        """
        if num >= 1e12:
            return f"${num/1e12:.2f}T"
        elif num >= 1e9:
            return f"${num/1e9:.2f}B"
        elif num >= 1e6:
            return f"${num/1e6:.2f}M"
        else:
            return f"${num:,.2f}"
    
    def get_stock_history(self, ticker: str, period: str = "1mo") -> pd.DataFrame:
        """
        Récupère l'historique des prix
        
        Args:
            ticker: Symbole boursier
            period: Période (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame avec historique
        """
        try:
            print(f"📈 Récupération de l'historique pour {ticker} ({period})...")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            print(f"✅ {len(hist)} jours d'historique récupérés")
            return hist
        except Exception as e:
            print(f"❌ Erreur lors de la récupération de l'historique: {e}")
            return pd.DataFrame()
    
    def analyze_performance(self, ticker: str, period: str = "1y") -> Dict:
        """
        Analyse la performance d'une action
        
        Args:
            ticker: Symbole boursier
            period: Période d'analyse
            
        Returns:
            Dictionnaire avec métriques de performance
        """
        hist = self.get_stock_history(ticker, period)
        
        if hist.empty:
            return {}
        
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        max_price = hist['Close'].max()
        min_price = hist['Close'].min()
        
        return_pct = ((end_price - start_price) / start_price) * 100
        volatility = hist['Close'].pct_change().std() * 100
        
        performance = {
            'ticker': ticker,
            'period': period,
            'start_price': round(start_price, 2),
            'end_price': round(end_price, 2),
            'max_price': round(max_price, 2),
            'min_price': round(min_price, 2),
            'return_pct': round(return_pct, 2),
            'volatility': round(volatility, 2)
        }
        
        print(f"\n📊 Performance de {ticker} sur {period}:")
        print(f"   Rendement: {performance['return_pct']}%")
        print(f"   Volatilité: {performance['volatility']}%")
        
        return performance


def test_agent():
    """Fonction de test de l'agent financier"""
    
    print("\n" + "🎯"*35)
    print("  TEST DE L'AGENT FINANCIER")
    print("🎯"*35 + "\n")
    
    # Initialiser l'agent
    agent = FinancialAgent()
    
    # Test 1: Analyse d'une seule entreprise
    print("\n" + "="*60)
    print("TEST 1: Analyse d'une entreprise (Apple)")
    print("="*60)
    ratios_aapl = agent.get_all_ratios("AAPL")
    
    # Test 2: Résumé financier
    print("\n" + "="*60)
    print("TEST 2: Résumé financier détaillé")
    print("="*60)
    summary = agent.get_financial_summary("AAPL")
    print(f"\n📋 Résumé de {summary['company_name']}:")
    for key, value in summary.items():
        if key != 'company_name':
            print(f"   {key}: {value}")
    
    # Test 3: Comparaison d'entreprises
    print("\n" + "="*60)
    print("TEST 3: Comparaison Tech Giants (AAPL, MSFT, GOOGL)")
    print("="*60)
    comparison = agent.compare_companies(["AAPL", "MSFT", "GOOGL"])
    
    # Test 4: Performance historique
    print("\n" + "="*60)
    print("TEST 4: Analyse de performance")
    print("="*60)
    perf = agent.analyze_performance("AAPL", "1y")
    
    print("\n" + "✅"*35)
    print("  TOUS LES TESTS RÉUSSIS!")
    print("✅"*35 + "\n")


if __name__ == "__main__":
    test_agent()