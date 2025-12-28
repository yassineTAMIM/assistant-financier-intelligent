"""
Module d'intégration LLM avec Ollama
Génère des réponses intelligentes basées sur le contexte RAG
"""

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama non installé. Utilisation du mode fallback.")


class FinancialLLM:
    """
    Gestionnaire LLM pour générer des réponses intelligentes
    """
    
    def __init__(self, model_name: str = "phi3:mini", use_ollama: bool = True):
        """
        Initialise le LLM
        
        Args:
            model_name: Nom du modèle Ollama (llama2, mistral, phi3:mini)
            use_ollama: Utiliser Ollama si disponible
        """
        self.model_name = model_name
        self.llm = None
        self.use_ollama = use_ollama and OLLAMA_AVAILABLE
        
        if self.use_ollama:
            try:
                print(f"🤖 Initialisation du LLM: {model_name}...")
                self.llm = Ollama(model=model_name, temperature=0.7)
                print(f"✅ LLM {model_name} prêt")
            except Exception as e:
                print(f"⚠️  Échec de connexion à Ollama: {e}")
                print("📝 Mode fallback activé")
                self.use_ollama = False
    
    def generate_response(self, question: str, context: str = "") -> str:
        """
        Génère une réponse basée sur le contexte
        
        Args:
            question: Question de l'utilisateur
            context: Contexte extrait du RAG
            
        Returns:
            Réponse générée
        """
        if self.use_ollama and self.llm:
            try:
                # Créer le prompt
                prompt = f"""Tu es un assistant financier expert. Utilise le contexte suivant pour répondre à la question de manière claire et précise.

Contexte:
{context[:2000]}

Question: {question}

Réponse (sois concis et professionnel):"""
                
                response = self.llm.invoke(prompt)
                return response
                
            except Exception as e:
                print(f"⚠️  Erreur LLM: {e}")
                return self._fallback_response(question, context)
        else:
            return self._fallback_response(question, context)
    
    def _fallback_response(self, question: str, context: str) -> str:
        """
        Réponse simple sans LLM (fallback)
        
        Args:
            question: Question
            context: Contexte
            
        Returns:
            Réponse basique
        """
        if not context:
            return "Je n'ai pas trouvé d'informations pertinentes dans les documents pour répondre à cette question."
        
        # Extraction simple de chiffres et mots-clés
        response = "📊 **Informations trouvées dans les documents:**\n\n"
        
        # Prendre les premières lignes pertinentes
        lines = context.split('\n')[:5]
        for line in lines:
            if line.strip():
                response += f"• {line.strip()}\n"
        
        response += "\n💡 *Pour une analyse plus approfondie, utilisez les onglets d'analyse.*"
        
        return response
    
    def summarize_document(self, text: str, max_length: int = 200) -> str:
        """
        Résume un texte
        
        Args:
            text: Texte à résumer
            max_length: Longueur maximale
            
        Returns:
            Résumé
        """
        if self.use_ollama and self.llm:
            try:
                prompt = f"Résume ce texte financier en français en {max_length} mots maximum:\n\n{text[:1500]}"
                return self.llm.invoke(prompt)
            except:
                pass
        
        # Fallback: prendre les premières phrases
        sentences = text.split('.')[:3]
        return '. '.join(sentences) + '.'
    
    def extract_key_metrics(self, text: str) -> dict:
        """
        Extrait les métriques clés d'un texte
        
        Args:
            text: Texte contenant des métriques
            
        Returns:
            Dictionnaire de métriques
        """
        import re
        
        metrics = {}
        
        # Patterns courants pour les métriques financières
        patterns = {
            'revenue': r'(?:revenue|chiffre d\'affaires|revenus?)[\s:]+(?:\$|€)?[\d,.]+ ?(?:billion|million|B|M)?',
            'profit': r'(?:profit|bénéfice|net income)[\s:]+(?:\$|€)?[\d,.]+ ?(?:billion|million|B|M)?',
            'growth': r'(?:growth|croissance)[\s:]+[\d.]+%?',
            'margin': r'(?:margin|marge)[\s:]+[\d.]+%?'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metrics[key] = match.group(0)
        
        return metrics
    
    def is_available(self) -> bool:
        """Vérifie si le LLM est disponible"""
        return self.use_ollama


def test_llm():
    """Test du module LLM"""
    print("\n" + "="*60)
    print("TEST DU MODULE LLM")
    print("="*60 + "\n")
    
    # Test 1: Initialisation
    llm = FinancialLLM(model_name="phi3:mini")
    
    if llm.is_available():
        print("✅ LLM Ollama disponible")
    else:
        print("⚠️  Mode fallback actif (sans Ollama)")
    
    # Test 2: Génération de réponse
    question = "Quel est le chiffre d'affaires?"
    context = """
    La société a réalisé un chiffre d'affaires de $100 milliards au Q3 2025,
    en hausse de 15% par rapport à l'année précédente. Le bénéfice net s'élève
    à $25 milliards avec une marge nette de 25%.
    """
    
    print("\n📝 Test de génération de réponse...")
    print(f"Question: {question}")
    response = llm.generate_response(question, context)
    print(f"\nRéponse:\n{response}")
    
    # Test 3: Extraction de métriques
    print("\n\n📊 Test d'extraction de métriques...")
    metrics = llm.extract_key_metrics(context)
    print("Métriques extraites:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")
    
    print("\n✅ Tests terminés")


if __name__ == "__main__":
    test_llm()