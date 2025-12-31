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
    
    def __init__(self, model_name: str = "mistral", use_ollama: bool = True):  # Changé à mistral pour meilleur perf
        """
        Initialise le LLM
        
        Args:
            model_name: Nom du modèle Ollama (mistral recommandé pour finance)
            use_ollama: Utiliser Ollama si disponible
        """
        self.model_name = model_name
        self.llm = None
        self.use_ollama = use_ollama and OLLAMA_AVAILABLE
        
        if self.use_ollama:
            try:
                print(f"🤖 Initialisation du LLM: {model_name}...")
                self.llm = Ollama(model=model_name, temperature=0.3)  # Température baissée pour plus de précision
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
            context: Contexte extrait du RAG (avec métadonnées)
            
        Returns:
            Réponse générée
        """
        if self.use_ollama and self.llm:
            try:
                # Prompt amélioré
                prompt = f"""Tu es un analyste financier expert et précis. Utilise exclusivement le contexte fourni pour répondre à la question.

Règles importantes :
- Si le contexte contient des chiffres de revenus, croissance, segments, cite-les clairement avec la source (nom du fichier et page).
- Si plusieurs documents mentionnent des valeurs différentes, priorise le rapport le plus récent ou pertinent (ex: Q3/Q4 earnings release).
- Si l'information exacte n'est pas présente, dis "Non mentionné explicitement dans les documents fournis" sans spéculer.
- Sois concis, professionnel et chiffré. Structure ta réponse : Résumé + Détails + Sources.

Contexte (plusieurs documents) :
{context[:6000]}  # Augmenté à 6000 pour mistral

Question : {question}

Réponse :"""
                
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
        
        # Extraction basique améliorée en fallback
        metrics = self.extract_key_metrics(context)
        response = "Réponse fallback basée sur le contexte :\n"
        for key, value in metrics.items():
            response += f"- {key.capitalize()}: {value}\n"
        return response
    
    def summarize_document(self, text: str, max_length: int = 200) -> str:
        """
        Résume un document financier
        
        Args:
            text: Texte à résumer
            max_length: Longueur maximale
            
        Returns:
            Résumé
        """
        if self.use_ollama and self.llm:
            try:
                prompt = f"Résume ce texte financier en français en {max_length} mots maximum, en mettant l'accent sur les métriques clés (revenus, croissance, bénéfices) :\n\n{text[:3000]}"  # Augmenté pour mistral
                return self.llm.invoke(prompt)
            except:
                pass
        
        # Fallback: prendre les premières phrases
        sentences = text.split('.')[:5]
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
        
        # Patterns améliorés pour finances
        patterns = {
            'revenue': r'(?:revenue|total revenue|revenus?|chiffre d\'affaires)[\s:]+(?:\$|€|\£)?[\d,.]+ ?(?:billion|million|thousand|B|M|K)?',
            'profit': r'(?:profit|bénéfice|net income|net profit)[\s:]+(?:\$|€|\£)?[\d,.]+ ?(?:billion|million|thousand|B|M|K)?',
            'growth': r'(?:growth|croissance|year-over-year|YoY)[\s:]+[\d.+-]+%?',
            'margin': r'(?:margin|marge)[\s:]+[\d.+-]+%?',
            'segments': r'(?:revenue by segment|sources of revenue|revenus par segment)[\s:]+.*'  # Basic for segments
        }
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics[key] = ", ".join(matches[:3])  # Prends les 3 premiers matchs
        
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
    llm = FinancialLLM(model_name="mistral")
    
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