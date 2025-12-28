# 🤖 Assistant Financier Intelligent

Un assistant conversationnel intelligent capable d'analyser des documents financiers en utilisant RAG (Retrieval-Augmented Generation) et des agents autonomes.

## 📋 Vue d'ensemble du projet

Ce projet combine :
- **RAG (Retrieval-Augmented Generation)** : Pour rechercher dans des documents financiers réels
- **LLM (Large Language Model)** : Pour générer des réponses intelligentes
- **Agent Financier** : Pour effectuer des calculs et analyses automatiques
- **Interface Streamlit** : Pour une interaction utilisateur fluide

## 🏗️ Architecture

```
assistant-financier-intelligent/
├── data/                          # Dossier pour vos PDFs financiers
│   ├── 2025-Earnings-Release-Final.pdf
│   ├── 3Q25-Slides-WPRT-FINAL.pdf
│   └── ...
├── vectorstore/                   # Index vectoriel (généré automatiquement)
├── rag.py                         # Pipeline RAG
├── rag_test.py                    # Tests du RAG
├── agent.py                       # Agent financier
├── app.py                         # Interface Streamlit
├── requirements.txt               # Dépendances Python
└── README.md                      # Ce fichier
```

## 🚀 Installation

### 1. Prérequis
- Python 3.8+
- pip

### 2. Cloner le repository
```bash
git clone <votre-repo>
cd assistant-financier-intelligent
```

### 3. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 4. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 📦 Préparation des données

### 1. Placer vos documents
Copiez vos PDFs financiers dans le dossier `data/` :

```bash
# Créer le dossier si nécessaire
mkdir data

# Copier vos PDFs
cp /chemin/vers/vos/pdfs/*.pdf data/
```

Documents actuels dans le projet :
- `2025-Earnings-Release-Final.pdf`
- `3Q25-Slides-WPRT-FINAL.pdf`
- `2025-FY-Results-Press-Release.pdf`
- `2025q3-alphabet-earnings-release.pdf`
- `2025q3release.pdf`
- `FY2025-4th-Quarter-Earnings-Release.pdf`
- `q2-2025-earnings-release.pdf`
- `Q3-2025-Earnings-Release.pdf`
- `Q4-2025-Earnings-Release_vF.pdf`
- `Workday-Announces-Fiscal-2026-Third-Q.pdf`

### 2. Tester le pipeline RAG
```bash
python rag_test.py
```

Ce script va :
- ✅ Charger tous les PDFs
- ✅ Les découper en chunks
- ✅ Créer l'index vectoriel
- ✅ Sauvegarder le vectorstore
- ✅ Tester la recherche sémantique

## 🧪 Tests et Validation

### Test du RAG
```bash
python rag_test.py
```

**Sortie attendue :**
```
🎯🎯🎯 TEST COMPLET DU PIPELINE RAG - ASSISTANT FINANCIER 🎯🎯🎯

======================================================================
  TEST 1: Initialisation du RAG
======================================================================

🚀 Initialisation du pipeline RAG...
✅ Pipeline RAG initialisé avec succès
✅ Initialisation réussie

...

✅ Tous les tests sont passés avec succès!
🚀 Le pipeline RAG est opérationnel!
```

## 💡 Utilisation du RAG

### Exemple simple
```python
from rag import FinancialRAG

# Initialiser le RAG
rag = FinancialRAG()

# Construire l'index (première fois seulement)
rag.build_index("data/")

# Ou charger un index existant
# rag.load_vectorstore("vectorstore")

# Faire une recherche
results = rag.search("Quel est le chiffre d'affaires?", k=3)

# Afficher les résultats
for doc in results:
    print(f"Source: {doc.metadata['source_file']}")
    print(f"Contenu: {doc.page_content[:200]}...\n")
```

### Recherche avec scores
```python
# Recherche avec scores de similarité
results_with_scores = rag.search_with_scores("revenus", k=5)

for doc, score in results_with_scores:
    print(f"Score: {score:.4f}")
    print(f"Contenu: {doc.page_content[:150]}...\n")
```

## 🔧 Configuration

### Paramètres du RAG

Dans `rag.py`, vous pouvez ajuster :

```python
rag = FinancialRAG(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Modèle d'embeddings
    chunk_size=1000,          # Taille des chunks
    chunk_overlap=200         # Chevauchement entre chunks
)
```

### Modèles d'embeddings alternatifs

```python
# Plus léger et rapide
"sentence-transformers/all-MiniLM-L6-v2"

# Plus performant (mais plus lourd)
"sentence-transformers/all-mpnet-base-v2"

# Multilingue
"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

## 📊 Prochaines étapes

### ✅ Phase 1 : RAG (Complété)
- [x] Pipeline de chargement de PDFs
- [x] Découpage en chunks
- [x] Indexation vectorielle avec FAISS
- [x] Recherche sémantique
- [x] Tests complets

### ✅ Phase 2 : Agent Financier (Complété)
- [x] Calculs de ratios financiers (ROE, ROA, Dette/Equity, Marge)
- [x] Intégration avec YFinance
- [x] Comparaison entre entreprises
- [x] Génération de rapports
- [x] Analyse de performance
- [x] Historique des prix

### 🚧 Phase 3 : Interface Streamlit
- [ ] Chat interactif
- [ ] Upload de PDFs
- [ ] Visualisations (graphiques, tableaux)
- [ ] Historique des conversations

### 🚧 Phase 4 : Intégration LLM
- [ ] Connexion avec Ollama (LLaMA, Mistral, Phi-3)
- [ ] Génération de réponses contextualisées
- [ ] Pipeline RAG + LLM complet

## 🛠️ Technologies utilisées

- **LangChain** : Framework pour applications LLM
- **FAISS** : Recherche vectorielle ultra-rapide
- **Sentence Transformers** : Embeddings de qualité
- **PyPDF** : Extraction de texte des PDFs
- **Streamlit** : Interface web interactive
- **YFinance** : Données financières en temps réel

## 📝 Livrables du projet

1. **Rapport technique** : Documentation complète
2. **Présentation vidéo** : Démonstration (15-20 min)
3. **Présentation PowerPoint** : Support visuel
4. **Code source** : Repository GitHub complet

## 🤝 Contribution

Ce projet est développé dans le cadre d'un projet académique.

## 📄 Licence

Voir le fichier LICENSE pour plus de détails.

## 🆘 Support

En cas de problème :
1. Vérifiez que tous les PDFs sont dans `data/`
2. Vérifiez que toutes les dépendances sont installées
3. Relancez `python rag_test.py` pour diagnostiquer

## 🎯 Objectif final

Créer un assistant conversationnel capable de :
- 📄 Analyser des rapports financiers complexes
- 💬 Répondre à des questions en langage naturel
- 📊 Calculer automatiquement des ratios financiers
- 📈 Comparer les performances entre entreprises
- 🎨 Visualiser les données de manière interactive