"""
Script de test pour le pipeline RAG
Teste le chargement, l'indexation et la recherche de documents financiers
"""
from pathlib import Path
import sys

# Permet d'importer le package `src/` quand on exécute le fichier directement
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import os
from src.rag import FinancialRAG
from typing import List
from langchain_core.documents import Document



def print_separator(title: str = ""):
    """Affiche un séparateur visuel"""
    print("\n" + "="*70)
    if title:
        print(f"  {title}")
        print("="*70)
    print()


def test_1_initialization():
    """Test 1: Initialisation du pipeline RAG"""
    print_separator("TEST 1: Initialisation du RAG")
    
    try:
        rag = FinancialRAG(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=1500,
            chunk_overlap=300
        )
        print("✅ Initialisation réussie")
        return rag
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        return None


def test_2_load_documents(rag: FinancialRAG, data_folder: str = "data"):
    """Test 2: Chargement des documents PDF"""
    print_separator("TEST 2: Chargement des PDFs")
    
    if not os.path.exists(data_folder):
        print(f"❌ Le dossier '{data_folder}' n'existe pas")
        print(f"📁 Création du dossier '{data_folder}'...")
        os.makedirs(data_folder)
        print(f"⚠️  Veuillez placer vos PDFs dans le dossier '{data_folder}' et relancer le script")
        return None
    
    try:
        documents = rag.load_multiple_pdfs(data_folder)
        
        if not documents:
            print("⚠️  Aucun document trouvé dans le dossier")
            print("📝 Documents attendus:")
            expected_files = [
                "2025-Earnings-Release-Final.pdf",
                "3Q25-Slides-WPRT-FINAL.pdf",
                "2025-FY-Results-Press-Release.pdf",
                "2025q3-alphabet-earnings-release.pdf",
                "2025q3release.pdf",
                "FY2025-4th-Quarter-Earnings-Release.pdf",
                "q2-2025-earnings-release.pdf",
                "Q3-2025-Earnings-Release.pdf",
                "Q4-2025-Earnings-Release_vF.pdf",
                "Workday-Announces-Fiscal-2026-Third-Q.pdf"
            ]
            for f in expected_files:
                print(f"   - {f}")
            return None
        
        print(f"\n📊 Statistiques:")
        print(f"   - Total de pages: {len(documents)}")
        print(f"   - Fichiers chargés: {len(set(doc.metadata.get('source_file') for doc in documents))}")
        
        # Afficher les fichiers chargés
        print(f"\n📄 Fichiers détectés:")
        files = {}
        for doc in documents:
            filename = doc.metadata.get('source_file', 'Unknown')
            files[filename] = files.get(filename, 0) + 1
        
        for filename, page_count in sorted(files.items()):
            print(f"   - {filename}: {page_count} pages")
        
        print("\n✅ Chargement des documents réussi")
        return documents
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None


def test_3_split_documents(rag: FinancialRAG, documents: List[Document]):
    """Test 3: Découpage des documents en chunks"""
    print_separator("TEST 3: Découpage en chunks")
    
    try:
        chunks = rag.split_documents(documents)
        
        print(f"\n📊 Statistiques des chunks:")
        print(f"   - Nombre total de chunks: {len(chunks)}")
        print(f"   - Taille moyenne: ~{sum(len(c.page_content) for c in chunks) // len(chunks)} caractères")
        
        # Afficher un exemple de chunk
        if chunks:
            print(f"\n📝 Exemple de chunk (premier):")
            print(f"   Source: {chunks[0].metadata.get('source_file', 'Unknown')}")
            print(f"   Page: {chunks[0].metadata.get('page', 'Unknown')}")
            print(f"   Contenu (150 premiers caractères):")
            print(f"   {chunks[0].page_content[:150]}...")
        
        print("\n✅ Découpage réussi")
        return chunks
        
    except Exception as e:
        print(f"❌ Erreur lors du découpage: {e}")
        return None


def test_4_create_vectorstore(rag: FinancialRAG, chunks: List[Document]):
    """Test 4: Création du vectorstore FAISS"""
    print_separator("TEST 4: Création du vectorstore")
    
    try:
        vectorstore = rag.create_vectorstore(chunks)
        
        print(f"\n📊 Informations sur le vectorstore:")
        print(f"   - Nombre de vecteurs: {vectorstore.index.ntotal}")
        print(f"   - Dimension des embeddings: {vectorstore.index.d}")
        
        print("\n✅ Vectorstore créé avec succès")
        return vectorstore
        
    except Exception as e:
        print(f"❌ Erreur lors de la création du vectorstore: {e}")
        return None


def test_5_save_load_vectorstore(rag: FinancialRAG, save_path: str = "vectorstore"):
    """Test 5: Sauvegarde et chargement du vectorstore"""
    print_separator("TEST 5: Sauvegarde/Chargement")
    
    try:
        # Sauvegarder
        rag.save_vectorstore(save_path)
        print(f"✅ Vectorstore sauvegardé dans '{save_path}'")
        
        # Tester le chargement
        rag_test = FinancialRAG()
        rag_test.load_vectorstore(save_path)
        print(f"✅ Vectorstore rechargé avec succès")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde/chargement: {e}")
        return False


def test_6_search(rag: FinancialRAG):
    """Test 6: Recherche sémantique"""
    print_separator("TEST 6: Tests de recherche sémantique")
    
    # Liste de questions de test
    test_queries = [
        "Quel est le chiffre d'affaires total?",
        "Quels sont les résultats financiers du trimestre?",
        "Quelle est la croissance des revenus?",
        "Quels sont les principaux indicateurs de performance?",
        "Quelle est la rentabilité de l'entreprise?"
    ]
    
    print("🔍 Exécution de requêtes de test...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Requête {i}: {query} ---")
        
        try:
            results = rag.search(query, k=3)
            
            print(f"✅ {len(results)} résultats trouvés\n")
            
            for j, doc in enumerate(results, 1):
                print(f"  Résultat {j}:")
                print(f"    Source: {doc.metadata.get('source_file', 'Unknown')}")
                print(f"    Page: {doc.metadata.get('page', 'Unknown')}")
                print(f"    Extrait: {doc.page_content[:150].replace(chr(10), ' ')}...")
                print()
                
        except Exception as e:
            print(f"❌ Erreur lors de la recherche: {e}")
            return False
    
    print("✅ Tous les tests de recherche réussis")
    return True


def test_7_search_with_scores(rag: FinancialRAG):
    """Test 7: Recherche avec scores de similarité"""
    print_separator("TEST 7: Recherche avec scores")
    
    query = "revenus et bénéfices de l'entreprise"
    print(f"🔍 Recherche: '{query}'\n")
    
    try:
        results = rag.search_with_scores(query, k=5)
        
        print(f"📊 Résultats avec scores de similarité:\n")
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. Score: {score:.4f}")
            print(f"     Source: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"     Page: {doc.metadata.get('page', 'Unknown')}")
            print(f"     Extrait: {doc.page_content[:100].replace(chr(10), ' ')}...")
            print()
        
        print("✅ Test avec scores réussi")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def run_all_tests(data_folder: str = "data", vectorstore_path: str = "vectorstore"):
    """Exécute tous les tests du pipeline RAG"""
    
    print("\n" + "🎯"*35)
    print("  TEST COMPLET DU PIPELINE RAG - ASSISTANT FINANCIER")
    print("🎯"*35 + "\n")
    
    # Test 1: Initialisation
    rag = test_1_initialization()
    if not rag:
        return False
    
    # Test 2: Chargement
    documents = test_2_load_documents(rag, data_folder)
    if not documents:
        return False
    
    # Test 3: Découpage
    chunks = test_3_split_documents(rag, documents)
    if not chunks:
        return False
    
    # Test 4: Vectorstore
    vectorstore = test_4_create_vectorstore(rag, chunks)
    if not vectorstore:
        return False
    
    # Test 5: Sauvegarde/Chargement
    if not test_5_save_load_vectorstore(rag, vectorstore_path):
        return False
    
    # Test 6: Recherche
    if not test_6_search(rag):
        return False
    
    # Test 7: Recherche avec scores
    if not test_7_search_with_scores(rag):
        return False
    
    # Résumé final
    print_separator("RÉSUMÉ FINAL")
    print("✅ Tous les tests sont passés avec succès!")
    print(f"\n📦 Votre vectorstore est prêt à être utilisé:")
    print(f"   - Chemin: {vectorstore_path}/")
    print(f"   - Documents indexés: {len(documents)} pages")
    print(f"   - Chunks créés: {len(chunks)}")
    print(f"   - Prêt pour l'intégration avec l'agent et Streamlit!")
    print("\n" + "🎉"*35 + "\n")
    
    return True


if __name__ == "__main__":
    # Configuration
    DATA_FOLDER = "data"
    VECTORSTORE_PATH = "vectorstore"
    
    # Lancer tous les tests
    success = run_all_tests(DATA_FOLDER, VECTORSTORE_PATH)
    
    if not success:
        print("\n⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        exit(1)
    else:
        print("🚀 Le pipeline RAG est opérationnel!")
        print("📝 Prochaine étape: Développer l'agent financier (agent.py)")