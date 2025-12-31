import os
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class FinancialRAG:
    """
    Pipeline RAG pour l'analyse de documents financiers
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 1500,  # Augmenté pour capturer tableaux
                 chunk_overlap: int = 300):  # Augmenté pour continuité
        """
        Initialise le pipeline RAG
        
        Args:
            embedding_model: Modèle d'embeddings à utiliser
            chunk_size: Taille des chunks de texte
            chunk_overlap: Chevauchement entre chunks
        """
        print("🚀 Initialisation du pipeline RAG...")
        
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Vector store
        self.vectorstore = None
        
        print("✅ Pipeline RAG initialisé avec succès")
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Charge un fichier PDF
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Liste de documents
        """
        print(f"📄 Chargement du PDF : {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"✅ {len(documents)} pages chargées")
        return documents
    
    def load_multiple_pdfs(self, pdf_folder: str) -> List[Document]:
        """
        Charge plusieurs PDFs depuis un dossier
        
        Args:
            pdf_folder: Chemin vers le dossier contenant les PDFs
            
        Returns:
            Liste de tous les documents
        """
        print(f"📁 Chargement des PDFs depuis : {pdf_folder}")
        
        all_documents = []
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            print("⚠️ Aucun fichier PDF trouvé")
            return all_documents
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            try:
                docs = self.load_pdf(pdf_path)
                # Ajouter métadonnées
                for doc in docs:
                    doc.metadata['source_file'] = pdf_file
                all_documents.extend(docs)
            except Exception as e:
                print(f"❌ Erreur lors du chargement de {pdf_file}: {e}")
        
        print(f"✅ Total: {len(all_documents)} pages chargées depuis {len(pdf_files)} fichiers")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Découpe les documents en chunks
        
        Args:
            documents: Liste de documents à découper
            
        Returns:
            Liste de chunks
        """
        print(f"✂️ Découpage de {len(documents)} documents...")
        
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"✅ {len(chunks)} chunks créés")
        return chunks
    
    def create_vectorstore(self, chunks: List[Document]) -> FAISS:
        """
        Crée un vectorstore FAISS à partir des chunks
        
        Args:
            chunks: Liste de chunks à indexer
            
        Returns:
            Vectorstore FAISS
        """
        print(f"🔢 Création du vectorstore avec {len(chunks)} chunks...")
        
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        print("✅ Vectorstore créé avec succès")
        return self.vectorstore
    
    def save_vectorstore(self, save_path: str = "vectorstore"):
        """
        Sauvegarde le vectorstore sur disque
        
        Args:
            save_path: Chemin de sauvegarde
        """
        if self.vectorstore is None:
            raise ValueError("Aucun vectorstore à sauvegarder")
        
        print(f"💾 Sauvegarde du vectorstore dans : {save_path}")
        self.vectorstore.save_local(save_path)
        print("✅ Vectorstore sauvegardé")
    
    def load_vectorstore(self, load_path: str = "vectorstore"):
        """
        Charge un vectorstore depuis le disque
        
        Args:
            load_path: Chemin de chargement
        """
        print(f"📥 Chargement du vectorstore depuis : {load_path}")
        
        self.vectorstore = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print("✅ Vectorstore chargé")
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Recherche les documents les plus pertinents
        
        Args:
            query: Question de l'utilisateur
            k: Nombre de documents à retourner
            
        Returns:
            Liste des documents pertinents
        """
        if self.vectorstore is None:
            raise ValueError("Aucun vectorstore disponible. Créez-en un d'abord.")
        
        print(f"🔍 Recherche pour : '{query}'")
        
        results = self.vectorstore.similarity_search(query, k=k)
        
        print(f"✅ {len(results)} résultats trouvés")
        return results
    
    def search_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """
        Recherche avec scores de similarité
        
        Args:
            query: Question de l'utilisateur
            k: Nombre de documents à retourner
            
        Returns:
            Liste de tuples (document, score)
        """
        if self.vectorstore is None:
            raise ValueError("Aucun vectorstore disponible")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        return results
    
    def build_index(self, pdf_source: str, save: bool = True, save_path: str = "vectorstore"):
        """
        Pipeline complet : charge, découpe et indexe les documents
        
        Args:
            pdf_source: Chemin vers un PDF ou un dossier de PDFs
            save: Sauvegarder le vectorstore
            save_path: Chemin de sauvegarde
        """
        print("\n" + "="*50)
        print("🏗️ CONSTRUCTION DE L'INDEX RAG")
        print("="*50 + "\n")
        
        # Charger les documents
        if os.path.isfile(pdf_source):
            documents = self.load_pdf(pdf_source)
        elif os.path.isdir(pdf_source):
            documents = self.load_multiple_pdfs(pdf_source)
        else:
            raise ValueError(f"Source invalide : {pdf_source}")
        
        if not documents:
            raise ValueError("Aucun document chargé")
        
        # Découper
        chunks = self.split_documents(documents)
        
        # Créer vectorstore
        self.create_vectorstore(chunks)
        
        # Sauvegarder
        if save:
            self.save_vectorstore(save_path)
        
        print("\n" + "="*50)
        print("✅ INDEX RAG CONSTRUIT AVEC SUCCÈS")
        print("="*50 + "\n")


# Fonction utilitaire pour tester
def test_rag():
    """
    Fonction de test du pipeline RAG
    """
    # Initialiser
    rag = FinancialRAG()
    
    # Construire l'index (depuis un dossier de PDFs)
    base_dir = Path(__file__).resolve().parents[1]
    rag.build_index(str(base_dir / "data"), save=True, save_path=str(base_dir / "vectorstore"))
    
    # Test de recherche
    query = "Quel est le chiffre d'affaires de l'entreprise?"
    results = rag.search(query, k=3)
    
    print("\n🔍 Résultats de recherche:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Résultat {i} ---")
        print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Contenu: {doc.page_content[:200]}...")


if __name__ == "__main__":
    test_rag()