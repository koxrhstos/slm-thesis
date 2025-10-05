import json
import uuid
import re
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import numpy as np
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import concurrent.futures
from nltk.stem.snowball import SnowballStemmer


class NotesRAGProcessor:
    """
    Processor for Notes data in Greek for RAG systems.
    Supports JSON format.
    Creates chunks focused on notes, definitions, and summaries.
    Builds indices for keywords, classes, topics, content, definitions, and sources.
    """

    def __init__(self,
                 embedding_model_name: str = "intfloat/multilingual-e5-large",
                 chunk_size: int = 500,  # medium size for notes
                 chunk_overlap: int = 100,
                 tfidf_max_features: int = 4000):
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Text splitter for Notes
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )

        # Indices
        self.keyword_index = defaultdict(list)
        self.class_index = defaultdict(list)
        self.topic_index = defaultdict(list)
        self.content_index = defaultdict(list)
        self.definition_index = defaultdict(list)
        self.source_index = defaultdict(list)

        # Notes specific keywords
        self.notes_keywords = self._get_notes_keywords()
        self.definition_patterns = self._get_definition_patterns()

        # Greek stopwords
        self.greek_stopwords = self._greek_stopwords()
        
        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=(1, 3),  # Bigger range for notes
            stop_words=self.greek_stopwords
        )

        # Stemmer
        try:
            self.stemmer = SnowballStemmer("greek")
        except Exception:
            self.stemmer = None

        # Data holders
        self.all_texts: List[str] = []
        self.tfidf_matrix = None
        self.docs: List[Document] = []
        self.notes_data: List[Dict] = []

    def _get_notes_keywords(self) -> Dict[str, List[str]]:
        """Keywords specific for Notes context"""
        return {
            'ορισμοί': ['ορισμός', 'ορίζεται', 'νοείται', 'θεωρείται', 'καλείται', 'σημαίνει', 'εννοείται'],
            'εργατικό_δίκαιο': ['εργατικό δίκαιο', 'εργασιακή νομοθεσία', 'εργασιακές σχέσεις', 'εργασιακό περιβάλλον'],
            'συμβάσεις': ['σύμβαση εργασίας', 'εργασιακή σύμβαση', 'συλλογική σύμβαση', 'ατομική σύμβαση'],
            'εργαζόμενος': ['εργαζόμενος', 'εργάτης', 'υπάλληλος', 'εργοδοτούμενος', 'εργατικό δυναμικό'],
            'εργοδότης': ['εργοδότης', 'εταιρεία', 'εταιρία', 'οργανισμός', 'επιχείρηση', 'κεφάλαιο'],
            'αμοιβή_ορισμοί': ['αμοιβή', 'μισθός', 'αποδοχές', 'ημερομίσθιο', 'ωρομίσθιο', 'μισθοδοσία'],
            'χρόνος_εργασίας': ['ωράριο εργασίας', 'ώρες εργασίας', 'χρόνος εργασίας', 'εργάσιμες ώρες'],
            'άδειες_ορισμοί': ['κανονική άδεια', 'ετήσια άδεια', 'αναρρωτική άδεια', 'ειδική άδεια'],
            'ασφάλιση_ορισμοί': ['κοινωνική ασφάλιση', 'ασφαλιστικές εισφορές', 'ασφαλιστικά ταμεία'],
            'απόλυση_ορισμοί': ['καταγγελία σύμβασης', 'λύση εργασιακής σχέσης', 'τερματισμός απασχόλησης'],
            'συνδικαλισμός_ορισμοί': ['συνδικαλιστική οργάνωση', 'συνδικαλιστικές ελευθερίες', 'εργατικά σωματεία'],
            'διαφορές_ορισμοί': ['εργατικές διαφορές', 'εργασιακές διαφορές', 'συλλογικές διαφορές'],
            'επιθεώρηση_ορισμοί': ['επιθεώρηση εργασίας', 'εργατική επιθεώρηση', 'ελεγκτικές αρμοδιότητες'],
            'διαιτησία_ορισμοί': ['διαιτησία', 'διαιτητικό δικαστήριο', 'μεσολάβηση', 'συμβιβασμός'],
            'παραβάσεις_ορισμοί': ['εργατική παράβαση', 'παράβαση νομοθεσίας', 'κυρώσεις', 'πρόστιμα'],
            'ασφάλεια_ορισμοί': ['υγιεινή και ασφάλεια', 'εργασιακή ασφάλεια', 'προστασία εργαζομένων'],
            'δικαιώματα_ορισμοί': ['εργασιακά δικαιώματα', 'συνταγματικά δικαιώματα', 'θεμελιώδη δικαιώματα'],
            'υποχρεώσεις_ορισμοί': ['εργασιακές υποχρεώσεις', 'συμβατικές υποχρεώσεις', 'νομικές υποχρεώσεις'],
            'διακρίσεις_ορισμοί': ['απαγόρευση διακρίσεων', 'ίση μεταχείριση', 'διακρίσεις στην εργασία'],
            'παρενόχληση_ορισμοί': ['εργασιακή παρενόχληση', 'σεξουαλική παρενόχληση', 'ηθική παρενόχληση'],
            'νομοθεσία_ορισμοί': ['νομοθετικό πλαίσιο', 'νομικές διατάξεις', 'κανονιστικό περιβάλλον'],
            'δικαστήρια_ορισμοί': ['εργατικά δικαστήρια', 'δικαστική προστασία', 'δικαστικές αποφάσεις'],
            'φορείς_ορισμοί': ['αρμόδιοι φορείς', 'θεσμικοί φορείς', 'δημόσιοι οργανισμοί']
        }

    def _get_definition_patterns(self) -> List[str]:
        """Patterns indicating definitions in Greek"""
        return [
            r'\bορίζεται ως\b', r'\bνοείται\b', r'\bεννοείται\b',
            r'\bσημαίνει\b', r'\bκαλείται\b', r'\bθεωρείται\b',
            r'\bπεριλαμβάνει\b', r'\bαποτελεί\b', r'\bείναι\b.*\bότι\b',
            r'\bχαρακτηρίζεται ως\b', r'\bκατηγοροποιείται ως\b'
        ]

    def _greek_stopwords(self) -> List[str]:
        return ['και', 'ο', 'η', 'το', 'σε', 'με', 'για', 'που', 'των', 'να', 'τι', 'στο', 'στη', 'στην']

    def normalize_text(self, text: str) -> str:
        """Normalization of text"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()

    def extract_notes_keywords(self, text: str, note_class: str = "", topic: str = "") -> Dict[str, Any]:
        """Extraction of keywords from Notes text"""
        normalized_text = self.normalize_text(text)
        found_keywords = set()
        definition_patterns_found = []
        is_definition = False
        
        # Keyword matching
        for category, keyword_list in self.notes_keywords.items():
            for keyword in keyword_list:
                if keyword in normalized_text:
                    found_keywords.add(keyword)
                    found_keywords.add(category)

        # Definition pattern matching
        for pattern in self.definition_patterns:
            if re.search(pattern, normalized_text):
                definition_patterns_found.append(pattern)
                is_definition = True

        # Class and topic as keywords
        if note_class:
            class_norm = self.normalize_text(note_class)
            found_keywords.add(class_norm)
            # Split class into words
            class_words = re.findall(r"\b[\w']+\b", class_norm)
            for word in class_words:
                if len(word) > 2:
                    found_keywords.add(word)

        if topic:
            topic_norm = self.normalize_text(topic)
            found_keywords.add(topic_norm)
            # Split topic into words
            topic_words = re.findall(r"\b[\w']+\b", topic_norm)
            for word in topic_words:
                if len(word) > 2:
                    found_keywords.add(word)

        # Stemming-based matching
        words = re.findall(r"\b[\w']+\b", normalized_text)
        if self.stemmer:
            for word in words:
                try:
                    stem = self.stemmer.stem(word)
                    for cat, kwlist in self.notes_keywords.items():
                        for kw in kwlist:
                            if self.stemmer.stem(kw) == stem:
                                found_keywords.add(kw)
                                found_keywords.add(cat)
                except Exception:
                    continue

        return {
            'keywords': found_keywords,
            'definition_patterns': definition_patterns_found,
            'keyword_count': len(found_keywords),
            'is_definition': is_definition,
            'class_keywords': class_norm if note_class else "",
            'topic_keywords': topic_norm if topic else ""
        }

    def create_notes_chunks(self, note_item: Dict) -> List[Dict]:
        """Creation of chunks from a Notes item"""
        chunks = []
        
        note_class = note_item.get("class", "").strip()
        topic = note_item.get("topic", "").strip()
        content = note_item.get("content", "").strip()
        source = note_item.get("source", "").strip()

        if not content:
            return chunks

        # Chunk 1: Full structured note
        full_text = f"Κατηγορία: {note_class}\nΘέμα: {topic}\nΠεριεχόμενο: {content}"
        full_keywords = self.extract_notes_keywords(content, note_class, topic)
        
        full_chunk = {
            'text': full_text,
            'type': 'note_full',
            'class': note_class,
            'topic': topic,
            'content': content,
            'source': source,
            'keywords': full_keywords,
            'priority_score': self._calculate_priority_score(full_keywords, content)
        }
        chunks.append(full_chunk)

        # Chunk 2: Content-only (only if long enough)
        if len(content) > 300:
            content_keywords = self.extract_notes_keywords(content, note_class, topic)
            content_chunk = {
                'text': content,
                'type': 'note_content',
                'class': note_class,
                'topic': topic,
                'content': content,
                'source': source,
                'keywords': content_keywords,
                'priority_score': self._calculate_priority_score(content_keywords, content)
            }
            chunks.append(content_chunk)

        # Chunk 3: Class-Topic summary (for better indexing)
        if note_class and topic:
            summary_text = f"{note_class}: {topic}"
            summary_keywords = self.extract_notes_keywords(summary_text, note_class, topic)
            summary_chunk = {
                'text': summary_text,
                'type': 'note_summary',
                'class': note_class,
                'topic': topic,
                'content': content[:200] + "..." if len(content) > 200 else content,
                'source': source,
                'keywords': summary_keywords,
                'priority_score': self._calculate_priority_score(summary_keywords, summary_text)
            }
            chunks.append(summary_chunk)

        return chunks

    def _calculate_priority_score(self, keywords_data: Dict, text: str) -> float:
        """ Calculate a priority score based on keywords and text features """
        score = 0.0
        
        # Bonus for being a definition
        if keywords_data['is_definition']:
            score += 0.3
        
        # Bonus for keywords count
        score += min(0.4, keywords_data['keyword_count'] * 0.05)
        
        # Bonus for text length
        text_len = len(text)
        if 100 <= text_len <= 800:
            score += 0.2
        elif text_len > 800:
            score += 0.1
        
        # Bonus for important categories
        important_categories = ['ορισμοί', 'συμβάσεις', 'δικαιώματα_ορισμοί', 'υποχρεώσεις_ορισμοί']
        for cat in important_categories:
            if cat in keywords_data['keywords']:
                score += 0.1
        
        return min(1.0, score)

    def process_notes_json(self, file_path: str) -> List[Document]:
        """Edit and process Notes JSON file"""
        print(f"Edit Notes file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                notes_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return []

        if not isinstance(notes_data, list):
            print("JSON root is not a list")
            return []

        docs = []
        self.all_texts = []
        self.notes_data = notes_data

        print(f"Editing {len(notes_data)} Notes items...")

        for i, note_item in enumerate(notes_data):
            try:
                chunks = self.create_notes_chunks(note_item)
                
                for chunk_data in chunks:
                    if len(chunk_data['text'].strip()) < 20:
                        continue

                    # Metadata
                    metadata = {
                        "note_index": i,
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_type": chunk_data['type'],
                        "class": chunk_data['class'],
                        "topic": chunk_data['topic'],
                        "source": chunk_data['source'],
                        "original_content": chunk_data['content'],
                        "keywords": list(chunk_data['keywords']['keywords']),
                        "keyword_count": chunk_data['keywords']['keyword_count'],
                        "is_definition": chunk_data['keywords']['is_definition'],
                        "priority_score": chunk_data['priority_score'],
                        "has_important_content": chunk_data['keywords']['keyword_count'] > 2
                    }

                    # Create document
                    doc = Document(
                        page_content=chunk_data['text'],
                        metadata=metadata
                    )
                    
                    doc_index = len(docs)
                    docs.append(doc)
                    self.all_texts.append(doc.page_content)

                    # Update indices
                    for keyword in chunk_data['keywords']['keywords']:
                        self.keyword_index[keyword].append(doc_index)
                    
                    # Class index
                    if chunk_data['class']:
                        class_norm = self.normalize_text(chunk_data['class'])
                        self.class_index[class_norm].append(doc_index)
                        # Add class words to keyword index
                        class_words = re.findall(r"\b[\w']+\b", class_norm)
                        for word in class_words:
                            if len(word) > 2:
                                self.class_index[word].append(doc_index)
                    
                    # Topic index
                    if chunk_data['topic']:
                        topic_norm = self.normalize_text(chunk_data['topic'])
                        self.topic_index[topic_norm].append(doc_index)
                        # Add topic words to keyword index
                        topic_words = re.findall(r"\b[\w']+\b", topic_norm)
                        for word in topic_words:
                            if len(word) > 2:
                                self.topic_index[word].append(doc_index)

                    # Content index (important words from content)
                    content_words = set(re.findall(r"\b\w{4,}\b", chunk_data['content'].lower()))
                    for word in content_words:
                        if word not in self.greek_stopwords:
                            self.content_index[word].append(doc_index)

                    # Definition index
                    if chunk_data['keywords']['is_definition']:
                        self.definition_index['is_definition'].append(doc_index)

                    # Source index
                    if chunk_data['source']:
                        self.source_index[chunk_data['source']].append(doc_index)

            except Exception as e:
                print(f"Error in Notes item {i}: {e}")
                continue

        self.docs = docs

        # Train TF-IDF
        if self.all_texts:
            try:
                print("Training TF-IDF vectorizer...")
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_texts)
                print("TF-IDF vectorizer trained.")
            except Exception as e:
                print(f"Error TF-IDF: {e}")
                self.tfidf_matrix = None

        print(f"Created {len(docs)} Notes chunks")
        return docs

    def create_notes_vector_store(self, docs: List[Document], save_path: str = "./faiss-files/faiss_notes_hybrid"):
        """Creation of Notes vector store with FAISS"""
        if not docs:
            print("There are no Notes documents to index.")
            return None

        print(f"Creating Notes embeddings...")
        try:
            embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)

            # Check if exists
            if os.path.exists(save_path):
                try:
                    db = FAISS.load_local(save_path, embedding_model)
                    print(f"Loaded exidting FAISS index from '{save_path}'")
                    return db
                except Exception:
                    print("Creating new FAISS index")

            db = FAISS.from_documents(docs, embedding_model)
            db.save_local(save_path)

            # Save documents
            docs_path = f"{save_path}_docs.pkl"
            with open(docs_path, 'wb') as f:
                pickle.dump(self.docs, f)

            # Save indices
            indices_data = {
                'keyword_index': dict(self.keyword_index),
                'class_index': dict(self.class_index),
                'topic_index': dict(self.topic_index),
                'content_index': dict(self.content_index),
                'definition_index': dict(self.definition_index),
                'source_index': dict(self.source_index),
                'notes_keywords': self.notes_keywords,
                'notes_data': self.notes_data
            }
            indices_path = f"{save_path}_indices.json"
            with open(indices_path, 'w', encoding='utf-8') as f:
                json.dump(indices_data, f, ensure_ascii=False, indent=2)

            # Save TF-IDF
            if self.tfidf_vectorizer and self.tfidf_matrix is not None:
                tfidf_path = f"{save_path}_tfidf.pkl"
                with open(tfidf_path, 'wb') as f:
                    pickle.dump({
                        'vectorizer': self.tfidf_vectorizer,
                        'matrix': self.tfidf_matrix
                    }, f)

            print(f"Notes FAISS index saved in: '{save_path}'")
            return db

        except Exception as e:
            print(f"Error creating Notes vector store: {e}")
            return None

    def hybrid_notes_search(self, db: FAISS, query: str, top_k: int = 5, weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)):
        """Hybrid search for Notes data"""
        query_norm = self.normalize_text(query)
        print(f"Notes Hybrid search for: '{query}'")

        all_candidates = {}
        
        # 1) Keyword matching
        keyword_matches = set()
        query_words = re.findall(r"\b[\w']+\b", query_norm)
        
        for word in query_words:
            # General keyword matching
            if word in self.keyword_index:
                keyword_matches.update(self.keyword_index[word])
            
            # Class-specific matching
            if word in self.class_index:
                keyword_matches.update(self.class_index[word])
            
            # Topic-specific matching
            if word in self.topic_index:
                keyword_matches.update(self.topic_index[word])
            
            # Content matching
            if word in self.content_index:
                keyword_matches.update(self.content_index[word])

            # Category matching
            for category, keyword_list in self.notes_keywords.items():
                if word in keyword_list and category in self.keyword_index:
                    keyword_matches.update(self.keyword_index[category])

        for idx in keyword_matches:
            if idx < len(self.docs):
                # Boost score για definitions
                is_definition = self.docs[idx].metadata.get('is_definition', False)
                priority_score = self.docs[idx].metadata.get('priority_score', 0.5)
                
                base_score = 1.0
                if is_definition:
                    base_score += 0.3
                base_score += priority_score * 0.2
                
                all_candidates[idx] = {
                    'doc': self.docs[idx],
                    'keyword_score': min(1.0, base_score),
                    'tfidf_score': 0.0,
                    'embedding_score': 0.0,
                    'structure_score': 0.0
                }

        print(f"Keyword matches: {len(keyword_matches)}")

        # 2) Structure-based matching (class/topic exact matches)
        structure_matches = set()
        
        # Check for exact class matches
        for class_name in self.class_index.keys():
            if class_name in query_norm:
                structure_matches.update(self.class_index[class_name])
        
        # Check for exact topic matches
        for topic_name in self.topic_index.keys():
            if topic_name in query_norm:
                structure_matches.update(self.topic_index[topic_name])

        for idx in structure_matches:
            if idx < len(self.docs):
                if idx not in all_candidates:
                    all_candidates[idx] = {
                        'doc': self.docs[idx],
                        'keyword_score': 0.0,
                        'tfidf_score': 0.0,
                        'embedding_score': 0.0,
                        'structure_score': 1.0
                    }
                else:
                    all_candidates[idx]['structure_score'] = 1.0

        print(f"Structure matches: {len(structure_matches)}")

        # 3) TF-IDF search
        if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
            try:
                q_vec = self.tfidf_vectorizer.transform([query_norm])
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
                
                top_tfidf_idx = similarities.argsort()[-top_k*2:][::-1]
                max_tfidf = similarities.max() if len(similarities) > 0 else 1.0
                
                for idx in top_tfidf_idx:
                    if idx < len(self.docs) and similarities[idx] > 0:
                        score = similarities[idx] / max_tfidf if max_tfidf > 0 else 0
                        if idx not in all_candidates:
                            all_candidates[idx] = {
                                'doc': self.docs[idx],
                                'keyword_score': 0.0,
                                'tfidf_score': score,
                                'embedding_score': 0.0,
                                'structure_score': 0.0
                            }
                        else:
                            all_candidates[idx]['tfidf_score'] = score

                print(f"TF-IDF matches: {len([s for s in similarities if s > 0])}")
                
            except Exception as e:
                print(f"TF-IDF search error: {e}")

        # 4) Embedding search
        try:
            embedding_results = db.similarity_search_with_score(query, k=top_k*2)
            
            for doc, score in embedding_results:
                doc_idx = None
                for i, stored_doc in enumerate(self.docs):
                    if stored_doc.page_content == doc.page_content:
                        doc_idx = i
                        break
                
                if doc_idx is not None:
                    normalized_score = min(1.0, max(0.0, 1.0 - score))
                    if doc_idx not in all_candidates:
                        all_candidates[doc_idx] = {
                            'doc': doc,
                            'keyword_score': 0.0,
                            'tfidf_score': 0.0,
                            'embedding_score': normalized_score,
                            'structure_score': 0.0
                        }
                    else:
                        all_candidates[doc_idx]['embedding_score'] = normalized_score

            print(f"Embedding matches: {len(embedding_results)}")
            
        except Exception as e:
            print(f"Embedding search error: {e}")

        # 5) Combine scores with 4 components
        final_scores = []
        for idx, data in all_candidates.items():
            combined_score = (
                weights[0] * data['embedding_score'] +
                weights[1] * data['tfidf_score'] +
                weights[2] * data['keyword_score'] +
                weights[3] * data['structure_score']
            )
            
            # Bonus για high-priority items
            priority_bonus = data['doc'].metadata.get('priority_score', 0.0) * 0.1
            combined_score += priority_bonus
            
            final_scores.append((data['doc'], min(1.0, combined_score)))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"🏆 Final Notes results: {len(final_scores[:top_k])}")
        
        return final_scores[:top_k]

    def load_notes_search_system(self, base_path: str = "./faiss-files/faiss_notes_hybrid"):
        """Loading Notes search system"""
        print(f"Loading Notes system from {base_path}...")
        
        try:
            embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)
            db = FAISS.load_local(base_path, embedding_model, allow_dangerous_deserialization=True)
            
            # Load documents
            docs_path = f"{base_path}_docs.pkl"
            if os.path.exists(docs_path):
                with open(docs_path, 'rb') as f:
                    self.docs = pickle.load(f)
            
            # Load indices
            indices_path = f"{base_path}_indices.json"
            if os.path.exists(indices_path):
                with open(indices_path, 'r', encoding='utf-8') as f:
                    indices_data = json.load(f)
                    self.keyword_index = defaultdict(list, indices_data['keyword_index'])
                    self.class_index = defaultdict(list, indices_data['class_index'])
                    self.topic_index = defaultdict(list, indices_data['topic_index'])
                    self.content_index = defaultdict(list, indices_data['content_index'])
                    self.definition_index = defaultdict(list, indices_data['definition_index'])
                    self.source_index = defaultdict(list, indices_data['source_index'])
                    self.notes_data = indices_data.get('notes_data', [])
            
            # Load TF-IDF
            tfidf_path = f"{base_path}_tfidf.pkl"
            if os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as f:
                    tfidf_data = pickle.load(f)
                    if isinstance(tfidf_data, dict):
                        self.tfidf_vectorizer = tfidf_data.get('vectorizer')
                        self.tfidf_matrix = tfidf_data.get('matrix')

            if not self.all_texts and self.docs:
                self.all_texts = [doc.page_content for doc in self.docs]
            
            print("Notes system loaded successfully.")
            return db
            
        except Exception as e:
            print(f"Notes Loading error: {e}")
            return None

    def analyze_notes_coverage(self):
        """Analysis of Notes data coverage"""
        print("\nNotes Analysis:")
        print(f"Total Notes Documents: {len(self.docs)}")
        print(f"Initial Notes items: {len(self.notes_data)}")

        # Chunk types
        chunk_types = Counter([doc.metadata.get('chunk_type', 'unknown') for doc in self.docs])
        print(f"\nType of Chunks:")
        for chunk_type, count in chunk_types.items():
            print(f"   {chunk_type}: {count}")

        # Classes coverage
        classes = Counter([doc.metadata.get('class', 'Άγνωστο') for doc in self.docs])
        print(f"\ncategories (Classes):")
        for class_name, count in classes.most_common(10):
            print(f"   {class_name}: {count} chunks")

        # Definitions
        definitions_count = len(self.definition_index.get('is_definition', []))
        print(f"\nDefinitions: {definitions_count} chunks")

        # Priority distribution
        priority_scores = [doc.metadata.get('priority_score', 0.0) for doc in self.docs]
        high_priority = len([s for s in priority_scores if s > 0.7])
        medium_priority = len([s for s in priority_scores if 0.3 < s <= 0.7])
        low_priority = len([s for s in priority_scores if s <= 0.3])
        
        print(f"\n Priority Distribution:")
        print(f"   High (>0.7): {high_priority}")
        print(f"   Medium (0.3-0.7): {medium_priority}")
        print(f"   Low (≤0.3): {low_priority}")

        # Keywords coverage
        print(f"\n Keywords:")
        for category, keywords in self.notes_keywords.items():
            total_chunks = sum(len(self.keyword_index.get(keyword, [])) for keyword in keywords)
            if total_chunks > 0:
                print(f"   {category}: {total_chunks} chunks")

        # Important content
        important_chunks = sum(1 for doc in self.docs if doc.metadata.get('has_important_content', False))
        percentage = (important_chunks/len(self.docs)*100) if len(self.docs) > 0 else 0
        print(f"\nChunks with important content: {important_chunks}/{len(self.docs)} ({percentage:.1f}%)")

    def search_by_class(self, class_name: str, top_k: int = 10) -> List[Document]:
        """Search by class"""
        class_norm = self.normalize_text(class_name)
        if class_norm in self.class_index:
            indices = self.class_index[class_norm][:top_k]
            return [self.docs[i] for i in indices if i < len(self.docs)]
        return []

    def search_by_topic(self, topic: str, top_k: int = 10) -> List[Document]:
        """ Search by topic"""
        topic_norm = self.normalize_text(topic)
        if topic_norm in self.topic_index:
            indices = self.topic_index[topic_norm][:top_k]
            return [self.docs[i] for i in indices if i < len(self.docs)]
        return []

    def get_definitions(self, top_k: int = 10) -> List[Document]:
        """ Get definition chunks """
        if 'is_definition' in self.definition_index:
            indices = self.definition_index['is_definition'][:top_k]
            return [self.docs[i] for i in indices if i < len(self.docs)]
        return []

    def get_high_priority_notes(self, min_priority: float = 0.7, top_k: int = 10) -> List[Document]:
        """ Get high priority notes """
        high_priority_docs = []
        for doc in self.docs:
            if doc.metadata.get('priority_score', 0.0) >= min_priority:
                high_priority_docs.append(doc)
        
        # Sort by priority score
        high_priority_docs.sort(key=lambda x: x.metadata.get('priority_score', 0.0), reverse=True)
        return high_priority_docs[:top_k]


def main():
    print("Notes RAG Processor")
    print("="*50)

    processor = NotesRAGProcessor(
        embedding_model_name="intfloat/multilingual-e5-large",
        chunk_size=500,
        chunk_overlap=100
    )

    try:
        # Process Notes JSON
        docs = processor.process_notes_json("./datasets/notes.json")  # Αλλαγή το όνομα αρχείου
        
        if not docs:
            print("Did not create any Notes documents.")
            return

        processor.analyze_notes_coverage()

        # Create vector store
        db = processor.create_notes_vector_store(docs, "./faiss-files/faiss_notes_hybrid")
        
        if db:
            print("\nSuccessfully created Notes vector store.")
            
            # Test searches
            print("\n" + "="*50)
            print("TESTING SEARCHES...")
            print("="*50)
            
            # Test 1: General hybrid search
            test_query = "Τι είναι η σύμβαση εργασίας;"
            results = processor.hybrid_notes_search(db, test_query, top_k=3)
            
            print(f"\Hybrid search results for: '{test_query}'")
            for i, (doc, score) in enumerate(results):
                print(f"{i+1}. Score: {score:.3f}")
                print(f"   Type: {doc.metadata.get('chunk_type', 'unknown')}")
                print(f"   Class: {doc.metadata.get('class', 'N/A')}")
                print(f"   Topic: {doc.metadata.get('topic', 'N/A')}")
                print(f"   Content: {doc.page_content[:150]}...")
                print()

            # Test 2: Search by class
            class_results = processor.search_by_class("εργατικό δίκαιο", top_k=2)
            print(f"\Class search results for 'εργατικό δίκαιο': {len(class_results)} results")
            for i, doc in enumerate(class_results):
                print(f"{i+1}. {doc.metadata.get('topic', 'N/A')}")

            # Test 3: Get definitions
            definitions = processor.get_definitions(top_k=3)
            print(f"\Definitions found: {len(definitions)}")
            for i, doc in enumerate(definitions):
                print(f"{i+1}. {doc.page_content[:100]}...")

            # Test 4: High priority notes
            high_priority = processor.get_high_priority_notes(min_priority=0.5, top_k=3)
            print(f"\High priority notes: {len(high_priority)}")
            for i, doc in enumerate(high_priority):
                priority = doc.metadata.get('priority_score', 0.0)
                print(f"{i+1}. Priority: {priority:.2f} - {doc.metadata.get('topic', 'N/A')}")

        else:
            print("Error creating Notes vector store.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()