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


class QARAGProcessor:
    """
    Processor for Q&A data in Greek for RAG systems.
    Supports JSON and JSONL formats.
    Creates chunks focused on questions, answers, and combined Q&A.
    Builds indices for keywords, questions, answers, and sources.
    """

    def __init__(self,
                 embedding_model_name: str = "intfloat/multilingual-e5-large",
                 chunk_size: int = 400,  # Smaller for Q&A
                 chunk_overlap: int = 50,
                 tfidf_max_features: int = 3000):
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Text splitter for Q&A
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )

        # Indices
        self.keyword_index = defaultdict(list)
        self.question_index = defaultdict(list)
        self.answer_index = defaultdict(list)
        self.source_index = defaultdict(list)

        # Q&A specific keywords
        self.qa_keywords = self._get_qa_keywords()
        self.question_patterns = self._get_question_patterns()

        # Greek stopwords
        self.greek_stopwords = self._greek_stopwords()
        
        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=(1, 2),
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
        self.qa_data: List[Dict] = []

    def _get_qa_keywords(self) -> Dict[str, List[str]]:
        """Keywords for Q&A in Greek"""
        return {
            'ερωτήσεις': ['τι', 'πότε', 'πώς', 'γιατί', 'που', 'ποιος', 'ποια', 'ποιο', 'πόσο', 'πόσα'],
            'εγκυμοσύνη': ['έγκυος', 'εγκύου', 'εγκυμοσύνη', 'μητρότητα', 'κύηση', 'τοκετός', 'γέννα'],
            'άδεια': ['άδεια', 'άδειας', 'αναρρωτική', 'κανονική άδεια', 'άδεια μητρότητας', 'γονική άδεια'],
            'απόλυση': ['απόλυση', 'καταγγελία', 'τερματισμός', 'λύση σύμβασης', 'απολύω', 'απολύει'],
            'αποζημίωση': ['αποζημίωση', 'αποζημιώσεις', 'χρηματική παροχή', 'αποζημίωση απόλυσης'],
            'μισθός': ['μισθός', 'αμοιβή', 'πληρωμή', 'αποδοχές', 'καθυστέρηση μισθού'],
            'ωράριο': ['ωράριο', 'υπερωρίες', 'βάρδια', 'νυχτερινή εργασία', 'ώρες εργασίας'],
            'ασφάλιση': ['ασφάλιση', 'ΕΦΚΑ', 'εισφορές', 'κοινωνική ασφάλιση', 'ασφαλιστικές εισφορές'],
            'σύμβαση': ['σύμβαση', 'εργασιακή σύμβαση', 'συλλογική σύμβαση', 'ατομική σύμβαση'],
            'δικαιώματα': ['δικαιώματα', 'εργασιακά δικαιώματα', 'υποχρεώσεις', 'νόμος'],
            'διακρίσεις': ['διακρίσεις', 'ρατσισμός', 'σεξισμός', 'διακρίσεις φύλου', 'παρενόχληση'],
            'υπερωρία': ['υπερωρία', 'υπερωριακή εργασία', 'υπερωρίες', 'προαιρετική υπερωρία'],
            'παραίτηση': ['παραίτηση', 'παραίτηση εργαζομένου', 'παραίτηση από εργασία'],
            'ρεπό': ['ρεπό', 'ημέρα ανάπαυσης', 'εργασία Κυριακής'],
            'δώρα': ['δώρο Χριστουγέννων', 'δώρο Πάσχα', '13ος μισθός', '14ος μισθός'],
            'φορείς': ['ΟΑΕΔ', 'ΕΦΚΑ', 'ΣΕΠΕ', 'Υπουργείο Εργασίας', 'ΓΣΕΕ', 'ΑΔΕΔΥ', 'ΟΜΕΔ'],
            'απεργία': ['απεργία', 'απεργιακές κινητοποιήσεις', 'συνδικαλισμός'],
            'δοκιμαστική': ['δοκιμαστική περίοδος', 'περίοδος δοκιμής', 'probation'],
            'μερική': ['μερική απασχόληση', 'part-time', 'εκ περιτροπής εργασία']
        }

    def _get_question_patterns(self) -> List[str]:
        """Usual question patterns in Greek"""
        return [
            r'\bτι ισχύει\b', r'\bπώς γίνεται\b', r'\bπότε\b', r'\bγιατί\b',
            r'\bμπορώ να\b', r'\bέχω δικαίωμα\b', r'\bτι πρέπει\b',
            r'\bπώς υπολογίζεται\b', r'\bποια είναι\b', r'\bποιες είναι\b',
            r'\bπόσο\b', r'\bπόσα\b', r'\bσε ποιες περιπτώσεις\b'
        ]

    def _greek_stopwords(self) -> List[str]:
        return ['και', 'ο', 'η', 'το', 'σε', 'με', 'για', 'που', 'των', 'να', 'τι', 'στο', 'στη', 'στην']

    def normalize_text(self, text: str) -> str:
        """normalization of text"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()

    def extract_qa_keywords(self, text: str, is_question: bool = False) -> Dict[str, Any]:
        """Extraction of Q&A keywords"""
        normalized_text = self.normalize_text(text)
        found_keywords = set()
        question_patterns_found = []
        
        # Keyword matching
        for category, keyword_list in self.qa_keywords.items():
            for keyword in keyword_list:
                if keyword in normalized_text:
                    found_keywords.add(keyword)
                    found_keywords.add(category)

        # Question pattern matching
        if is_question:
            for pattern in self.question_patterns:
                if re.search(pattern, normalized_text):
                    question_patterns_found.append(pattern)

        # Stemming-based matching
        words = re.findall(r"\b[\w']+\b", normalized_text)
        if self.stemmer:
            for word in words:
                try:
                    stem = self.stemmer.stem(word)
                    for cat, kwlist in self.qa_keywords.items():
                        for kw in kwlist:
                            if self.stemmer.stem(kw) == stem:
                                found_keywords.add(kw)
                                found_keywords.add(cat)
                except Exception:
                    continue

        return {
            'keywords': found_keywords,
            'question_patterns': question_patterns_found,
            'keyword_count': len(found_keywords),
            'is_question': is_question
        }

    def create_qa_chunks(self, qa_item: Dict) -> List[Dict]:
        """Creation of Q&A chunks"""
        chunks = []
        
        # New format: "input" & "output"
        question = qa_item.get("input", "").strip()
        answer = qa_item.get("output", "").strip()
        source = qa_item.get("source", "").strip()  # May be empty

        if not question or not answer:
            return chunks

        # Chunk 1: Question-focused (Input-focused)
        q_keywords = self.extract_qa_keywords(question, is_question=True)
        question_chunk = {
            'text': f"Ερώτηση: {question}",
            'type': 'question',
            'original_question': question,
            'original_answer': answer,
            'source': source,
            'keywords': q_keywords,
            'combined_text': f"{question} {answer}"
        }
        chunks.append(question_chunk)

        # Chunk 2: Answer-focused (Output-focused) only if long enough
        if len(answer) > 200:
            a_keywords = self.extract_qa_keywords(answer, is_question=False)
            answer_chunk = {
                'text': f"Απάντηση: {answer}",
                'type': 'answer',
                'original_question': question,
                'original_answer': answer,
                'source': source,
                'keywords': a_keywords,
                'combined_text': f"{question} {answer}"
            }
            chunks.append(answer_chunk)

        # Chunk 3: Combined Q&A (Input + Output)
        combined_text = f"Ερώτηση: {question}\n\nΑπάντηση: {answer}"
        combined_keywords = self.extract_qa_keywords(combined_text, is_question=False)
        combined_chunk = {
            'text': combined_text,
            'type': 'qa_combined',
            'original_question': question,
            'original_answer': answer,
            'source': source,
            'keywords': combined_keywords,
            'combined_text': combined_text
        }
        chunks.append(combined_chunk)

        return chunks

    def process_qa_jsonl(self, file_path: str) -> List[Document]:
        """Editing of Q&A JSONL file (new method for JSONL format)"""
        print(f"Edititng Q&A JSONL file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist!")
            return []

        qa_data = []
        try:
            # Read JSONL file line by line
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            qa_item = json.loads(line)
                            qa_data.append(qa_item)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Σφάλμα JSON στη γραμμή {line_num}: {e}")
                            continue
        except Exception as e:
            print(f"Error loading JSONL: {e}")
            return []

        if not qa_data:
            print("Dint find any Q&A items in JSONL file")
            return []

        docs = []
        self.all_texts = []
        self.qa_data = qa_data

        print(f"Editing {len(qa_data)} Q&A items...")

        for i, qa_item in enumerate(qa_data):
            try:
                # Check required fields
                if "input" not in qa_item or "output" not in qa_item:
                    print(f"⚠️ Q&A item {i} δεν έχει 'input' ή 'output' πεδία")
                    continue

                chunks = self.create_qa_chunks(qa_item)
                
                for chunk_data in chunks:
                    if len(chunk_data['text'].strip()) < 20:
                        continue

                    # Metadata
                    metadata = {
                        "qa_index": i,
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_type": chunk_data['type'],
                        "source": chunk_data['source'],
                        "original_question": chunk_data['original_question'],
                        "original_answer": chunk_data['original_answer'],
                        "keywords": list(chunk_data['keywords']['keywords']),
                        "keyword_count": chunk_data['keywords']['keyword_count'],
                        "is_question": chunk_data['keywords']['is_question'],
                        "has_important_content": chunk_data['keywords']['keyword_count'] > 0
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
                    
                    # Question/Answer specific indices
                    if chunk_data['type'] == 'question':
                        q_words = set(re.findall(r"\b\w+\b", chunk_data['original_question'].lower()))
                        for word in q_words:
                            if len(word) > 2:
                                self.question_index[word].append(doc_index)
                    
                    if chunk_data['type'] in ['answer', 'qa_combined']:
                        a_words = set(re.findall(r"\b\w+\b", chunk_data['original_answer'].lower()))
                        for word in a_words:
                            if len(word) > 2:
                                self.answer_index[word].append(doc_index)

                    # Source index (if available)
                    if chunk_data['source']:
                        self.source_index[chunk_data['source']].append(doc_index)

            except Exception as e:
                print(f"Error in Q&A item {i}: {e}")
                continue

        self.docs = docs

        # Train TF-IDF
        if self.all_texts:
            try:
                print("Training TF-IDF vectorizer...")
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_texts)
                print("TF-IDF training completed")
            except Exception as e:
                print(f"Error TF-IDF: {e}")
                self.tfidf_matrix = None

        print(f"Created {len(docs)} Q&A chunks")
        return docs

    def process_qa_json(self, file_path: str) -> List[Document]:
        """Editing of Q&A JSON file (old method for JSON format)"""
        print(f"Edititng Q&A file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist!")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return []

        if not isinstance(qa_data, list):
            print("JSON format Error: Expected a list of Q&A items")
            return []

        docs = []
        self.all_texts = []
        self.qa_data = qa_data

        print(f" Edit {len(qa_data)} Q&A items...")

        for i, qa_item in enumerate(qa_data):
            try:
                # Transition support for old format
                if "question" in qa_item and "answer" in qa_item:
                    qa_item_converted = {
                        "input": qa_item["question"],
                        "output": qa_item["answer"],
                        "source": qa_item.get("source", "")
                    }
                else:
                    qa_item_converted = qa_item

                chunks = self.create_qa_chunks(qa_item_converted)
                
                for chunk_data in chunks:
                    if len(chunk_data['text'].strip()) < 20:
                        continue

                    # Metadata
                    metadata = {
                        "qa_index": i,
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_type": chunk_data['type'],
                        "source": chunk_data['source'],
                        "original_question": chunk_data['original_question'],
                        "original_answer": chunk_data['original_answer'],
                        "keywords": list(chunk_data['keywords']['keywords']),
                        "keyword_count": chunk_data['keywords']['keyword_count'],
                        "is_question": chunk_data['keywords']['is_question'],
                        "has_important_content": chunk_data['keywords']['keyword_count'] > 0
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
                    
                    # Question/Answer specific indices
                    if chunk_data['type'] == 'question':
                        q_words = set(re.findall(r"\b\w+\b", chunk_data['original_question'].lower()))
                        for word in q_words:
                            if len(word) > 2:
                                self.question_index[word].append(doc_index)
                    
                    if chunk_data['type'] in ['answer', 'qa_combined']:
                        a_words = set(re.findall(r"\b\w+\b", chunk_data['original_answer'].lower()))
                        for word in a_words:
                            if len(word) > 2:
                                self.answer_index[word].append(doc_index)

                    # Source index
                    if chunk_data['source']:
                        self.source_index[chunk_data['source']].append(doc_index)

            except Exception as e:
                print(f" Error in Q&A item {i}: {e}")
                continue

        self.docs = docs

        # Train TF-IDF
        if self.all_texts:
            try:
                print(" Train TF-IDF vectorizer...")
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_texts)
                print(" TF-IDF training completed")
            except Exception as e:
                print(f" Error TF-IDF: {e}")
                self.tfidf_matrix = None

        print(f" Created {len(docs)} Q&A chunks")
        return docs

    def create_qa_vector_store(self, docs: List[Document], save_path: str = "./faiss-files/faiss_qa_hybrid"):
        """Creation of FAISS vector store for Q&A"""
        if not docs:
            print("There are no documents to index.")
            return None

        print(f" Creating Q&A embeddings...")
        try:
            embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)

            # Check if exists
            if os.path.exists(save_path):
                try:
                    db = FAISS.load_local(save_path, embedding_model)
                    print(f" Loaded existing FAISS index from '{save_path}'")
                    return db
                except Exception:
                    print("Will create new FAISS index...")

            db = FAISS.from_documents(docs, embedding_model)
            db.save_local(save_path)

            # Save documents
            docs_path = f"{save_path}_docs.pkl"
            with open(docs_path, 'wb') as f:
                pickle.dump(self.docs, f)

            # Save indices
            indices_data = {
                'keyword_index': dict(self.keyword_index),
                'question_index': dict(self.question_index),
                'answer_index': dict(self.answer_index),
                'source_index': dict(self.source_index),
                'qa_keywords': self.qa_keywords,
                'qa_data': self.qa_data
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

            print(f" Q&A FAISS index stored: '{save_path}'")
            return db

        except Exception as e:
            print(f" Error creating Q&A vector store: {e}")
            return None

    def hybrid_qa_search(self, db: FAISS, query: str, top_k: int = 5, weights: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
        """Hybrid search dor Q&A data (embeddings + tfidf + keyword)"""
        query_norm = self.normalize_text(query)
        print(f" Q&A Hybrid search for: '{query}'")

        all_candidates = {}
        
        # 1) Keyword matching
        keyword_matches = set()
        query_words = re.findall(r"\b[\w']+\b", query_norm)
        
        for word in query_words:
            if word in self.keyword_index:
                keyword_matches.update(self.keyword_index[word])
            
            # Question-specific matching
            if word in self.question_index:
                keyword_matches.update(self.question_index[word])
            
            # Answer-specific matching
            if word in self.answer_index:
                keyword_matches.update(self.answer_index[word])

            # Category matching
            for category, keyword_list in self.qa_keywords.items():
                if word in keyword_list and category in self.keyword_index:
                    keyword_matches.update(self.keyword_index[category])

        for idx in keyword_matches:
            if idx < len(self.docs):
                all_candidates[idx] = {
                    'doc': self.docs[idx],
                    'keyword_score': 1.0,
                    'tfidf_score': 0.0,
                    'embedding_score': 0.0
                }

        print(f" Keyword matches: {len(keyword_matches)}")

        # 2) TF-IDF search
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
                                'embedding_score': 0.0
                            }
                        else:
                            all_candidates[idx]['tfidf_score'] = score

                print(f" TF-IDF matches: {len([s for s in similarities if s > 0])}")
                
            except Exception as e:
                print(f" TF-IDF search error: {e}")

        # 3) Embedding search
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
                            'embedding_score': normalized_score
                        }
                    else:
                        all_candidates[doc_idx]['embedding_score'] = normalized_score

            print(f" Embedding matches: {len(embedding_results)}")
            
        except Exception as e:
            print(f" Embedding search error: {e}")

        # 4) Combine scores
        final_scores = []
        for idx, data in all_candidates.items():
            combined_score = (
                weights[0] * data['embedding_score'] +
                weights[1] * data['tfidf_score'] +
                weights[2] * data['keyword_score']
            )
            
            final_scores.append((
                data['doc'],
                combined_score,
                {
                    'embedding': data['embedding_score'],
                    'tfidf': data['tfidf_score'],
                    'keyword': data['keyword_score']
                }
            ))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        print(f" Final Q&A results: {len(final_scores[:top_k])}")
        
        return final_scores[:top_k]

    def load_qa_search_system(self, base_path: str = "./faiss-files/faiss_qa_hybrid"):
        """Load Q&A search system"""
        print(f" Load Q&A system from {base_path}...")
        
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
                    self.question_index = defaultdict(list, indices_data['question_index'])
                    self.answer_index = defaultdict(list, indices_data['answer_index'])
                    self.source_index = defaultdict(list, indices_data['source_index'])
                    self.qa_data = indices_data.get('qa_data', [])
            
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
            
            print(" Q&A system loaded successfully.")
            return db
            
        except Exception as e:
            print(f" Q&A Loading error: {e}")
            return None

    def analyze_qa_coverage(self):
        """Debugging & analysis of Q&A coverage"""
        print("\n Q&A Coverage Analysis:")
        print(f" Total Q&A Documents: {len(self.docs)}")
        print(f" Initial Q&A items: {len(self.qa_data)}")

        # Chunk types
        chunk_types = Counter([doc.metadata.get('chunk_type', 'unknown') for doc in self.docs])
        print(f"\n Chunks Types:")
        for chunk_type, count in chunk_types.items():
            print(f"   {chunk_type}: {count}")

        # Keywords coverage
        print(f"\n Keywords:")
        for category, keywords in self.qa_keywords.items():
            total_chunks = sum(len(self.keyword_index.get(keyword, [])) for keyword in keywords)
            if total_chunks > 0:
                print(f"   {category}: {total_chunks} chunks")

        # Important content
        important_chunks = sum(1 for doc in self.docs if doc.metadata.get('has_important_content', False))
        percentage = (important_chunks/len(self.docs)*100) if len(self.docs) > 0 else 0
        print(f"\nChunks with important content: {important_chunks}/{len(self.docs)} ({percentage:.1f}%)")

    def detect_file_format(self, file_path: str) -> str:
        """Detect if file is JSON or JSONL"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('['):
                    return 'json'
                elif first_line.startswith('{'):
                    return 'jsonl'
                else:
                    # Try to parse as JSON first
                    f.seek(0)
                    try:
                        json.load(f)
                        return 'json'
                    except:
                        return 'jsonl'
        except Exception:
            return 'unknown'

    def process_qa_file(self, file_path: str) -> List[Document]:
        """Automatic processing of Q&A file based on its format"""
        file_format = self.detect_file_format(file_path)
        
        print(f"Found file: {file_format}")
        
        if file_format == 'jsonl':
            return self.process_qa_jsonl(file_path)
        elif file_format == 'json':
            return self.process_qa_json(file_path)
        else:
            print(f"Unknowned source of file: {file_path}")
            return []


def main():
    print(" Q&A RAG Processor (Updated for JSONL Format)")
    print("="*60)

    processor = QARAGProcessor(
        embedding_model_name="intfloat/multilingual-e5-large",
        chunk_size=400,
        chunk_overlap=50
    )

    try:
        # Adjust the path to your JSONL file here
        file_path = "./datasets/qa_all_for_query.jsonl"  # Change to your JSONL file path
        
        # If the new file doesn't exist, try the old one for compatibility
        if not os.path.exists(file_path):
            file_path = "./datasets/train_data_all_for_query.json"
            print(f"Try with old format: {file_path}")
        
        docs = processor.process_qa_file(file_path)
        
        if not docs:
            print("Didn't process any documents.")
            return

        processor.analyze_qa_coverage()

        # Create vector store
        db = processor.create_qa_vector_store(docs, "./faiss-files/faiss_qa_hybrid")
        
        if db:
            print("\n Successfully created or loaded FAISS index.")
            
            # Test search with some example queries
            test_queries = [
                "Τι ισχύει για την άδεια μητρότητας;",
                "Μπορεί ο εργοδότης να αλλάξει τη σύμβαση;",
                "Πώς υπολογίζονται τα δώρα;",
                "Τι γίνεται με την αποζημίωση απόλυσης;"
            ]
            
            for query in test_queries:
                print(f"\n Test search: '{query}'")
                results = processor.hybrid_qa_search(db, query, top_k=3)
                
                for i, (doc, score, details) in enumerate(results):
                    print(f"{i+1}. Combined Score: {score:.3f}")
                    print(f"   - Embedding: {details['embedding']:.3f}")
                    print(f"   - TF-IDF:    {details['tfidf']:.3f}")
                    print(f"   - Keyword:   {details['keyword']:.3f}")
                    print(f"   Type: {doc.metadata.get('chunk_type', 'unknown')}")
                    print(f"   Content: {doc.page_content[:100]}...")
                    print()

        else:
            print("Failed to create or load FAISS index.")

    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()