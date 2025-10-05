import os
import json
import pickle
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, asdict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class SearchResult:
    """result from a single search"""
    content: str
    source: str
    source_type: str  # 'qa', 'notes', 'laws'
    score: float
    raw_score: float    
    weight_applied: float
    metadata: Dict[str, Any]
    search_method: str  # 'semantic', 'keyword', 'tfidf'
    rank: int = 0
    question: str = ""
    answer: str = ""

@dataclass
class QueryConfig:
    """settings for query"""
    qa_weight: float = 0.6
    notes_weight: float = 0.35
    laws_weight: float = 0.05
    max_results_per_source: int = 20
    final_max_results: int = 40
    semantic_k: int = 20
    # Enhanced keyword search parameters
    tfidf_threshold: float = 0.1
    keyword_boost: float = 2.0
    phrase_boost: float = 3.0

class MultiSourceRAGQuery:
    """
    Multi-source RAG Query System
    """
    
    def __init__(self, 
                 qa_faiss_path: str = "./faiss-files/faiss_qa_hybrid",
                 notes_faiss_path: str = "./faiss-files/faiss_notes_hybrid", 
                 laws_faiss_path: str = "./faiss-files/faiss_laws_hybrid",
                 embedding_model: str = "intfloat/multilingual-e5-large"):
        """
        Initialization Multi-source RAG Query System
        
        Args:
            qa_faiss_path:  path Q&A FAISS files
            notes_faiss_path: path Notes FAISS files  
            laws_faiss_path: path Laws FAISS files
            embedding_model: Embedding model name
        """
        self.qa_faiss_path = qa_faiss_path
        self.notes_faiss_path = notes_faiss_path
        self.laws_faiss_path = laws_faiss_path
        self.embedding_model_name = embedding_model
        
        print(" Multi-Source RAG Query System")
        print(f" Q&A: {qa_faiss_path}")
        print(f" Notes: {notes_faiss_path}")
        print(f" Laws: {laws_faiss_path}")
        
        # Αρχικοποίηση embedding model
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name=self.embedding_model_name
        )
        
        # Φόρτωση όλων των πηγών
        self.sources = {}
        self._load_all_sources()
        
        # Query expansion patterns
        self.query_expansion_patterns = {
            'εγκυμοσύνη': ['άδεια μητρότητας', 'προστασία εγκύων', 'μητρικές παροχές'],
            'απόλυση': ['καταγγελία σύμβασης', 'τερματισμός εργασίας', 'αποζημίωση απόλυσης'],
            'άδεια': ['κανονική άδεια', 'αναρρωτική άδεια', 'ειδική άδεια'],
            'μισθός': ['αμοιβή εργασίας', 'βασικός μισθός', 'πληρωμή'],
            'ωράριο': ['ώρες εργασίας', 'υπερωρίες', 'διάρκεια εργασίας'],
            'ασφάλιση': ['κοινωνική ασφάλιση', 'ασφαλιστικές εισφορές', 'παροχές ασφάλισης'],
            'σύμβαση': ['εργασιακή σύμβαση', 'σύμβαση εργασίας', 'συμφωνία εργασίας'],
            'παρενόχληση': ['σεξουαλική παρενόχληση', 'ψυχολογική παρενόχληση', 'εργασιακή βία'],
            'δικαιώματα': ['εργασιακά δικαιώματα', 'δικαιώματα εργαζομένων', 'προστασία δικαιωμάτων'],
            'συνθήκες': ['εργασιακές συνθήκες', 'υγιεινή και ασφάλεια', 'περιβάλλον εργασίας'],
            'δώρο': ['επίδομα εορτών', 'δώρο πάσχα', 'δώρο χριστούγεννα']
        }
    
    def _load_all_sources(self):
        """load all sources"""
        source_configs = [
            ("qa", self.qa_faiss_path),
            ("notes", self.notes_faiss_path), 
            ("laws", self.laws_faiss_path)
        ]
        
        for source_type, faiss_path in source_configs:
            print(f"\n Load {source_type} from {faiss_path}...")
            try:
                source_data = self._load_single_source(source_type, faiss_path)
                if source_data:
                    self.sources[source_type] = source_data
                    print(f" {source_type}: {len(source_data['docs'])} documents")
                else:
                    print(f" {source_type}: Dind't load source data")
            except Exception as e:
                print(f" {source_type}: Error loading data - {e}")
    
    def _load_single_source(self, source_type: str, faiss_path: str) -> Optional[Dict[str, Any]]:
        """Load single source from FAISS files"""
        try:
            # load FAISS database
            db = FAISS.load_local(
                faiss_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            # load documents
            docs = []
            docs_path = f"{faiss_path}_docs.pkl"
            if os.path.exists(docs_path):
                try:
                    with open(docs_path, 'rb') as f:
                        docs = pickle.load(f)
                    print(f"    Loaded  {len(docs)} docs from pickle")
                except Exception as e:
                    print(f"    Error loading pickle docs: {e}")
                    docs = self._extract_documents_from_db(db)
            else:
                docs = self._extract_documents_from_db(db)
            
            # load indices
            indices_path = f"{faiss_path}_indices.json"
            indices = {}
            if os.path.exists(indices_path):
                with open(indices_path, 'r', encoding='utf-8') as f:
                    indices = json.load(f)
            
            # load TF-IDF
            tfidf_path = f"{faiss_path}_tfidf.pkl"
            tfidf_vectorizer = None
            if os.path.exists(tfidf_path):
                try:
                    with open(tfidf_path, 'rb') as f:
                        tfidf_data = pickle.load(f)
                        if isinstance(tfidf_data, dict):
                            tfidf_vectorizer = tfidf_data.get('vectorizer')
                        else:
                            tfidf_vectorizer = tfidf_data
                except Exception as e:
                    print(f"    TF-IDF error: {e}")
            
            return {
                'db': db,
                'docs': docs,
                'indices': indices,
                'tfidf_vectorizer': tfidf_vectorizer,
                'keyword_index': indices.get('keyword_index', {}),
                'phrase_index': indices.get('phrase_index', {}),
                'context_index': indices.get('context_index', {}),
                'important_keywords': indices.get('important_keywords', {}),
                'important_phrases': indices.get('important_phrases', []),
                'question_index': indices.get('question_index', {}),
                'answer_index': indices.get('answer_index', {}),
                'qa_data': indices.get('qa_data', [])
            }
            
        except Exception as e:
            print(f"   Error: {e}")
            return None
    
    def _extract_documents_from_db(self, db) -> List[Any]:
        """extract documents from FAISS db if pickle not available"""
        docs = []
        try:
            # Method 1: Similarity search with generic queries
            generic_queries = ["εργασία", "μισθός", "άδεια", "σύμβαση", "δικαιώματα"]
            seen_contents = set()
            
            for query in generic_queries:
                try:
                    search_docs = db.similarity_search(query, k=50)
                    for doc in search_docs:
                        content_hash = hash(doc.page_content[:100])
                        if content_hash not in seen_contents:
                            docs.append(doc)
                            seen_contents.add(content_hash)
                except Exception as e:
                    continue
                    
            # Method 2: Direct extraction from docstore if available
            if len(docs) < 100 and hasattr(db, 'docstore') and hasattr(db.docstore, '_dict'):
                for doc_id, doc in db.docstore._dict.items():
                    if hasattr(doc, 'page_content'):
                        content_hash = hash(doc.page_content[:100])
                        if content_hash not in seen_contents:
                            docs.append(doc)
                            seen_contents.add(content_hash)
                            
        except Exception as e:
            print(f"   Document extraction Error: {e}")
            
        return docs
    
    def normalize_query(self, query: str) -> str:
        """Normalize query"""
        query = query.lower()
        query = re.sub(r'\s+', ' ', query)
        query = re.sub(r'\s+([.,!?;:])', r'\1', query)
        return query.strip()
    
    def expand_query(self, query: str) -> List[str]:
        """Query expansion"""
        expanded_queries = [query]
        normalized_query = self.normalize_query(query)
        
        # Pattern-based expansion
        for pattern, expansions in self.query_expansion_patterns.items():
            if pattern in normalized_query:
                for expansion in expansions:
                    expanded_queries.append(f"{query} {expansion}")
        
        # Synonym replacement
        synonym_replacements = {
            'μισθός': ['αμοιβή', 'αποδοχές'],
            'απόλυση': ['καταγγελία', 'τερματισμός'],
            'άδεια': ['απουσία', 'αναρρωτική'],
            'εγκύων': ['μητέρας', 'εγκυμοσύνη'],
        }
        
        for original, synonyms in synonym_replacements.items():
            if original in normalized_query:
                for synonym in synonyms:
                    expanded_query = normalized_query.replace(original, synonym)
                    expanded_queries.append(expanded_query)
        
        return list(set(expanded_queries))
    
    def extract_query_features(self, query: str) -> Dict[str, Any]:
        """Εξαγωγή features από query"""
        normalized_query = self.normalize_query(query)
        expanded_queries = self.expand_query(query)
        
        # Tokenization
        words = re.findall(r'\b\w+\b', normalized_query)
        important_words = set()
        for word in words:
            if len(word) > 3 and word not in {'είναι', 'έχει', 'μπορώ', 'πρέπει', 'όταν', 'πώς', 'γιατί'}:
                important_words.add(word)
        
        return {
            'original': query,
            'normalized': normalized_query,
            'expanded_queries': expanded_queries,
            'words': words,
            'important_words': important_words
        }
    
    def search_single_source(self, source_type: str, query_features: Dict[str, Any], 
                           max_results: int = 10) -> List[SearchResult]:
        """Serch in a single source"""
        if source_type not in self.sources:
            return []
        
        source = self.sources[source_type]
        results = []
        
        # 1. Semantic Search
        semantic_results = self._semantic_search(source, query_features, max_results)
        results.extend(semantic_results)
        
        # 2. Enhanced Keyword Search per source type
        keyword_results = self._enhanced_keyword_search(source, query_features, max_results, source_type)
        results.extend(keyword_results)
        
        # 3. TF-IDF Search
        tfidf_results = self._tfidf_search(source, query_features, max_results)
        results.extend(tfidf_results)
        
        # Combine και deduplicate
        combined_results = self._combine_and_deduplicate_results(results, source_type, max_results)
        
        return combined_results
    
    def _semantic_search(self, source: Dict[str, Any], query_features: Dict[str, Any], 
                        max_results: int) -> List[SearchResult]:
        """Semantic search in a source"""
        results = []
        
        try:
            db = source['db']
            
            for i, expanded_query in enumerate(query_features['expanded_queries'][:3]):
                try:
                    search_docs = db.similarity_search_with_score(expanded_query, k=max_results)
                    
                    for doc, score in search_docs:
                        # Create SearchResult
                        result = self._create_search_result(
                            doc, float(score), float(score), "semantic", source
                        )
                        results.append(result)
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"   Semantic search Error: {e}")
            
        return results
    
    def _enhanced_keyword_search(self, source: Dict[str, Any], query_features: Dict[str, Any], 
                               max_results: int, source_type: str) -> List[SearchResult]:
        """Enhanced keyword search with source-specific logic"""
        results = []
        
        try:
            docs = source['docs']
            doc_scores = defaultdict(float)
            
            # Generic keyword searches
            keyword_index = source['keyword_index']
            phrase_index = source.get('phrase_index', {})
            context_index = source.get('context_index', {})
            
            # Phrase matching (highest score)
            for expanded_query in query_features['expanded_queries']:
                normalized = self.normalize_query(expanded_query)
                for phrase, doc_indices in phrase_index.items():
                    if phrase in normalized:
                        for doc_idx in doc_indices:
                            if 0 <= doc_idx < len(docs):
                                doc_scores[doc_idx] += 3.0
            
            # Keyword matching
            for word in query_features['important_words']:
                for keyword, doc_indices in keyword_index.items():
                    if word in keyword or keyword in word:
                        for doc_idx in doc_indices:
                            if 0 <= doc_idx < len(docs):
                                doc_scores[doc_idx] += 2.0
            
            # Source-specific logic
            if source_type == 'qa':
                self._qa_specific_search(docs, query_features, doc_scores, source)
            elif source_type == 'notes':
                self._notes_specific_search(docs, query_features, doc_scores)
            elif source_type == 'laws':
                self._laws_specific_search(docs, query_features, doc_scores)
            
            # Context matching
            for word in query_features['important_words']:
                if word in context_index:
                    for doc_idx in context_index[word]:
                        if 0 <= doc_idx < len(docs):
                            doc_scores[doc_idx] += 1.0
            
            # Get top results
            sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            for doc_idx, score in sorted_scores[:max_results]:
                if 0 <= doc_idx < len(docs) and score > 0:
                    doc = docs[doc_idx]
                    result = self._create_search_result(
                        doc, float(score), float(score), "keyword", source
                    )
                    results.append(result)
                    
        except Exception as e:
            print(f"   Enhanced keyword search Error: {e}")
            
        return results
    
    def _qa_specific_search(self, docs: List[Any], query_features: Dict[str, Any], doc_scores: Dict[int, float], source: Dict[str, Any]):
        """Specific search for Q&A with input/output format"""
        question_index = source.get('question_index', {})
        answer_index = source.get('answer_index', {})
        
        # Search in input/output fields
        for doc_idx, doc in enumerate(docs):
            if 0 <= doc_idx < len(docs):
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                
                # Check input (question)
                input_text = metadata.get('input', '').lower()
                output_text = metadata.get('output', '').lower()
                
                # Score for match in input (query) and output (answer)
                for word in query_features['important_words']:
                    if word in input_text:
                        doc_scores[doc_idx] += 4.0  #  Good score for question match
                    if word in output_text:
                        doc_scores[doc_idx] += 2.5  # Good score for answer match
                
                # Bonus for long queries with multiple important words
                query_normalized = query_features['normalized']
                if len(query_normalized) > 10:  # Long query
                    query_words = set(query_features['important_words'])
                    input_words = set(re.findall(r'\b\w+\b', input_text))
                    
                    # Jaccard similarity
                    if query_words and input_words:
                        intersection = query_words.intersection(input_words)
                        union = query_words.union(input_words)
                        jaccard = len(intersection) / len(union) if union else 0
                        
                        if jaccard > 0.3:  # Threshold
                            doc_scores[doc_idx] += 5.0 * jaccard
        
        # Index-based boosting
        for word in query_features['important_words']:
            if word in question_index:
                for doc_idx in question_index[word]:
                    if 0 <= doc_idx < len(docs):
                        doc_scores[doc_idx] += 3.0
            
            if word in answer_index:
                for doc_idx in answer_index[word]:
                    if 0 <= doc_idx < len(docs):
                        doc_scores[doc_idx] += 2.0
    
    def _notes_specific_search(self, docs: List[Any], query_features: Dict[str, Any], doc_scores: Dict[int, float]):
        """Specific search for Notes with class/topic/content format"""
        
        for doc_idx, doc in enumerate(docs):
            if 0 <= doc_idx < len(docs):
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                
                # Check fields: class, topic, content
                class_text = metadata.get('class', '').lower()
                topic_text = metadata.get('topic', '').lower()
                content_text = metadata.get('content', '').lower()
                
                for word in query_features['important_words']:
                    # Highest score for match in class
                    if word in class_text:
                        doc_scores[doc_idx] += 3.5
                    
                    # High score for match in topic
                    if word in topic_text:
                        doc_scores[doc_idx] += 3.0
                    
                    # Normal score for match in content
                    if word in content_text:
                        doc_scores[doc_idx] += 1.5
                
                # Bonus for full phrase match in topic or class
                for expanded_query in query_features['expanded_queries']:
                    normalized_query = self.normalize_query(expanded_query)
                    
                    if normalized_query in topic_text:
                        doc_scores[doc_idx] += 4.0
                    if normalized_query in class_text:
                        doc_scores[doc_idx] += 4.5
    
    def _laws_specific_search(self, docs: List[Any], query_features: Dict[str, Any], doc_scores: Dict[int, float]):
        """Specific search for Laws with volume/chapter/subject/text format"""
        
        for doc_idx, doc in enumerate(docs):
            if 0 <= doc_idx < len(docs):
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                
                # Check fields: volume, chapter, subject, text
                volume_text = metadata.get('volume', '').lower()
                chapter_text = metadata.get('chapter', '').lower()
                subject_text = metadata.get('subject', '').lower()
                text_content = metadata.get('text', '').lower()
                
                for word in query_features['important_words']:
                    # Highest score for match in subject
                    if word in subject_text:
                        doc_scores[doc_idx] += 4.0
                    
                    # High score for match in chapter or volume
                    if word in chapter_text:
                        doc_scores[doc_idx] += 2.5
                    if word in volume_text:
                        doc_scores[doc_idx] += 2.0
                    
                    # Normal score for match in text
                    if word in text_content:
                        doc_scores[doc_idx] += 1.5
                
                # Bonus for full phrase match in subject
                for expanded_query in query_features['expanded_queries']:
                    normalized_query = self.normalize_query(expanded_query)
                    
                    if normalized_query in subject_text:
                        doc_scores[doc_idx] += 5.0
    
    def _tfidf_search(self, source: Dict[str, Any], query_features: Dict[str, Any], max_results: int) -> List[SearchResult]:
        """TF-IDF search in a source"""
        results = []
        
        try:
            tfidf_vectorizer = source['tfidf_vectorizer']
            docs = source['docs']
            
            if not tfidf_vectorizer or len(docs) == 0:
                return results
            
            # Prepare query
            combined_query = " ".join(query_features['expanded_queries'])
            query_vector = tfidf_vectorizer.transform([combined_query])
            
            # Prepare document texts
            doc_texts = []
            for doc in docs:
                if hasattr(doc, 'page_content'):
                    doc_texts.append(doc.page_content)
                else:
                    doc_texts.append(str(doc))
            
            if len(doc_texts) == 0:
                return results
            
            # Compute similarities
            doc_vectors = tfidf_vectorizer.transform(doc_texts)
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:max_results]
            
            for idx in top_indices:
                if similarities[idx] > 0.1 and idx < len(docs):  # Threshold
                    doc = docs[idx]
                    result = self._create_search_result(
                        doc, float(similarities[idx]), float(similarities[idx]), "tfidf", source
                    )
                    results.append(result)
                    
        except Exception as e:
            print(f"   TF-IDF search Error: {e}")
            
        return results
    
    def _create_search_result(self, doc: Any, score: float, raw_score: float, search_method: str, source: Dict[str, Any]) -> SearchResult:
        """Create SearchResult from doc"""
        
        # Extract content and metadata
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        doc_source = metadata.get('source', 'unknown')
        
        # Extract question and answer if available
        question = ""
        answer = ""
        
        # New format with input/output
        if 'input' in metadata or 'output' in metadata:
            question = metadata.get('input', '')      # input -> question
            answer = metadata.get('output', '')       # output -> answer
        
        # Fallback for old formats
        elif 'original_question' in metadata or 'original_answer' in metadata:
            question = metadata.get('original_question', '')
            answer = metadata.get('original_answer', '')
        elif 'question' in metadata or 'answer' in metadata:
            question = metadata.get('question', '')
            answer = metadata.get('answer', '')
        
        return SearchResult(
            content=content,
            source=doc_source,
            source_type="unknown",
            score=score,
            raw_score=raw_score,
            weight_applied=1.0,
            metadata=metadata,
            search_method=search_method,
            rank=0,
            question=question,
            answer=answer
        )
    
    def _combine_and_deduplicate_results(self, results: List[SearchResult], source_type: str, max_results: int) -> List[SearchResult]:
        """Combine and deduplicate results"""
        # Set source type
        for result in results:
            result.source_type = source_type
        
        # Deduplicate based on content hash
        unique_results = {}
        for result in results:
            content_hash = hash(result.content[:500])  # Use first 500 chars for hash
            
            if content_hash in unique_results:
                # Keep the one with higher score
                if result.score > unique_results[content_hash].score:
                    unique_results[content_hash] = result
            else:
                unique_results[content_hash] = result
        
        # Sort by score
        sorted_results = sorted(unique_results.values(), key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(sorted_results[:max_results]):
            result.rank = i + 1
        
        return sorted_results[:max_results]
    
    def multi_source_query(self, query: str, config: QueryConfig = None) -> Dict[str, Any]:
        """
        Multi-source query
        
        Args:
            query: Question string
            config: Query settings (weights, max results, etc.)
            
        Returns:
            Dictionary with results and statistics
        """
        if config is None:
            config = QueryConfig()
        
        print(f" Multi-source query: '{query}'")
        print(f" Weights: QA={config.qa_weight}, Notes={config.notes_weight}, Laws={config.laws_weight}")
        
        # Extract query features
        query_features = self.extract_query_features(query)
        print(f" Key words: {list(query_features['important_words'])[:5]}")
        
        # Search each source
        all_results = {}
        weights = {
            'qa': config.qa_weight,
            'notes': config.notes_weight,
            'laws': config.laws_weight
        }
        
        for source_type in ['qa', 'notes', 'laws']:
            if source_type in self.sources:
                print(f" Searching {source_type}...")
                source_results = self.search_single_source(
                    source_type, 
                    query_features, 
                    config.max_results_per_source
                )
                
                # Apply weights
                weight = weights[source_type]
                for result in source_results:
                    result.weight_applied = weight
                    result.score = result.raw_score * weight
                
                all_results[source_type] = source_results
                print(f"    {source_type}: {len(source_results)} results")
            else:
                print(f"    {source_type}: Not available")
                all_results[source_type] = []
        
        # Combine all results
        combined_results = []
        for source_type, source_results in all_results.items():
            combined_results.extend(source_results)
        
        # Sort by weighted score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Final ranking
        final_results = combined_results[:config.final_max_results]
        for i, result in enumerate(final_results):
            result.rank = i + 1
        
        # Create summary statistics
        stats = self._generate_statistics(all_results, final_results, query_features)
        
        print(f" Final results: {len(final_results)}")
        
        return {
            'query': query,
            'query_features': query_features,
            'config': asdict(config),
            'results_by_source': {k: [asdict(r) for r in v] for k, v in all_results.items()},
            'final_results': [asdict(r) for r in final_results],
            'statistics': stats,
            'success': True
        }
    
    def _generate_statistics(self, all_results: Dict[str, List[SearchResult]], final_results: List[SearchResult],query_features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistics from results"""
        stats = {
            'total_sources_searched': len([k for k, v in all_results.items() if len(v) > 0]),
            'results_per_source': {k: len(v) for k, v in all_results.items()},
            'final_results_count': len(final_results),
            'source_distribution': {},
            'score_distribution': {},
            'search_method_distribution': {},
            'query_expansion_count': len(query_features['expanded_queries']),
            'important_words_found': len(query_features['important_words'])
        }
        
        if final_results:
            # Source distribution
            source_counts = defaultdict(int)
            for result in final_results:
                source_counts[result.source_type] += 1
            stats['source_distribution'] = dict(source_counts)
            
            # Score statistics
            scores = [r.score for r in final_results]
            stats['score_distribution'] = {
                'min': min(scores),
                'max': max(scores),
                'mean': np.mean(scores),
                'median': np.median(scores)
            }
            
            # Search method distribution
            method_counts = defaultdict(int)
            for result in final_results:
                method_counts[result.search_method] += 1
            stats['search_method_distribution'] = dict(method_counts)
            
            # Top score by source
            top_by_source = {}
            for result in final_results:
                source = result.source_type
                if source not in top_by_source or result.score > top_by_source[source]:
                    top_by_source[source] = result.score
            stats['top_score_by_source'] = top_by_source
        
        return stats
    
    def print_enhanced_results_summary(self, query_result: Dict[str, Any], show_details: bool = True):
        """Print enhanced results summary"""
        print("\n" + "="*80)
        print("ENHANCED MULTI-SOURCE RAG QUERY RESULTS")
        print("="*80)
        
        query = query_result['query']
        stats = query_result['statistics']
        final_results = query_result['final_results']
        config = query_result['config']
        
        print(f" Query: {query}")
        print(f" Weights: QA={config['qa_weight']}, Notes={config['notes_weight']}, Laws={config['laws_weight']}")
        print(f" Total Results: {len(final_results)}")
        print(f" Sources Searched: {stats['total_sources_searched']}")
        
        # Results per source
        print(f"\n Results per source:")
        for source, count in stats['results_per_source'].items():
            print(f"   • {source.upper()}: {count}")
        
        # Source distribution in final results
        if stats['source_distribution']:
            print(f"\n Final results distribution:")
            for source, count in stats['source_distribution'].items():
                percentage = (count / len(final_results)) * 100
                print(f"   • {source.upper()}: {count} ({percentage:.1f}%)")
        
        # Score statistics
        if stats['score_distribution']:
            score_stats = stats['score_distribution']
            print(f"\n Score statistics:")
            print(f"   • Min: {score_stats['min']:.3f}")
            print(f"   • Max: {score_stats['max']:.3f}")
            print(f"   • Mean: {score_stats['mean']:.3f}")
            print(f"   • Median: {score_stats['median']:.3f}")
        
        # Search method distribution
        if stats['search_method_distribution']:
            print(f"\n Search methods:")
            for method, count in stats['search_method_distribution'].items():
                percentage = (count / len(final_results)) * 100
                print(f"   • {method}: {count} ({percentage:.1f}%)")
        
        # Query expansion and important words
        if show_details and final_results:
            print(f"\n🏆 TOP {min(5, len(final_results))} RESULTS:")
            print("-" * 80)
            
            for i, result in enumerate(final_results[:5]):
                print(f"\n[{i+1}] {result['source_type'].upper()} - Score: {result['score']:.3f}")
                print(f"    Method: {result['search_method']} | Weight: {result['weight_applied']}")
                print(f"    Source: {result['source']}")
                
                # Q&A results (JSONL format)
                if result['source_type'] == 'qa':
                    if result.get('question'):
                        question_preview = result['question'][:200] + "..." if len(result['question']) > 200 else result['question']
                        print(f"     Question: {question_preview}")
                    if result.get('answer'):
                        answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
                        print(f"     Answer: {answer_preview}")
                
                # Notes results
                elif result['source_type'] == 'notes':
                    metadata = result.get('metadata', {})
                    if metadata.get('class'):
                        print(f"     Class: {metadata['class']}")
                    if metadata.get('topic'):
                        print(f"     Topic: {metadata['topic']}")
                    if metadata.get('content'):
                        content = metadata['content'][:200] + "..." if len(metadata.get('content', '')) > 200 else metadata.get('content', '')
                        print(f"     Content: {content}")
                
                # Laws results
                elif result['source_type'] == 'laws':
                    metadata = result.get('metadata', {})
                    if metadata.get('subject'):
                        print(f"     Subject: {metadata['subject']}")
                    if metadata.get('volume'):
                        print(f"     Volume: {metadata['volume']}")
                    if metadata.get('chapter'):
                        print(f"     Chapter: {metadata['chapter']}")
                    if metadata.get('text'):
                        text = metadata['text'][:200] + "..." if len(metadata.get('text', '')) > 200 else metadata['text']
                        print(f"     Text: {text}")
                
                # Generic content fallback
                else:
                    content = result['content']
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"     Content: {content}")
        
        print("\n" + "="*80)

def main():
    """Demo function"""
    print(" Enhanced Multi-Source RAG Query System")
    
    # Initialize system
    try:
        query_system = MultiSourceRAGQuery(
            qa_faiss_path="./faiss-files/faiss_qa_hybrid",
            notes_faiss_path="./faiss-files/faiss_notes_hybrid",
            laws_faiss_path="./faiss-files/faiss_laws_hybrid"
        )
    except Exception as e:
        print(f" Initialization error: {e}")
        return
    
    # Interactive query loop
    print("\n Commands:")
    print("  'exit'")
    print("  'config' - change weights")
    print("  'weights X Y Z' - set weights (π.χ. 'weights 0.6 0.3 0.1')")
    print("  'details' - toggle detailed results")
    print("  'test' - test query")
    print("  'help' - show commands")
    print("-" * 80)
    
    # Default configuration - can be changed interactively
    config = QueryConfig(qa_weight=0.6, notes_weight=0.35, laws_weight=0.05)
    show_details = True
    
    while True:
        try:
            user_input = input("\n Query: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'έξοδος']:
                print(" Goodbye!")
                break
                
            elif user_input.lower() == 'help':
                print("\n💡 Available commands:")
                print("  'exit' - exit the program")
                print("  'config' - show current configuration")
                print("  'weights X Y Z' - set weights for QA, Notes, Laws (e.g. 'weights 0.6 0.3 0.1')")
                print("  'details' - toggle detailed results on/off")
                print("  'test' - run a test query about Easter Bonus")
                continue
                
            elif user_input.lower() == 'config':
                print(f"\n⚙️ Current Configuration:")
                print(f"   • QA Weight: {config.qa_weight}")
                print(f"   • Notes Weight: {config.notes_weight}")
                print(f"   • Laws Weight: {config.laws_weight}")
                print(f"   • Max results per source: {config.max_results_per_source}")
                print(f"   • Final max results: {config.final_max_results}")
                print(f"   • Show details: {show_details}")
                continue
                
            elif user_input.lower().startswith('weights'):
                try:
                    parts = user_input.split()
                    if len(parts) == 4:
                        qa_w, notes_w, laws_w = map(float, parts[1:4])
                        
                        # Validation
                        total = qa_w + notes_w + laws_w
                        if abs(total - 1.0) > 0.01:
                            print(f" Weights sum to {total:.3f}, not 1.0. Normalizing...")
                            qa_w, notes_w, laws_w = qa_w/total, notes_w/total, laws_w/total
                        
                        config.qa_weight = qa_w
                        config.notes_weight = notes_w
                        config.laws_weight = laws_w
                        
                        print(f" New weights: QA={qa_w:.3f}, Notes={notes_w:.3f}, Laws={laws_w:.3f}")
                    else:
                        print(" Usage: weights <qa> <notes> <laws> (e.g. 'weights 0.6 0.3 0.1')")
                except ValueError:
                    print(" Invalid weights format. Use numbers between 0 and 1.")
                continue
                
            elif user_input.lower() == 'details':
                show_details = not show_details
                status = "enabled" if show_details else "disabled"
                print(f" Detailed results {status}")
                continue
                
            elif user_input.lower() == 'test':
                user_input = "Πότε πρέπει να καταβληθεί το Δώρο Πάσχα;"
                print(f" Test query: {user_input}")
            
            # Execute query
            print(f"\n Processing query...")
            result = query_system.multi_source_query(user_input, config)
            
            if result['success']:
                query_system.print_enhanced_results_summary(result, show_details=show_details)
                
                # Επιπλέον αξιολόγηση για το test query
                if "δώρο πάσχα" in user_input.lower():
                    final_results = result['final_results']
                    if final_results:
                        top_result = final_results[0]
                        print(f"\n EVALUATION FOR EASTER BONUS QUERY:")
                        
                        # Έλεγχος αν βρήκε τη σωστή απάντηση
                        expected_phrases = ["μεγάλη τετάρτη", "τετάρτη", "πάσχα"]
                        found_answer = False
                        
                        if top_result['source_type'] == 'qa' and top_result.get('answer'):
                            answer = top_result['answer'].lower()
                            for phrase in expected_phrases:
                                if phrase in answer:
                                    print(f"    CORRECT: Found '{phrase}' in QA answer")
                                    found_answer = True
                                    break
                        
                        if not found_answer:
                            print(f"    Expected answer about 'Μεγάλη Τετάρτη' not found in top result")
                            
                        print(f"    Top result from: {top_result['source_type'].upper()}")
                        print(f"    Score: {top_result['score']:.3f}")
            else:
                print(" Query failed")
                
        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    main()
