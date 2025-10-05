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

class EnhancedHybridRAGProcessor:
    def __init__(self,
                 embedding_model_name: str = "intfloat/multilingual-e5-large",
                 chunk_size: int = 600,
                 chunk_overlap: int = 150):

        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Text splitter with custom separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\nΆρθρο",
                "\n\nΠαράγραφος", 
                "\n\n", 
                "\n", 
                ". ", 
                "! ", 
                "? ", 
                "; ",
                ", ",
                " ", 
                ""
            ]
        )
        
        # Indices for keywords, phrases, and context
        self.keyword_index = defaultdict(list)
        self.phrase_index = defaultdict(list)
        self.context_index = defaultdict(list)
        
        # Extended important keywords
        self.important_keywords = {
            'εγκυμοσύνη': [
                'εγκυος', 'εγκύου', 'εγκυμοσύνη', 'εγκυμοσύνης', 'εγκυμόνων',
                'μητρότητα', 'μητρότητας', 'μητέρα', 'μητέρας',
                'κύηση', 'κύησης', 'κυοφορία', 'τοκετός', 'γέννα',
                'λοχεία', 'προγεννητική', 'μετεγγενετική'
            ],
            
            'άδεια': [
                'άδεια', 'άδειας', 'αδείας', 'άδειες', 'αδειών',
                'αναρρωτική', 'αναρρωτικής', 'ανάρρωση',
                'ασθένεια', 'ασθενείας', 'ασθενή',
                'κανονική άδεια', 'ετήσια άδεια', 'ειδική άδεια',
                'άνευ αποδοχών', 'απουσία'
            ],
            
            'απόλυση': [
                'απόλυση', 'απόλυσης', 'απολύω', 'απολύεται',
                'καταγγελία', 'καταγγελίας', 'καταγγέλλω',
                'διαγραφή', 'τερματισμός', 'λύση σύμβασης',
                'παύση εργασίας', 'αποχώρηση', 'αποβολή'
            ],
            
            'αποζημίωση': [
                'αποζημίωση', 'αποζημιώσεις', 'αποζημίωσης', 'αποζημιώσεων',
                'αποδημίωση', 'αποδημιώσεις', 'αποδημίωσης',
                'αποζημιωτικό', 'αποδημιωτικό', 'χρηματική παροχή',
                'οφειλόμενα ποσά', 'αποζημιωτική παροχή'
            ],
            
            'ωράριο': [
                'ωράριο', 'ωραρίου', 'ωράρια', 'ωραρίων',
                'ώρες εργασίας', 'εργάσιμες ώρες', 'διάρκεια εργασίας',
                'υπερωρίες', 'υπερωριών', 'υπερωριακή',
                'νυχτερινή εργασία', 'κυριακάτικη εργασία',
                'εορταστική εργασία', 'βάρδια','βάρδιες',
                'εργασιακό ωράριο', 'εργασιακή διάρκεια','υπερωρία'
            ],
            
            'αμοιβή': [
                'μισθός', 'μισθού', 'μισθοί', 'μισθών',
                'αμοιβή', 'αμοιβής', 'αμοιβές', 'αμοιβών',
                'πληρωμή', 'απολαβές', 'αποδοχές',
                'ημερομίσθιο', 'μηνιαίος μισθός', 'ωρομίσθιο',
                'βασικός μισθός', 'κατώτατος μισθός', 'ελάχιστος μισθός'
            ],
            
            'ασφάλιση': [
                'ασφάλιση', 'ασφάλισης', 'ασφαλιστικό', 'ασφαλιστικές',
                'ΙΚΑ', 'ΕΦΚΑ', 'εισφορές', 'εισφορών',
                'κοινωνική ασφάλιση', 'ασφαλιστικά ταμεία',
                'συνταξιοδότηση', 'κοινωνικός τομέας'
            ],
            'επίδομα': [
                'επίδομα', 'επιδόματα', 'επιδόματος', 'επιδόσεων',
                'δώρο Χριστουγέννων', 'δώρο Πάσχα', 'επίδομα αδείας',
                'επίδομα ανεργίας', 'επίδομα μητρότητας', 'επιδόματα τέκνων',
                'επιδόματα αναπηρίας', 'επιδόματα στέγασης',
                'επιδόματα κοινωνικής πρόνοιας'
            ],
            'συμβάσεις': [
                'σύμβαση', 'συμβάσεις', 'σύμβασης', 'συμβάσεων',
                'εργασιακή σύμβαση', 'συλλογική σύμβαση', 'ατομική σύμβαση'
                'συμβατική υποχρέωση', 'συμβατική σχέση',
                'συμβατική αποζημίωση', 'συμβατική διάρκεια',
                'σύμβαση εργασίας', 'σύμβαση ορισμένου χρόνου',
                'σύμβαση αορίστου χρόνου'
            ],
            
            'εργασιακά δικαιώματα': [
                'εργασιακά δικαιώματα', 'εργασιακά καθήκοντα',
                'εργασιακή σχέση', 'εργασιακή ασφάλεια',
                'εργασιακή προστασία', 'εργασιακή ευημερία',
                'εργασιακή ισότητα', 'εργασιακή δικαιοσύνη',
                'εργασιακή συνθήκη', 'εργασιακή νομοθεσία',
                'εργασιακή πολιτική', 'εργασιακή κουλτούρα'
            ],
            
            'εργασιακή ασφάλεια': [
                'εργασιακή ασφάλεια', 'ασφάλεια εργασίας',
                'υγιεινή και ασφάλεια', 'εργασιακοί κίνδυνοι',
                'εργασιακά ατυχήματα', 'προστασία εργαζομένων',
                'μέτρα ασφαλείας', 'ασφαλιστικά μέτρα',
                'προληπτική ασφάλεια', 'εργασιακή υγιεινή',
                'εργασιακή προστασία', 'εργασιακή ευημερία'
            ],
            
            'εργασιακή εκπαίδευση': [
                'εργασιακή εκπαίδευση', 'επαγγελματική κατάρτιση',
                'επαγγελματική εκπαίδευση', 'επαγγελματική ανάπτυξη',
                'επαγγελματική κατάρτιση', 'επαγγελματική πιστοποίηση',
                'επαγγελματική εκπαίδευση', 'επαγγελματική εξέλιξη',
                'επαγγελματική κατάρτιση', 'επαγγελματική εκπαίδευση',
                'επαγγελματική ανάπτυξη', 'επαγγελματική πιστοποίηση'
            ],
            
            'εργασιακή ηθική': [
                'εργασιακή ηθική', 'επαγγελματική ηθική',
                'επαγγελματική συμπεριφορά', 'επαγγελματική ευθύνη'],
            
            'επιδόματα': [
                'επιδόματα', 'επιδόματος', 'επιδόματα', 'επιδόσεων',
                'επίδομα ανεργίας', 'επίδομα μητρότητας',
                'επίδομα τέκνων', 'επίδομα αναπηρίας',
                'επίδομα στέγασης', 'επίδομα κοινωνικής πρόνοιας',
                'επίδομα αδείας', 'δώρο Χριστουγέννων', 'δώρο Πάσχα'
            ],
            
            'ένσημα' : [
                'ένσημα', 'ένσημο', 'ένσημα εργασίας', 'ένσημα ασφάλισης',
                'εργασιακά ένσημα', 'ασφαλιστικά ένσημα',
                'ένσημα ΙΚΑ', 'ένσημα ΕΦΚΑ', 'ένσημα κοινωνικής ασφάλισης'
            ],
            
            'σύνταξη' : [
                'σύνταξη', 'συνταξιοδότηση', 'συνταξιούχος',
                'συνταξιοδοτικό', 'συνταξιοδοτικά δικαιώματα',
                'συνταξιοδοτική ασφάλιση', 'συνταξιοδοτικό ταμείο',
                'συνταξιοδοτική παροχή', 'συνταξιοδοτική ασφάλιση',
                'συνταξιοδοτικό πρόγραμμα', 'συνταξιοδοτική πολιτική'
            ],
            
            'δώρο' : [
                'δώρο', 'δώρου', 'δώρα', 'δώρων',
                'δώρο Χριστουγέννων', 'δώρο Πάσχα', 'δώρο αδείας',
                'δώρο εορτών', 'δώρο εργαζομένων', 'δώρο κοινωνικής πρόνοιας',
                'δώρο επιδόματος', 'δώρο εορτών Χριστουγέννων',
                'δώρο εορτών Πάσχα', 'δώρο εορτών αδείας'
            ],
            
            'άδεια' : [
                'άδεια', 'άδειας', 'άδειες', 'αδειών',
                'άδεια μητρότητας', 'αναρρωτική άδεια', 'κανονική άδεια'],
            
            'απαγόρευση' : [
                'απαγόρευση', 'απαγορεύεται', 'απαγορευμένο',
                'απαγορευμένες εργασίες', 'απαγορευμένη απασχόληση'],
            
            'ποινή' : [
                'ποινή', 'ποινές', 'ποινής', 'ποινών',
                'ποινική ευθύνη', 'ποινικές συνέπειες']
            
            
            
        }
        
        # Important phrases
        self.important_phrases = [
            'άδεια μητρότητας', 'αναρρωτική άδεια', 'κανονική άδεια',
            'καταγγελία σύμβασης', 'λύση σύμβασης', 'τερματισμός απασχόλησης',
            'αποζημίωση απόλυσης', 'προειδοποίηση απόλυσης',
            'ώρες εργασίας', 'υπερωριακή απασχόληση', 'νυχτερινή εργασία',
            'βασικός μισθός', 'κατώτατος μισθός', 'ελάχιστος μισθός',
            'δώρο Χριστουγέννων', 'επίδομα αδείας', 'κοινωνική ασφάλιση',
            'ασφαλιστικές εισφορές', 'συνταξιοδότηση', 'εργασιακή ασφάλεια',
            'υγιεινή και ασφάλεια', 'εργασιακή εκπαίδευση',
            'επαγγελματική κατάρτιση', 'επαγγελματική ανάπτυξη',
            'εργασιακή ηθική', 'επαγγελματική ηθική',
            'εργασιακή ευημερία', 'εργασιακή προστασία',
            'εργασιακή ισότητα', 'εργασιακή δικαιοσύνη',
            'εργασιακή συνθήκη', 'εργασιακή νομοθεσία',
            'εργασιακή πολιτική', 'εργασιακή κουλτούρα',
            'εργασιακά δικαιώματα', 'εργασιακά καθήκοντα',
            'εργασιακή σχέση', 'εργασιακή ασφάλεια',
            'ένσημα', 'ένσημα εργασίας',
            'εργασιακή ασφάλιση', 'εργασιακή προστασία',
            'εργασιακή εκπαίδευση', 'εργασιακή ηθική',
            
        ]
        
        # TF-IDF for semantic matching
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words=None
        )
        
        # Greek stems
        self.greek_stems = self._create_greek_stems()
    
    def _create_greek_stems(self) -> Dict[str, str]:
        """Create simple Greek stems for keywords"""
        stems = {}
        
        endings = [
            ('ματος', 'μα'), ('ματα', 'μα'), ('μάτων', 'μα'),
            ('σεως', 'ση'), ('σεις', 'ση'), ('σεων', 'ση'),
            ('ότητας', 'ότητα'), ('ότητες', 'ότητα'),
            ('ιστής', 'ιστ'), ('ιστή', 'ιστ'), ('ιστές', 'ιστ'),
            ('ουν', 'ω'), ('εις', 'ω'), ('ει', 'ω'),
        ]
        
        for category, keywords in self.important_keywords.items():
            for keyword in keywords:
                stems[keyword] = keyword
                for ending, replacement in endings:
                    if keyword.endswith(ending):
                        stem = keyword[:-len(ending)] + replacement
                        stems[stem] = keyword
        
        return stems
    
    def normalize_text(self, text: str) -> str:
        """Normalize text"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()
    
    def extract_enhanced_keywords(self, text: str) -> Dict[str, Any]:
        """Extract keywords, phrases, and context snippets"""
        normalized_text = self.normalize_text(text)
        
        found_keywords = set()
        found_phrases = set()
        keyword_positions = []
        
        # Keywords matching
        for category, keyword_list in self.important_keywords.items():
            for keyword in keyword_list:
                if keyword in normalized_text:
                    found_keywords.add(keyword)
                    found_keywords.add(category)
                    pos = normalized_text.find(keyword)
                    keyword_positions.append((keyword, pos))
        
        # Phrase matching
        for phrase in self.important_phrases:
            if phrase.lower() in normalized_text:
                found_phrases.add(phrase)
        
        # Stemming-based matching
        words = re.findall(r'\b\w+\b', normalized_text)
        for word in words:
            if word in self.greek_stems:
                found_keywords.add(self.greek_stems[word])
        
        # Context snippets
        context_snippets = []
        for keyword, pos in keyword_positions:
            start = max(0, pos - 100)
            end = min(len(normalized_text), pos + 100)
            context = normalized_text[start:end]
            context_snippets.append(context)
        
        return {
            'keywords': found_keywords,
            'phrases': found_phrases,
            'keyword_positions': keyword_positions,
            'context_snippets': context_snippets
        }
    
    def create_smart_chunks_v2(self, text: str, metadata: Dict) -> List[Dict]:
        """Create smart chunks with article awareness"""
        article_pattern = r'άρθρο\s*(\d+)'
        articles = list(re.finditer(article_pattern, text, re.IGNORECASE))
        
        chunks = []
        
        if len(articles) > 1:
            # Chunking by articles
            for i, article_match in enumerate(articles):
                start_pos = article_match.start()
                end_pos = articles[i + 1].start() if i + 1 < len(articles) else len(text)
                
                article_text = text[start_pos:end_pos].strip()
                article_num = article_match.group(1)
                
                if len(article_text) > self.chunk_size * 1.5:
                    sub_chunks = self.text_splitter.split_text(article_text)
                    for j, sub_chunk in enumerate(sub_chunks):
                        if len(sub_chunk.strip()) < 50:
                            continue
                        
                        enhanced_keywords = self.extract_enhanced_keywords(sub_chunk)
                        chunks.append({
                            'text': sub_chunk,
                            'article_num': article_num,
                            'sub_chunk': j,
                            'total_sub_chunks': len(sub_chunks),
                            'enhanced_keywords': enhanced_keywords,
                            'chunk_type': 'article_sub'
                        })
                else:
                    enhanced_keywords = self.extract_enhanced_keywords(article_text)
                    chunks.append({
                        'text': article_text,
                        'article_num': article_num,
                        'sub_chunk': 0,
                        'total_sub_chunks': 1,
                        'enhanced_keywords': enhanced_keywords,
                        'chunk_type': 'article_full'
                    })
        else:
            # No articles found, regular chunking
            regular_chunks = self.text_splitter.split_text(text)
            for i, chunk in enumerate(regular_chunks):
                if len(chunk.strip()) < 50:
                    continue
                
                enhanced_keywords = self.extract_enhanced_keywords(chunk)
                chunks.append({
                    'text': chunk,
                    'article_num': None,
                    'sub_chunk': i,
                    'total_sub_chunks': len(regular_chunks),
                    'enhanced_keywords': enhanced_keywords,
                    'chunk_type': 'regular'
                })
        
        return chunks
    
    def process_jsonl_v2(self, file_path: str) -> List[Document]:
        """Process JSONL file and create enhanced chunks"""
        print(f"Checking file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            
        
        docs = []
        all_texts = []
        
        print(f"Loading data from {file_path}...")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        original_text = data.get("text", "")
                        
                        if len(original_text.strip()) < 50:
                            print(f"Row {line_num}: Text too short, skipping")
                            continue
                        
                        cleaned_text = self.normalize_text(original_text)
                        smart_chunks = self.create_smart_chunks_v2(cleaned_text, data)
                        
                        for chunk_data in smart_chunks:
                            if len(chunk_data['text'].strip()) < 30:
                                continue
                            
                            enhanced_metadata = {
                                "source_line": line_num,
                                "volume": data.get("volume", "Άγνωστος Τόμος"),
                                "chapter": data.get("chapter", "Άγνωστο Κεφάλαιο"),
                                "subject": data.get("subject", "Άγνωστο Θέμα"),
                                "chunk_id": str(uuid.uuid4()),
                                "chunk_length": len(chunk_data['text']),
                                "article_num": chunk_data.get('article_num'),
                                "sub_chunk": chunk_data.get('sub_chunk', 0),
                                "total_sub_chunks": chunk_data.get('total_sub_chunks', 1),
                                "chunk_type": chunk_data.get('chunk_type', 'regular'),
                                "keywords": list(chunk_data['enhanced_keywords']['keywords']),
                                "phrases": list(chunk_data['enhanced_keywords']['phrases']),
                                "keyword_count": len(chunk_data['enhanced_keywords']['keywords']),
                                "phrase_count": len(chunk_data['enhanced_keywords']['phrases']),
                                "has_important_content": len(chunk_data['enhanced_keywords']['keywords']) > 0
                            }
                            
                            doc = Document(
                                page_content=chunk_data['text'].strip(),
                                metadata=enhanced_metadata
                            )
                            docs.append(doc)
                            all_texts.append(chunk_data['text'])
                            
                            doc_index = len(docs) - 1
                            
                            # Update indices
                            for keyword in chunk_data['enhanced_keywords']['keywords']:
                                self.keyword_index[keyword].append(doc_index)
                            
                            for phrase in chunk_data['enhanced_keywords']['phrases']:
                                self.phrase_index[phrase].append(doc_index)
                            
                            for context in chunk_data['enhanced_keywords']['context_snippets']:
                                context_words = set(re.findall(r'\b\w+\b', context.lower()))
                                for word in context_words:
                                    if len(word) > 3:
                                        self.context_index[word].append(doc_index)
                    
                    except json.JSONDecodeError as e:
                        print(f"Error JSON in line {line_num}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error in line {line_num}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error during file reading: {e}")
            return []
        
        # Training TF-IDF
        if all_texts:
            print("Training TF-IDF vectorizer...")
            try:
                self.tfidf_vectorizer.fit(all_texts)
                print("TF-IDF training completed")
            except Exception as e:
                print(f"Error TF-IDF: {e}")
        else:
            print("No texts for TF-IDF training")
        
        print(f"Created {len(docs)} chunks")
        print(f"Keyword index: {len(self.keyword_index)} unique keywords")
        print(f"Phrase index: {len(self.phrase_index)} unique phrases")
        print(f"Context index: {len(self.context_index)} unique context words")
        
        return docs
    
    def create_enhanced_vector_store(self, docs: List[Document], save_path: str = "./faiss-files/faiss_laws_hybrid"):
        """Create enhanced vector store with FAISS"""
        if not docs:
            print("No documents to index")
            return None
        
        print(f"Create embeddings using model: {self.embedding_model_name}")
        
        try:
            embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)
            db = FAISS.from_documents(docs, embedding_model)
            
            # Saving FAISS index
            db.save_local(save_path)
            print(f"Enhanced FAISS index: '{save_path}'")
            
            # Saving  all indices
            indices_data = {
                'keyword_index': dict(self.keyword_index),
                'phrase_index': dict(self.phrase_index),
                'context_index': dict(self.context_index),
                'important_keywords': self.important_keywords,
                'important_phrases': self.important_phrases
            }
            
            indices_path = f"{save_path}_indices.json"
            with open(indices_path, 'w', encoding='utf-8') as f:
                json.dump(indices_data, f, ensure_ascii=False, indent=2)
            print(f"Indices: '{indices_path}'")
            
            # Save TF-IDF vectorizer
            tfidf_path = f"{save_path}_tfidf.pkl"
            try:
                with open(tfidf_path, 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f)
                print(f"TF-IDF vectorizer: '{tfidf_path}'")
            except Exception as e:
                print(f"Error saving TF-IDF: {e}")
            
            return db
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None
    
    def analyze_enhanced_coverage(self, docs: List[Document]):
        """Analyze enhanced coverage of keywords and phrases"""
        print("\nEnhanced Coverage Analysis:")
        
        print(f"\nTotal Documents: {len(docs)}")
        
        # Analyze keywords
        print("\Keywords:")
        for category, keywords in self.important_keywords.items():
            total_chunks = sum(len(self.keyword_index.get(keyword, [])) for keyword in keywords)
            print(f"   {category}: {total_chunks} chunks")
        
        # Analyze phrases
        print("\Phrases:")
        phrase_count = 0
        for phrase in self.important_phrases:
            chunk_count = len(self.phrase_index.get(phrase, []))
            if chunk_count > 0:
                print(f"   '{phrase}': {chunk_count} chunks")
                phrase_count += 1
        
        if phrase_count == 0:
            print("   No important phrases found in chunks")
        
        # Chunk types
        chunk_types = Counter([doc.metadata.get('chunk_type', 'unknown') for doc in docs])
        print(f"\nChunk types:")
        for chunk_type, count in chunk_types.items():
            print(f"   {chunk_type}: {count}")
        
        # Chunks with important content
        important_chunks = sum(1 for doc in docs if doc.metadata.get('has_important_content', False))
        percentage = (important_chunks/len(docs)*100) if len(docs) > 0 else 0
        print(f"\nChunks with important content: {important_chunks}/{len(docs)} ({percentage:.1f}%)")
        
        # Sample content
        print(f"\nSample content:")
        for i, doc in enumerate(docs[:3]):
            print(f"   Document {i+1}:")
            print(f"      Lenght: {len(doc.page_content)}")
            print(f"      Keywords: {doc.metadata.get('keyword_count', 0)}")
            print(f"      Content: {doc.page_content[:150]}...")

def main():
    print("="*50)
    
    processor = EnhancedHybridRAGProcessor(
        embedding_model_name="intfloat/multilingual-e5-large",
        chunk_size=600,
        chunk_overlap=150
    )
    
    try:
        # Edit the path to your JSONL file here
        docs = processor.process_jsonl_v2("./datasets/ergatiki_nomothesia_all.jsonl")
        
        if not docs:
            print("No documents processed, exiting")
            return
        
        # Analysis of enhanced coverage
        processor.analyze_enhanced_coverage(docs)
        
        # create vector store
        db = processor.create_enhanced_vector_store(docs, "./faiss-files/faiss_laws_hybrid")
        
        if db:
            print("\nSuccessfully created vector store")
            print("\nCreated files:")
            print("   • faiss_laws_hybrid/ (FAISS index)")
            print("   • faiss_laws_hybrid.json (Keywords & Phrases)")
            print("   • faiss_laws_hybrid.pkl (TF-IDF model)")
            print("\nYou can now use the 'faiss_laws_hybrid' vector store in your RAG system.")
        else:
            print("Error creating vector store")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()