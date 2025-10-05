import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

import sys
import json
import re
import requests
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import το query system
from query import MultiSourceRAGQuery, QueryConfig

class RemoteLLMClient:
    """Client for remote LLM"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        # Different aproaches for remote APIs
        self.available_methods = {
            "google_ai": self._call_google_ai,
            "openai_compatible": self._call_openai_compatible,
            "huggingface_inference": self._call_huggingface_inference
        }
        
        self.current_method = "google_ai" if api_key else "huggingface_inference"
        
    def _call_google_ai(self, prompt: str, max_tokens: int = 2048) -> Dict[str, Any]:
        """Google AI API call"""
        if not self.api_key:
            return {"success": False, "error": "Requires Google AI API key"}
        
        url = f"{self.base_url}/models/gemma-3n-e2b-it:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                    return {
                        "success": True,
                        "response": text.strip(),
                        "method": "google_ai",
                        "model": "gemma-3n-e2b-it"
                    }
                else:
                    return {"success": False, "error": "No candidates in response from Google AI"}
            else:
                error_msg = f"Google AI API error: {response.status_code}"
                if response.text:
                    error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                    error_msg += f" - {error_data}"
                return {"success": False, "error": error_msg}
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Timeout στο Google AI API"}
        except Exception as e:
            return {"success": False, "error": f"Google AI error: {str(e)}"}
    
    def _call_openai_compatible(self, prompt: str, max_tokens: int = 2048) -> Dict[str, Any]:
        """OpenAI compatible API call"""
        url = "http://localhost:11434/v1/chat/completions"  # Ollama endpoint
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gemma2:2b",  # or other model available
            "messages": [
                {
                    "role": "system",
                    "content": "Είσαι εξειδικευμένος νομικός σύμβουλος στο εργατικό δίκαιο."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    text = result["choices"][0]["message"]["content"]
                    return {
                        "success": True,
                        "response": text.strip(),
                        "method": "openai_compatible",
                        "model": data["model"]
                    }
            
            return {"success": False, "error": f"OpenAI compatible API error: {response.status_code}"}
            
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Δεν μπόρεσα να συνδεθώ στο local API"}
        except Exception as e:
            return {"success": False, "error": f"OpenAI compatible error: {str(e)}"}
    
    def _call_huggingface_inference(self, prompt: str, max_tokens: int = 2048) -> Dict[str, Any]:
        """HuggingFace Inference API call"""
        
        models_to_try = [
            "microsoft/DialoGPT-medium",
            "google/flan-t5-large",
            "bigscience/bloom-1b7"
        ]
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:  # HF token
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        for model in models_to_try:
            url = f"https://api-inference.huggingface.co/models/{model}"
            
            data = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True
                }
            }
            
            try:
                response = requests.post(url, headers=headers, json=data, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        if "generated_text" in result[0]:
                            text = result[0]["generated_text"]
                            if text.startswith(prompt):
                                text = text[len(prompt):].strip()
                            
                            return {
                                "success": True,
                                "response": text,
                                "method": "huggingface_inference",
                                "model": model
                            }
                
                continue
                
            except Exception as e:
                continue
        
        return {"success": False, "error": "All HuggingFace models failed"}
    
    def generate(self, prompt: str, max_tokens: int = 2048, method: str = None) -> Dict[str, Any]:
        """Κύρια μέθοδος για generation"""
        
        method = method or self.current_method
        
        if method not in self.available_methods:
            return {"success": False, "error": f"Unknown method {method}"}
        
        print(f"Sending to remote LLM ({method})...")
        
        result = self.available_methods[method](prompt, max_tokens)
        
        if result["success"]:
            print(f"Successfull answer from {result.get('model', 'remote LLM')}")
        else:
            print(f"Error in remote LLM: {result.get('error', 'Unknown error')}")
        
        return result
    
    def test_connection(self) -> Dict[str, Any]:
        """Test all available methods"""
        test_prompt = "Γεια σου! Αυτό είναι ένα τεστ."
        
        results = {}
        for method_name in self.available_methods.keys():
            print(f"Testing {method_name}...")
            result = self.generate(test_prompt, max_tokens=50, method=method_name)
            results[method_name] = {
                "success": result["success"],
                "error": result.get("error", ""),
                "model": result.get("model", "")
            }
            time.sleep(1)  # Avoid rate limiting
        
        return results

class EnhancedMultiSourceChatbot:
    """
    Enhanced Multi-Source Chatbot with Local LLM and Remote LLM Review Mode
    """
    
    def __init__(self, 
                 qa_faiss_path: str = "./faiss-files/faiss_qa_hybrid",
                 notes_faiss_path: str = "./faiss-files/faiss_notes_hybrid", 
                 laws_faiss_path: str = "./faiss-files/faiss_laws_hybrid",
                 embedding_model: str = "intfloat/multilingual-e5-large",
                 llm_model_id: str = "./models/gemma-greek-4b-legal-light-epoch7-lr4e-5-bs1-gas8",
                 hf_token: str = None,
                 google_ai_key: str = None):
        """
        Initializing Multi-Source Chatbot
        """
        self.llm_model_id = llm_model_id
        self.hf_token = hf_token
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Enhanced Multi-Source RAG Chatbot with LLM Review")
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        
        # Initialization of Multi-Source Query System
        print("Initialization of Multi-Source Query System...")
        try:
            self.query_system = MultiSourceRAGQuery(
                qa_faiss_path=qa_faiss_path,
                notes_faiss_path=notes_faiss_path,
                laws_faiss_path=laws_faiss_path,
                embedding_model=embedding_model
            )
            print("Multi-Source Query System initialized successfully.")
        except Exception as e:
            print(f"Error initializing Query System: {e}")
            raise
        
        # Initializing Remote LLM Client
        self.remote_llm = RemoteLLMClient(api_key=google_ai_key or hf_token)
        
        # Login in HuggingFace
        if hf_token:
            login(token=hf_token)
        
        # Load local LLM
        self.load_llm()
        
        # Default configuration for queries
        self.default_config = QueryConfig(
            qa_weight=0.6,
            notes_weight=0.35,
            laws_weight=0.05,
            max_results_per_source=20,
            final_max_results=40
        )
        
        # Cache for last query and context
        self.last_query = None
        self.last_context = None
        self.last_local_response = None
        self.last_query_result = None
    
    def load_llm(self):
        """Load local LLM model with dtype fallbacks and memory management """
        print(f"Load LLM model: {self.llm_model_id}")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_id,
                trust_remote_code=True
            )

            # try different dtypes for memory efficiency
            tried_dtypes = []
            model_loaded = False
            for dtype in (torch.bfloat16, torch.float16):
                try:
                    tried_dtypes.append(dtype)
                    print(f"   • Try loading with dtype={dtype}")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.llm_model_id,
                        torch_dtype=dtype,
                        device_map="auto" if self.device.type == "cuda" else None,
                        trust_remote_code=True
                    )
                    model_loaded = True
                    print(f"   Loade with dtype={dtype}")
                    break
                except Exception as e:
                    print(f"   Failed loading with {dtype}: {e}")
                    # Cleanup
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

            if not model_loaded:
                # Last resort: load without dtype (CPU/float32)
                print("   • Fallback: load without torch_dtype (CPU/float32)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_id,
                    device_map="auto" if self.device.type == "cuda" else None,
                    trust_remote_code=True
                )

            if self.device.type == "cpu":
                self.model = self.model.to(self.device)

            # Deactivate training mode
            self.model.eval()
            print("Successfully loaded local LLM model.")

        except Exception as e:
            print(f"Error loading local LLM: {e}")
            raise
    
    def create_context_from_results(self, query_result: Dict[str, Any], max_length: int = 8000) -> str:
        final_results = query_result['final_results']
        if not final_results:
            return "Didnt find any relevant documents."

        # 1) take weights from config or defaults
        cfg = query_result.get('config', {}) or {
            'qa_weight': self.default_config.qa_weight,
            'notes_weight': self.default_config.notes_weight,
            'laws_weight': self.default_config.laws_weight,
        }
        weight_by_source = {
            'qa': float(cfg.get('qa_weight', self.default_config.qa_weight)),
            'notes': float(cfg.get('notes_weight', self.default_config.notes_weight)),
            'laws': float(cfg.get('laws_weight', self.default_config.laws_weight)),
        }

        # 2) Sort results by combined score = raw_score * weight
        def combined_score_of(item: Dict[str, Any]) -> float:
            st = (item.get('source_type') or '').lower()
            raw_score = float(item.get('score', 0.0))
            return raw_score * weight_by_source.get(st, 1.0)

        final_results_sorted = sorted(final_results, key=combined_score_of, reverse=True)

        context_parts = []
        total_length = 0

        for i, result_data in enumerate(final_results_sorted, 1):
            if total_length >= max_length:
                break

            source_type = (result_data.get('source_type') or '').upper()
            metadata = result_data.get('metadata', {})
            raw_score = float(result_data.get('score', 0.0))
            combined_score = raw_score * weight_by_source.get(source_type.lower(), 1.0)

            # Construct content
            if source_type == 'QA':
                answer = (result_data.get('answer') or '').strip()
                if not answer:
                    raw_content = (result_data.get('content') or '').strip()
                    content = self.clean_content(raw_content)
                else:
                    content = f"Απάντηση: {answer}"
            else:
                raw_content = (result_data.get('content') or '').strip()
                content = self.clean_content(raw_content)

            if not content or len(content) < 50:
                continue

            # Priority based combined_score
            priority = "🔥" if combined_score >= 3 else "⭐" if combined_score >= 1 else "📄"

            # Header: write source info
            header = f"[{priority} {source_type} #{i} | score={combined_score:.3f}]"

            useful_meta = []
            if metadata.get("article_num"):
                useful_meta.append(f"Άρθρο {metadata['article_num']}")
            if metadata.get("law_title"):
                useful_meta.append(f"Νόμος: {metadata['law_title']}")
            if useful_meta:
                header += " - " + ", ".join(useful_meta)

            remaining_space = max_length - total_length
            if remaining_space < 200:
                break
            if len(content) > remaining_space - 200:
                content = content[:remaining_space - 200] + "..."

            context_parts.append(f"{header}\n{content}")
            total_length += len(context_parts[-1]) + 2

            if i >= 15:
                break

        context = "\n\n".join(context_parts)
        print(f"Clean Context Info:")
        print(f"   • Selected: {len(context_parts)} documents")
        print(f"   • Total length: {len(context)} characters")
        return context
    
    
    def clean_content(self, content: str) -> str:
        patterns_to_remove = [
            r'\[🔥 NOTES #\d+ - Score: [\d.]+ - \w+\]',
            r'\[⭐ \w+ #\d+ - Score: [\d.]+ - \w+\]',
            r'\[📄 \w+ #\d+ - Score: [\d.]+ - \w+\]',
            r'Κλάση:\s*[^,\n]+,?\s*',
            r'Τίτλος:\s*[^,\n]+,?\s*',
            r'Κατηγορία:\s*[^,\n]+,?\s*',
            r'Θέμα:\s*[^,\n]+,?\s*',
            r'Περιεχόμενο:\s*',
            r'Score:\s*[\d.]+\s*-\s*\w+',
            r'Weight:\s*\w+\s*=\s*[\d.]+',
            r'Ανάλυση:\s*[^\n]*\n?',
            r'\* Weight:[^\n]*\n?',
            r'\* Ανάλυση:[^\n]*\n?',
        ]
        cleaned_content = content
        for pattern in patterns_to_remove:
            cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.MULTILINE | re.IGNORECASE)

        cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
        cleaned_content = re.sub(r'^\s+', '', cleaned_content, flags=re.MULTILINE).strip()

        lines = cleaned_content.split('\n')
        cleaned_lines = []
        skip_keywords = [
            'ΕΡΩΤΗΣΗ ΧΡΗΣΤΗ:',
            # 'Απάντηση:' # keep user question if appears in content
            'Weight:',
            'Score:',
            'Ανάλυση:',
        ]
        for line in lines:
            l = line.strip()
            if not l:
                cleaned_lines.append('')
                continue
            if any(l.startswith(k) for k in skip_keywords):
                continue
            cleaned_lines.append(l)

        return '\n'.join(cleaned_lines).strip()
    
    
    def create_advanced_prompt(self, query: str, context: str, query_result: Dict[str, Any]) -> str:
        prompt = f"""Είσαι εξειδικευμένος νομικός σύμβουλος στο εργατικό δίκαιο.

        ΚΡΙΣΙΜΕΣ ΟΔΗΓΙΕΣ:
        - Απάντησε μόνο βάσει των παρεχόμενων νομικών κειμένων.
        - Στάθμισε περισσότερο τα κείμενα με υψηλότερο συνδυαστικό βαθμό (score στο context).
        - ΜΗΝ αναφέρεις score, weight, metadata ή τεχνικές λεπτομέρειες στην απάντηση.
        - Δώσε μόνο το ουσιαστικό νομικό συμπέρασμα με σαφείς αναφορές
        - Αν κάτι δεν καλύπτεται επαρκώς, δήλωσέ το ρητά.
        - Απάντησε στα ελληνικά με ακρίβεια.
        - Μην επαναλαμβάνεις την ερώτηση στην απάντηση.

        ΝΟΜΙΚΑ ΚΕΙΜΕΝΑ:
        {context}

        ΕΡΩΤΗΣΗ: {query}

        ΑΠΑΝΤΗΣΗ (μόνο το νομικό συμπέρασμα):"""
        return prompt
    
    def generate_local_response(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """Τοπική γενιά απάντησης — με έλεγχο context size & safer generate call"""
        messages = [
            {"role": "system", "content": "Είσαι εξειδικευμένος νομικός σύμβουλος..."},
            {"role": "user", "content": prompt}
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Tokenize with truncation (safeguard)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096  # arbitrary large limit, will be checked later
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]
        max_pos = getattr(self.model.config, "max_position_embeddings", 4096)

        # Adjust max_new_tokens if input is too long
        safe_new = max_new_tokens
        if input_len + max_new_tokens + 8 > max_pos:
            safe_new = max(16, max_pos - input_len - 8)  # leave some margin
            print(f"Truncating max_new_tokens: requested={max_new_tokens}, allowed={safe_new} (model.max_pos={max_pos}, input_len={input_len})")

        print("Data input:", inputs["input_ids"].shape)
        print(f"Generate local answer ({self.device.type.upper()}) with max_new_tokens={safe_new}...")

        # Safer generate call with try-except
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=safe_new,
            do_sample=True,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )

        # Safer generate with fallback
        try:
            with torch.no_grad():
                outputs = self.model.generate(**generate_kwargs)
        except RuntimeError as e:
            # Fallback: try again with reduced max_new_tokens
            print(f"RuntimeError κατά την generate(): {e}")
            print("   • Trying again with reduced max_new_tokens...")
            try:
                generate_kwargs["max_new_tokens"] = min(64, safe_new)
                with torch.no_grad():
                    outputs = self.model.generate(**generate_kwargs)
            except Exception as e2:
                print(f"Secondary generate failed: {e2}")
                raise

        response = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )

        return response.strip()
    
    def generate_remote_response(self, query: str, context: str, method: str = None) -> Dict[str, Any]:
        """Generate response from remote LLM"""
        
        remote_prompt = f"""Είσαι εξειδικευμένος νομικός σύμβουλος στο ελληνικό εργατικό δίκαιο. 

            ΟΔΗΓΙΕΣ:
            - Απάντησε βάσει των παρεχόμενων νομικών κειμένων από πολλαπλές πηγές
            - Αν δεν βρεις σχετικά κείμενα, σκέψου και απάντησε λογικά συνδυάζοντας τις γνώσεις που ήδη έχεις
            - Αναφέρε συγκεκριμένα άρθρα και νόμους
            - Απάντησε στα ελληνικά με ακρίβεια
            - Αν κάτι δεν καλύπτεται από τα κείμενα, δήλωσέ το

            ΠΟΛΥΠΗΓΕΣ ΝΟΜΙΚΑ ΚΕΙΜΕΝΑ:
            {context}

            ΕΡΩΤΗΣΗ:
            {query}

            ΑΠΑΝΤΗΣΗ:"""
                    
        return self.remote_llm.generate(remote_prompt, max_tokens=2048, method=method)
    
    def generate_remote_review_response(self, query: str, context: str, local_response: str, method: str = None) -> Dict[str, Any]:
        
        review_prompt = f"""Είσαι εμπειρογνώμονας νομικός σύμβουλος στο ελληνικό εργατικό δίκαιο. 

            Σου δίνεται μια ερώτηση, σχετικά νομικά κείμενα από πολλαπλές πηγές, και μια απάντηση που έδωσε ένα άλλο AI σύστημα. 

            Η δουλειά σου είναι να:
            1. ΑΝΑΛΥΣΕΙΣ την υπάρχουσα απάντηση
            2. ΕΝΤΟΠΙΣΕΙΣ τυχόν λάθη, ελλείψεις ή ανακρίβειες
            3. ΔΙΟΡΘΩΣΕΙΣ και ΒΕΛΤΙΩΣΕΙΣ την απάντηση
            4. ΠΑΡΕΧΕΙΣ μια πλήρη, ακριβή και καλά τεκμηριωμένη νομική απάντηση

            ΚΡΙΤΙΚΕΣ ΟΔΗΓΙΕΣ:
            - Χρησιμοποίησε τα παρεχόμενα νομικά κείμενα ως κύρια πηγή
            - Αναφέρε συγκεκριμένα άρθρα, νόμους και κανονισμούς
            - Αν η υπάρχουσα απάντηση είναι σωστή, επιβεβαίωσέ την και εμπλούτισέ την
            - Αν έχει λάθη, διόρθωσέ τα με σαφήνεια
            - Αν λείπουν στοιχεία, συμπλήρωσέ τα
            - Απάντησε στα ελληνικά με ακρίβεια και επαγγελματισμό

            ΠΟΛΥΠΗΓΕΣ ΝΟΜΙΚΑ ΚΕΙΜΕΝΑ:
            {context}

            ΕΡΩΤΗΣΗ ΧΡΗΣΤΗ:
            {query}

            ΥΠΑΡΧΟΥΣΑ ΑΠΑΝΤΗΣΗ ΑΠΟ AI ΣΥΣΤΗΜΑ:
            {local_response}

            ΑΝΑΛΥΣΗ & ΔΙΟΡΘΩΜΕΝΗ/ΒΕΛΤΙΩΜΕΝΗ ΑΠΑΝΤΗΣΗ:"""
                    
        return self.remote_llm.generate(review_prompt, max_tokens=3072, method=method)
    
    def enhanced_chat(self, 
                     query: str, 
                     config: QueryConfig = None,
                     show_context: bool = False, 
                     show_debug: bool = False, 
                     use_remote: bool = False, 
                     use_review_mode: bool = False,
                     remote_method: str = None) -> Dict[str, Any]:
        """
        Enhanced chat method with options for remote LLM and review mode
        """
        
        if config is None:
            config = self.default_config
        
        try:
            # Multi-source query
            print(f"Multi-source retrieval...")
            query_result = self.query_system.multi_source_query(query, config)
            
            if not query_result['success']:
                return {
                    "query": query,
                    "response": "Συγγνώμη, υπήρξε πρόβλημα με την ανάκτηση των κειμένων.",
                    "error": "Query system failure",
                    "success": False
                }
            
            # Cache for remote retry
            self.last_query = query
            self.last_query_result = query_result
            
            # Create context
            context = self.create_context_from_results(query_result)
            self.last_context = context
            
            if show_context:
                print("\n" + "="*80)
                print("MULTI-SOURCE CONTEXT:")
                print("="*80)
                print(context)
                print("="*80)
            
            if show_debug:
                self.query_system.print_results_summary(query_result, show_details=True)
            
            response = ""
            response_source = ""
            review_info = {}
            
            if use_remote and not use_review_mode:
                # Use remote LLM directly
                remote_result = self.generate_remote_response(query, context, method=remote_method)
                
                if remote_result["success"]:
                    response = remote_result["response"]
                    response_source = f"remote ({remote_result.get('model', 'unknown')})"
                else:
                    # Fallback to local LLM
                    print(f"Remote LLM failed: {remote_result.get('error', 'Unknown error')}")
                    print("Using local LLM as fallback...")
                    prompt = self.create_advanced_prompt(query, context, query_result)
                    response = self.generate_local_response(prompt)
                    response_source = f"local fallback ({self.llm_model_id})"
                    
            elif use_review_mode:
                # Review Mode: Local LLM -> Remote Review
                print("Review Mode: Local LLM -> Remote Review...")
                
                # 1. Generate local response
                prompt = self.create_advanced_prompt(query, context, query_result)
                local_response = self.generate_local_response(prompt)
                print(f"Local answer: {len(local_response)} characters")
                
                # Cache for retry
                self.last_local_response = local_response
                
                # 2. send to remote for review
                print("Send to remote LLM for review...")
                review_result = self.generate_remote_review_response(
                    query, context, local_response, method=remote_method
                )
                
                if review_result["success"]:
                    response = review_result["response"]
                    response_source = f"remote review ({review_result.get('model', 'unknown')})"
                    review_info = {
                        "local_response": local_response,
                        "local_response_length": len(local_response),
                        "review_successful": True,
                        "review_model": review_result.get('model', 'unknown'),
                        "review_method": review_result.get('method', remote_method)
                    }
                    print(f"Review completed by {review_result.get('model', 'remote LLM')}")
                else:
                    # Fallback to local response
                    print(f"Remote review failed: {review_result.get('error', 'Unknown error')}")
                    print("Using local response as final answer...")
                    response = local_response
                    response_source = f"local (review fallback - {self.llm_model_id})"
                    review_info = {
                        "local_response": local_response,
                        "local_response_length": len(local_response),
                        "review_successful": False,
                        "review_error": review_result.get('error', 'Unknown error')
                    }
                    
            else:
                # Use local LLM
                prompt = self.create_advanced_prompt(query, context, query_result)
                response = self.generate_local_response(prompt)
                response_source = f"local ({self.llm_model_id})"
            
            # Cache for remote retry
            if not use_review_mode:
                self.last_local_response = response if response_source.startswith("local") else None
            
            # cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            result = {
                "query": query,
                "response": response,
                "response_source": response_source,
                "query_result": query_result,
                "retrieval_info": {
                    "total_sources": query_result['statistics']['total_sources_searched'],
                    "results_per_source": query_result['statistics']['results_per_source'],
                    "final_results_count": query_result['statistics']['final_results_count'],
                    "source_distribution": query_result['statistics'].get('source_distribution', {}),
                    "score_range": {
                        "min": query_result['statistics']['score_distribution']['min'],
                        "max": query_result['statistics']['score_distribution']['max']
                    } if query_result['statistics'].get('score_distribution') else {}
                },
                "config_used": query_result['config'],
                "success": True
            }
            
            # add review info if exists
            if review_info:
                result["review_info"] = review_info
                
            return result
            
        except Exception as e:
            print(f"Error: {e}")
            return {
                "query": query,
                "response": f"Συγγνώμη, παρουσιάστηκε σφάλμα: {str(e)}",
                "error": str(e),
                "success": False
            }
    
    def retry_with_remote_llm(self, method: str = None, use_review_mode: bool = False) -> Dict[str, Any]:
        """Retry last query with remote LLM or review mode"""
        if not self.last_query or not self.last_context:
            return {
                "success": False,
                "error": "The last query or context is missing."
            }
        
        if use_review_mode and not self.last_local_response:
            return {
                "success": False,
                "error": "The last local response is missing for review mode."
            }
        
        if use_review_mode:
            print(f"Retry with review mode (local -> remote review)...")
            remote_result = self.generate_remote_review_response(
                self.last_query, self.last_context, self.last_local_response, method=method
            )
            
            if remote_result["success"]:
                print(f"Successful review by remote LLM")
                
                # Show comparison
                print(f"\nLocal Answer:")
                print("=" * 80)
                print(self.last_local_response)
                print("=" * 80)
                print(f"Length: {len(self.last_local_response)} characters")
                
                print(f"\nEnhanced Answer after Review (Remote LLM):")
                print("=" * 80)
                print(remote_result["response"])
                print("=" * 80)
                print(f"Length: {len(remote_result['response'])} characters")
                print(f"Review από: {remote_result.get('model', 'unknown')}")
                
                result = {
                    "success": True,
                    "query": self.last_query,
                    "response": remote_result["response"],
                    "response_source": f"remote review ({remote_result.get('model', 'unknown')})",
                    "method": remote_result.get("method", method),
                    "query_result": self.last_query_result,
                    "review_info": {
                        "local_response": self.last_local_response,
                        "local_response_length": len(self.last_local_response),
                        "review_successful": True,
                        "review_model": remote_result.get('model', 'unknown'),
                        "review_method": remote_result.get('method', method)
                    }
                }
                
                return result
        
        else:
            print(f"Retry with remote LLM...")
            remote_result = self.generate_remote_response(self.last_query, self.last_context, method=method)
            
            if remote_result["success"]:
                print(f"Successful answer from remote LLM")
                
                result = {
                    "success": True,
                    "query": self.last_query,
                    "response": remote_result["response"],
                    "response_source": f"remote ({remote_result.get('model', 'unknown')})",
                    "method": remote_result.get("method", method),
                    "query_result": self.last_query_result
                }
                
                return result
        
        print(f"Remote LLM failed: {remote_result.get('error', 'Unknown error')}")
        return {
            "success": False,
            "error": remote_result.get("error", "Unknown error"),
            "fallback_response": self.last_local_response
        }
    
    def update_config(self, qa_weight: float = None, notes_weight: float = None, laws_weight: float = None, max_results: int = None) -> QueryConfig:
        """Update default configuration weights"""
        
        if qa_weight is not None:
            self.default_config.qa_weight = qa_weight
        if notes_weight is not None:
            self.default_config.notes_weight = notes_weight
        if laws_weight is not None:
            self.default_config.laws_weight = laws_weight
        if max_results is not None:
            self.default_config.final_max_results = max_results
        
        # Normalize weights
        total_weight = self.default_config.qa_weight + self.default_config.notes_weight + self.default_config.laws_weight
        if abs(total_weight - 1.0) > 0.01:
            self.default_config.qa_weight /= total_weight
            self.default_config.notes_weight /= total_weight
            self.default_config.laws_weight /= total_weight
        
        print(f"Updated config: QA={self.default_config.qa_weight:.3f}, "
              f"Notes={self.default_config.notes_weight:.3f}, "
              f"Laws={self.default_config.laws_weight:.3f}")
        
        return self.default_config
    
    def interactive_chat(self):
        """Interactive chat loop with commands"""
        print("\nEnhanced Multi-Source RAG Chatbot with LLM Review!")
        print("Multi-Source Query System (Q&A + Notes + Laws) + LLM Review")
        print("\Comands:")
        print("  'exit' = exit the chat")
        print("  'context' = show/hide context")
        print("  'debug' = show/hide debug info")
        print("  'remote' = use remote LLM for next question")
        print("  'review' = use review mode for next question")
        print("  'retry' = retry last question with remote LLM")
        print("  'retry-review' = rerty with review mode")
        print("  'test' = test remote LLM connection")
        print("  'config' = show current config")
        print("  'weights X Y Z' = adjust weights (e.g. 'weights 0.6 0.3 0.1')")
        print("  'info' = show last query info")
        print("  'help' = show commands")
        print("-" * 80)
        
        show_context = False
        show_debug = False
        use_remote_next = False
        use_review_mode_next = False
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['exit', 'έξοδος', 'quit']:
                    print("Exiting chat. Goodbye!")
                    break
                
                if query.lower() == 'context':
                    show_context = not show_context
                    status = "Activated" if show_context else "Deactivated"
                    print(f"Show status: {status}")
                    continue
                
                if query.lower() == 'debug':
                    show_debug = not show_debug
                    status = "Activated" if show_debug else "Deactivated"
                    print(f"Debug mode {status}")
                    continue
                
                if query.lower() == 'remote':
                    use_remote_next = not use_remote_next
                    if use_remote_next:
                        use_review_mode_next = False  # Reset review mode
                    status = "Activated" if use_remote_next else "Deactivated"
                    print(f"Remote LLM for next quesion {status}")
                    continue
                
                if query.lower() == 'review':
                    use_review_mode_next = not use_review_mode_next
                    if use_review_mode_next:
                        use_remote_next = False  # Reset remote mode
                    status = "Activated" if use_review_mode_next else "Deactivated"
                    print(f"Review mode for next quesion {status}")
                    if use_review_mode_next:
                        print("   Mode: Local LLM → Remote Review & Correction")
                    continue
                
                if query.lower() == 'retry':
                    retry_result = self.retry_with_remote_llm()
                    if retry_result["success"]:
                        print(f"\nResponse from Remote LLM:")
                        print(retry_result["response"])
                        print(f"\nSource: {retry_result['response_source']}")
                        
                        # Εμφάνιση query statistics
                        if retry_result.get("query_result"):
                            stats = retry_result["query_result"]["statistics"]
                            print(f"Query Stats: {stats['source_distribution']}")
                    else:
                        print(f"\Retry failed: {retry_result['error']}")
                        if retry_result.get("fallback_response"):
                            print("Previous local response:")
                            print(retry_result["fallback_response"])
                    continue
                
                if query.lower() == 'retry-review':
                    retry_result = self.retry_with_remote_llm(use_review_mode=True)
                    if retry_result["success"]:
                        print(f"\nResponse after Review by Remote LLM:")
                        print(retry_result["response"])
                        print(f"\nSource: {retry_result['response_source']}")
                        
                        # Εμφάνιση review info
                        if retry_result.get("review_info"):
                            review_info = retry_result["review_info"]
                            print(f"\nReview Info:")
                            print(f"   • Original local response: {review_info['local_response_length']} characters")
                            print(f"   • Review by: {review_info['review_model']}")
                            
                            # Προαιρετικά εμφάνιση της αρχικής απάντησης
                            show_orig = input("\nShow previous local answer (y/n): ").strip().lower()
                            if show_orig in ['y', 'yes', 'ναι', 'ν']:
                                print(f"\nOriginal Local Answer:")
                                print("-" * 60)
                                print(review_info["local_response"])
                                print("-" * 60)
                        
                        # Εμφάνιση query statistics
                        if retry_result.get("query_result"):
                            stats = retry_result["query_result"]["statistics"]
                            print(f"Query Stats: {stats['source_distribution']}")
                    else:
                        print(f"\Review retry failed: {retry_result['error']}")
                        if retry_result.get("fallback_response"):
                            print("Previous local response:")
                            print(retry_result["fallback_response"])
                    continue
                
                if query.lower() == 'test':
                    print("\nTesting remote LLM connections...")
                    test_results = self.remote_llm.test_connection()
                    print("\nTest Results:")
                    for method, result in test_results.items():
                        status = "✅" if result["success"] else "❌"
                        model_info = f" ({result['model']})" if result['model'] else ""
                        error_info = f" - {result['error']}" if result['error'] else ""
                        print(f"  {status} {method}{model_info}{error_info}")
                    continue
                
                if query.lower() == 'config':
                    config = self.default_config
                    print(f"\nCurrent Configuration:")
                    print(f"   • QA Weight: {config.qa_weight:.3f}")
                    print(f"   • Notes Weight: {config.notes_weight:.3f}")
                    print(f"   • Laws Weight: {config.laws_weight:.3f}")
                    print(f"   • Max results per source: {config.max_results_per_source}")
                    print(f"   • Final max results: {config.final_max_results}")
                    print(f"   • Show context: {show_context}")
                    print(f"   • Show debug: {show_debug}")
                    print(f"   • Use remote next: {use_remote_next}")
                    print(f"   • Use review mode next: {use_review_mode_next}")
                    continue
                
                if query.lower().startswith('weights'):
                    try:
                        parts = query.split()
                        if len(parts) == 4:
                            qa_w, notes_w, laws_w = map(float, parts[1:4])
                            self.update_config(qa_weight=qa_w, notes_weight=notes_w, laws_weight=laws_w)
                        else:
                            print(" Usage: weights <qa> <notes> <laws> (π.χ. weights 0.6 0.3 0.1)")
                    except ValueError:
                        print(" Invalid weights format. Use numbers between 0 and 1.")
                    continue
                
                if query.lower() == 'info':
                    print("\n🔧 System Information:")
                    
                    # Query system info
                    system_info = {}
                    for source_type, source_data in self.query_system.sources.items():
                        system_info[source_type] = {
                            'documents': len(source_data['docs']),
                            'has_tfidf': source_data['tfidf_vectorizer'] is not None,
                            'has_indices': len(source_data['indices']) > 0
                        }
                    
                    for source_type, info in system_info.items():
                        print(f"  {source_type.upper()}:")
                        print(f"      • Documents: {info['documents']}")
                        print(f"      • TF-IDF: {'✅' if info['has_tfidf'] else '❌'}")
                        print(f"      • Indices: {'✅' if info['has_indices'] else '❌'}")
                    
                    print(f"    Local LLM: {self.llm_model_id}")
                    print(f"    Remote LLM: {self.remote_llm.current_method}")
                    print(f"    Device: {self.device}")
                    print(f"    Review Mode Available: ✅")
                    continue
                
                if query.lower() == 'help':
                    print("\n Available commands:")
                    print("  'exit' - exit the chat")
                    print("  'context' - toggle context display")
                    print("  'debug' - toggle debug info")
                    print("  'remote' - toggle remote LLM for next question")
                    print("  'review' - toggle review mode (local → remote review)")
                    print("  'retry' - retry last question with remote LLM")
                    print("  'retry-review' - retry last question with review mode")
                    print("  'test' - trest remote LLM connection")
                    print("  'config' - show current config")
                    print("  'weights X Y Z' - adjust weights (e.g. 'weights 0.6 0.3 0.1')")
                    print("  'info' - show last query info")
                    continue
                
                if not query:
                    print("Please enter a valid query or command.")
                    continue
                
                # Main query processing
                result = self.enhanced_chat(
                    query, 
                    show_context=show_context, 
                    show_debug=show_debug,
                    use_remote=use_remote_next,
                    use_review_mode=use_review_mode_next
                )
                
                # Reset flags
                if use_remote_next:
                    use_remote_next = False
                if use_review_mode_next:
                    use_review_mode_next = False
                
                if result["success"]:
                    if result.get("review_info") and result["review_info"].get("local_response"):
                        print(f"\nLocal Answer (before Review):")
                        print("=" * 80)
                        print(result["review_info"]["local_response"])
                        print("=" * 80)
                        print(f"Μήκος: {result['review_info'].get('local_response_length', 0)} characters")
                        print(f"Μοντέλο: {self.llm_model_id}")
                        
                        # If review was successful, show the improved answer
                        if result["review_info"].get("review_successful"):
                            print(f"\nEnhanced Answer after Review (Remote LLM):")
                            print("=" * 80)
                            print(result["response"])
                            print("=" * 80)
                            print(f"Μήκος: {len(result['response'])} characters")
                            print(f"Review από: {result['review_info'].get('review_model', 'unknown')}")
                        else:
                            # If review failed, show the local answer as fallback
                            print(f"\nEnhanced Answer (Review Failed, using Local):")
                            print(f"Error review: {result['review_info'].get('review_error', 'Unknown')}")
                    
                    else:
                        # Answer without review
                        print(f"\nModel Answer:")
                        print(result["response"])
                    
                    # Source of the response
                    print(f"\nSource: {result.get('response_source', 'unknown')}")
                    
                    # show review statistics if available
                    if result.get("review_info"):
                        review_info = result["review_info"]
                        print(f"\nReview Statistics:")
                        print(f"   • Review successful: {'✅' if review_info.get('review_successful') else '❌'}")
                        
                        if review_info.get('review_successful'):
                            local_len = review_info.get('local_response_length', 0)
                            final_len = len(result['response'])
                            
                            if final_len > local_len:
                                diff = final_len - local_len
                                print(f"   • Επέκταση: +{diff} χαρακτήρες ({((diff/local_len)*100):.1f}% αύξηση)")
                            elif final_len < local_len:
                                diff = local_len - final_len
                                print(f"   • Συμπίεση: -{diff} χαρακτήρες ({((diff/local_len)*100):.1f}% μείωση)")
                            else:
                                print(f"   • Μήκος: Παραμένει ίδιο")
                            
                            print(f"   • Review method: {review_info.get('review_method', 'unknown')}")
                        
                        
                    # Query statistics
                    info = result['retrieval_info']
                    print(f"\nMulti-Source RAG Statistics:")
                    print(f"   • Sources used: {info['total_sources']}")
                    
                    results_per_source = info['results_per_source']
                    print(f"   • Results per source:")
                    for source, count in results_per_source.items():
                        print(f"     - {source.upper()}: {count}")
                    
                    if info.get('source_distribution'):
                        print(f"   • Final distribution:")
                        for source, count in info['source_distribution'].items():
                            percentage = (count / info['final_results_count']) * 100
                            print(f"     - {source.upper()}: {count} ({percentage:.1f}%)")
                    
                    # Configuration used
                    config_used = result['config_used']
                    print(f"   • Weights used: QA={config_used['qa_weight']:.2f}, "
                          f"Notes={config_used['notes_weight']:.2f}, "
                          f"Laws={config_used['laws_weight']:.2f}")
                    
                    if info.get('score_range'):
                        score_range = info['score_range']
                        print(f"   • Score range: {score_range['min']:.3f} - {score_range['max']:.3f}")
                    
                    # Hints for improvement
                    response_source = result.get('response_source', '')
                    if response_source.startswith('local') and not result.get("review_info"):
                        print(f"\ndid like the answer?")
                        print("   • write 'retry' for remote LLM answer")
                        print("   • write 'retry-review' for review mode")
                    elif response_source.startswith('remote') and 'review' not in response_source:
                        print(f"\nwrite 'retry-review' for review local answer ")
                        
                    
                    # Quality assessment
                    final_results_count = info['final_results_count']
                    if final_results_count > 0:
                        if info.get('score_range') and info['score_range'].get('max', 0) > 0.8:
                            quality = ":Quality: Excellent"
                        elif info.get('score_range') and info['score_range'].get('max', 0) > 0.6:
                            quality = "Quality: Good" 
                        elif info.get('score_range') and info['score_range'].get('max', 0) > 0.4:
                            quality = "Quality: Fair"
                        else:
                            quality = "Quality: Poor"
                        
                        print(f"   • Query Quality: {quality}")
                        
                        # Source diversity
                        sources_used = len(info.get('source_distribution', {}))
                        print(f"   • Source Diversity: {sources_used}/3 Sources used")
                
                else:
                    print(f"\n❌ {result['response']}")
                
            except KeyboardInterrupt:
                print("\nExiting chat. Goodbye!")
                break
            except Exception as e:
                print(f"\nUnpreccedented error {e}")

    def batch_query(self, queries: List[str], config: QueryConfig = None, use_review_mode: bool = False) -> List[Dict[str, Any]]:
        """Batch processing of multiple queries"""
        if config is None:
            config = self.default_config
            
        results = []
        mode_text = "review mode" if use_review_mode else "standard mode"
        print(f"Batch processing {len(queries)} queries ({mode_text})...")
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Processing: {query[:50]}...")
            try:
                result = self.enhanced_chat(query, config=config, use_review_mode=use_review_mode)
                results.append(result)
                
                if result['success']:
                    print(f"✅ Completed ({result.get('response_source', 'unknown')})")
                    if result.get('review_info') and result['review_info'].get('review_successful'):
                        print(f"   Review by {result['review_info'].get('review_model', 'unknown')}")
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'response': f"Σφάλμα επεξεργασίας: {str(e)}"
                })
        
        # Batch statistics
        successful = sum(1 for r in results if r.get('success', False))
        reviewed = sum(1 for r in results if r.get('review_info', {}).get('review_successful', False))
        
        print(f"\Batch Results:")
        print(f"   • Total queries: {len(queries)}")
        print(f"   • Successful: {successful}")
        print(f"   • Failed: {len(queries) - successful}")
        if use_review_mode:
            print(f"   • Successfully reviewed: {reviewed}")
        
        return results

def main():
    """Κύρια συνάρτηση"""
    HF_TOKEN = " "  # use your HuggingFace token here
    GOOGLE_AI_KEY = ""  # use your Google AI key here 
    
    # Load from environment variables if available
    GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY", GOOGLE_AI_KEY)
    
    try:
        chatbot = EnhancedMultiSourceChatbot(
            qa_faiss_path="./faiss-files/faiss_qa_hybrid",
            notes_faiss_path="./faiss-files/faiss_notes_hybrid",
            laws_faiss_path="./faiss-files/faiss_laws_hybrid",
            embedding_model="intfloat/multilingual-e5-large",
            llm_model_id="./models/gemma-greek-4b-legal-light-epoch7-lr4e-5-bs1-gas8",
            hf_token=HF_TOKEN,
            google_ai_key=GOOGLE_AI_KEY
        )
        
        if GOOGLE_AI_KEY:
            print("Google AI API key provided - Google models enabled")
        else:
            print("No Google AI API key - Google models disabled")
        
        # Show system overview
        print(f"\nSystem Overview:")
        for source_type, source_data in chatbot.query_system.sources.items():
            print(f"   📚 {source_type.upper()}: {len(source_data['docs'])} documents")
        
        print(f"   Local LLM: {chatbot.llm_model_id}")
        print(f"   Remote LLM: {chatbot.remote_llm.current_method}")
        print(f"   Review Mode: Available")
        
        # show default configuration
        config = chatbot.default_config
        print(f"\nDefault Weights:")
        print(f"   • QA: {config.qa_weight:.2f} (Ερωτήσεις-Απαντήσεις)")
        print(f"   • Notes: {config.notes_weight:.2f} (Σημειώσεις)")
        print(f"   • Laws: {config.laws_weight:.2f} (Νομοθεσία)")
        
        print(f"\nReview Mode Features:")
        print("   • Local LLM → Remote Review & Correction")
        print("   • Comprehensive error detection and improvement")
        print("   • Context-aware legal advice enhancement")
        
        chatbot.interactive_chat()
        
    except Exception as e:
        print(f"Error : {e}")
        print("\nPlease check the following:")
        print(" You have the required packages installed (see requirements.txt)")
        print("-FAISS files exist:")
        print("  * faiss_qa_hybrid")
        print("  * faiss_notes_hybrid") 
        print("  * faiss_laws_hybrid")
        print("- Valid API keys if using remote LLMs:")

def demo():
    """Demo function for testing με review mode"""
    print("DEMO MODE - Enhanced Multi-Source Chatbot with Review")
    
    # Test queries
    test_queries = [
        "Τι άδεια δικαιούται μια έγκυος εργαζόμενη;",
        "Πώς υπολογίζεται η αποζημίωση απόλυσης;", 
        "Ποιες είναι οι υποχρεώσεις του εργοδότη για την ασφάλεια;",
        "Τι προστασία έχουν οι εργαζόμενοι από παρενόχληση;",
        "Πώς γίνεται η καταγγελία σύμβασης εργασίας;"
    ]
    
    try:
        chatbot = EnhancedMultiSourceChatbot(
            qa_faiss_path="./faiss-files/faiss_qa_hybrid",
            notes_faiss_path="./faiss-files/faiss_notes_hybrid",
            laws_faiss_path="./faiss-files/faiss_laws_hybrid",
        )
        
        print(f"\nTesting {len(test_queries)} queries with different modes...")
        
        # Test different modes
        modes = [
            {"name": "Local Only", "use_remote": False, "use_review": False},
            {"name": "Remote Direct", "use_remote": True, "use_review": False},
            {"name": "Review Mode", "use_remote": False, "use_review": True}
        ]
        
        for i, query in enumerate(test_queries[:2]):  # Test first 2 queries
            print(f"\n{'='*80}")
            print(f"TEST {i+1}: {query}")
            print(f"{'='*80}")
            
            for mode in modes:
                print(f"\n--- {mode['name']} ---")
                
                try:
                    result = chatbot.enhanced_chat(
                        query, 
                        show_debug=False,
                        use_remote=mode['use_remote'],
                        use_review_mode=mode['use_review']
                    )
                    
                    if result['success']:
                        print(f"Response generated")
                        print(f"   Source: {result.get('response_source', 'unknown')}")
                        
                        info = result['retrieval_info']
                        print(f"   RAG Sources: {info['source_distribution']}")
                        print(f"   Results: {info['final_results_count']}")
                        
                        if info.get('score_range'):
                            print(f"   Score range: {info['score_range']['min']:.3f}-{info['score_range']['max']:.3f}")
                        
                        # Review info
                        if result.get('review_info'):
                            review_info = result['review_info']
                            if review_info.get('review_successful'):
                                print(f"   Review: ✅ by {review_info.get('review_model', 'unknown')}")
                                print(f"   Original length: {review_info.get('local_response_length', 0)} chars")
                            else:
                                print(f"   Review: ❌ {review_info.get('review_error', 'Failed')}")
                        
                        print(f"   Response preview: {result['response'][:150]}...")
                        
                    else:
                        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"❌ Exception: {e}")
                
                print("-" * 50)
                
        # Test batch processing with review mode
        print(f"\nTesting batch processing with review mode...")
        batch_results = chatbot.batch_query(test_queries[:3], use_review_mode=True)
        
        print(f"\nBatch Results Summary:")
        for i, result in enumerate(batch_results, 1):
            status = "✅" if result.get('success', False) else "❌"
            source = result.get('response_source', 'unknown')
            review_status = ""
            if result.get('review_info', {}).get('review_successful'):
                review_status = " [REVIEWED]"
            print(f"   {status} Query {i}: {source}{review_status}")
                
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()