#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web API for Enhanced Multi-Source RAG Chatbot
This server provides endpoints to interact with a multi-source retrieval-augmented generation (RAG) chatbot
that uses a local LLM and multiple document sources.
"""

import os
import json
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import threading
import time
import sys

# Import new multi-source chatbot class
from chatbot_multi_source_llm_judge import EnhancedMultiSourceChatbot
from query import QueryConfig

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chatbot_server.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../templates')
CORS(app)

# Global chatbot instance / status
chatbot = None
chatbot_status = {
    "loaded": False,
    "loading": False,
    "error": None,
    "last_updated": None,
    "initialization_progress": ""
}

# Session storage for chat history
chat_sessions = {}

def initialize_chatbot():
    """ Initializes the Enhanced Multi-Source Chatbot in a background thread."""
    global chatbot, chatbot_status
    
    try:
        chatbot_status["loading"] = True
        chatbot_status["initialization_progress"] = "Starting initialization..."
        logger.info("Initializing Enhanced Multi-Source Chatbot...")
        
        # Configuration
        HF_TOKEN = ""
        GOOGLE_AI_KEY = ""
        
        # Environment variables override
        HF_TOKEN = os.getenv("HF_TOKEN", HF_TOKEN)
        GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY", GOOGLE_AI_KEY)
        
        chatbot_status["initialization_progress"] = "Loading models and indices..."
        
        chatbot = EnhancedMultiSourceChatbot(
            qa_faiss_path="./faiss-files/faiss_qa_hybrid",
            notes_faiss_path="./faiss-files/faiss_notes_hybrid",
            laws_faiss_path="./faiss-files/faiss_laws_hybrid",
            embedding_model="intfloat/multilingual-e5-large",
            llm_model_id="./models/gemma-greek-4b-legal-light-epoch7-lr4e-5-bs1-gas8",
            hf_token=HF_TOKEN,
            google_ai_key=GOOGLE_AI_KEY
        )
        
        chatbot_status["loaded"] = True
        chatbot_status["loading"] = False
        chatbot_status["error"] = None
        chatbot_status["last_updated"] = datetime.now().isoformat()
        chatbot_status["initialization_progress"] = "Multi-source system initialized successfully"
        
        logger.info("Enhanced Multi-Source Chatbot initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initialinzg multi-source chatbot: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        chatbot_status["loaded"] = False
        chatbot_status["loading"] = False
        chatbot_status["error"] = str(e)
        chatbot_status["last_updated"] = datetime.now().isoformat()
        chatbot_status["initialization_progress"] = f"Error: {str(e)}"

# Start chatbot initialization in a background thread
chatbot_thread = threading.Thread(target=initialize_chatbot, daemon=True)
chatbot_thread.start()

@app.route('/')
def index():
    """Serve το HTML UI"""
    return render_template('multi_source_ui.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns the current status of the multi-source chatbot system"""
    status_info = {
        "status": "ready" if chatbot_status["loaded"] else ("loading" if chatbot_status["loading"] else "error"),
        "chatbot_loaded": chatbot_status["loaded"],
        "loading": chatbot_status["loading"],
        "error": chatbot_status["error"],
        "last_updated": chatbot_status["last_updated"],
        "initialization_progress": chatbot_status["initialization_progress"],
        "timestamp": datetime.now().isoformat(),
        "system_type": "multi_source_rag"
    }
    
    # If loaded, add more detailed info
    if chatbot_status["loaded"] and chatbot:
        try:
            # Multi-source system info
            system_info = {}
            for source_type, source_data in chatbot.query_system.sources.items():
                system_info[source_type] = {
                    'documents': len(source_data['docs']),
                    'has_tfidf': source_data['tfidf_vectorizer'] is not None,
                    'has_indices': len(source_data['indices']) > 0,
                    'status': 'ready'
                }
            
            status_info["multi_source_info"] = {
                "sources": system_info,
                "total_sources": len(system_info),
                "embedding_model": chatbot.query_system.embedding_model_name,
                "default_config": {
                    "qa_weight": chatbot.default_config.qa_weight,
                    "notes_weight": chatbot.default_config.notes_weight,
                    "laws_weight": chatbot.default_config.laws_weight,
                    "max_results": chatbot.default_config.final_max_results
                }
            }
            
            status_info["model_info"] = {
                "llm_model": chatbot.llm_model_id,
                "embedding_model": chatbot.query_system.embedding_model_name,
                "device": str(chatbot.device),
                "remote_llm_available": True,
                "remote_method": chatbot.remote_llm.current_method
            }
            
        except Exception as e:
            logger.warning(f"Could not get multi-source info: {e}")
    
    return jsonify(status_info)

@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update the multi-source query configuration"""
    if not chatbot_status["loaded"] or chatbot is None:
        return jsonify({
            "success": False,
            "error": "Chatbot not loaded",
        }), 500
    
    if request.method == 'GET':
        # Retrieve current configuration
        config = chatbot.default_config
        return jsonify({
            "success": True,
            "config": {
                "qa_weight": config.qa_weight,
                "notes_weight": config.notes_weight,
                "laws_weight": config.laws_weight,
                "max_results_per_source": config.max_results_per_source,
                "final_max_results": config.final_max_results
            },
            "timestamp": datetime.now().isoformat()
        })
    
    elif request.method == 'POST':
        # Updated configuration
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    "success": False,
                    "error": "Παρακαλώ παρέχετε configuration data"
                }), 400
            
            # Update weights
            updated_config = chatbot.update_config(
                qa_weight=data.get('qa_weight'),
                notes_weight=data.get('notes_weight'),
                laws_weight=data.get('laws_weight'),
                max_results=data.get('max_results')
            )
            
            logger.info(f"Configuration updated: QA={updated_config.qa_weight:.3f}, "
                       f"Notes={updated_config.notes_weight:.3f}, "
                       f"Laws={updated_config.laws_weight:.3f}")
            
            return jsonify({
                "success": True,
                "message": "Configuration updated successfully",
                "config": {
                    "qa_weight": updated_config.qa_weight,
                    "notes_weight": updated_config.notes_weight,
                    "laws_weight": updated_config.laws_weight,
                    "max_results_per_source": updated_config.max_results_per_source,
                    "final_max_results": updated_config.final_max_results
                },
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return jsonify({
                "success": False,
                "error": f"Error updating config: {str(e)}"
            }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint for multi-source chat with local LLM"""
    try:
        # Check system status
        if chatbot_status["loading"]:
            return jsonify({
                "success": False,
                "error": "System is still loading, please wait...",
                "status": "loading",
                "progress": chatbot_status["initialization_progress"]
            }), 503
        
        if not chatbot_status["loaded"] or chatbot is None:
            return jsonify({
                "success": False,
                "error": f"Σφάλμα στην αρχικοποίηση του chatbot: {chatbot_status['error']}",
                "status": "error"
            }), 500
        
        # Parse request data
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Please provide a query"
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "success": False,
                "error": "The query cannot be empty"
            }), 400
        
        # Settings from UI
        show_context = data.get('show_context', False)
        show_debug = data.get('show_debug', False)
        session_id = data.get('session_id', 'default')
        
        # Custom configuration from UI
        custom_config = None
        if 'config' in data:
            config_data = data['config']
            custom_config = QueryConfig(
                qa_weight=config_data.get('qa_weight', 0.5),
                notes_weight=config_data.get('notes_weight', 0.35),
                laws_weight=config_data.get('laws_weight', 0.15),
                max_results_per_source=config_data.get('max_results_per_source', 10),
                final_max_results=config_data.get('final_max_results', 20)
            )
        
        logger.info(f"Multi-source ερώτηση (session: {session_id}): {query[:100]}...")
        
        #Initialize session if not exists
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                'history': [],
                'created_at': datetime.now().isoformat()
            }
        
        #Call with local LLM
        start_time = time.time()
        result = chatbot.enhanced_chat(
            query=query,
            config=custom_config,
            show_context=show_context,
            show_debug=show_debug,
            use_remote=False  # Local LLM
        )
        processing_time = time.time() - start_time
        
        logger.info(f"Multi-source local response in {processing_time:.2f} seconds.")
        
        # Save to history
        chat_entry = {
            'query': query,
            'response': result.get("response", ""),
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(processing_time, 2),
            'source': 'local_multisource',
            'success': result["success"],
            'config_used': result.get("config_used", {})
        }
        
        chat_sessions[session_id]['history'].append(chat_entry)
        
        # Prepare response
        if result["success"]:
            response_data = {
                "success": True,
                "response": result["response"],
                "response_source": result.get("response_source", "local"),
                "session_id": session_id,
                "can_retry_remote": True,
                "multi_source_stats": {
                    "total_sources": result["retrieval_info"]["total_sources"],
                    "results_per_source": result["retrieval_info"]["results_per_source"],
                    "final_results_count": result["retrieval_info"]["final_results_count"],
                    "source_distribution": result["retrieval_info"].get("source_distribution", {}),
                    "score_range": result["retrieval_info"].get("score_range", {}),
                    "processing_time": round(processing_time, 2)
                },
                "config_used": result["config_used"],
                "query_result": {
                    "success": result["query_result"]["success"],
                    "statistics": result["query_result"]["statistics"]
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            response_data = {
                "success": False,
                "error": result.get("error", "Άγνωστο σφάλμα"),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in multi-source chat endpoint: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"Error server: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/chat/remote', methods=['POST'])
def chat_remote():
    """Endpoint for multi-source chat with remote LLM"""
    try:
        if not chatbot_status["loaded"] or chatbot is None:
            return jsonify({
                "success": False,
                "error": "Chatbot not available",
                "status": "error"
            }), 500
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Please provide a query"
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "success": False,
                "error": "Query cannot be empty"
            }), 400
        
        show_context = data.get('show_context', False)
        show_debug = data.get('show_debug', False)
        remote_method = data.get('remote_method', None)
        session_id = data.get('session_id', 'default')
        
        # Custom configuration from UI
        custom_config = None
        if 'config' in data:
            config_data = data['config']
            custom_config = QueryConfig(
                qa_weight=config_data.get('qa_weight', 0.5),
                notes_weight=config_data.get('notes_weight', 0.35),
                laws_weight=config_data.get('laws_weight', 0.15),
                max_results_per_source=config_data.get('max_results_per_source', 10),
                final_max_results=config_data.get('final_max_results', 20)
            )
        
        logger.info(f"Multi-source Remote LLM query (method: {remote_method}): {query[:100]}...")
        
        # Initialize session if not exists
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                'history': [],
                'created_at': datetime.now().isoformat()
            }
        
        # Call with remote LLM
        start_time = time.time()
        result = chatbot.enhanced_chat(
            query=query,
            config=custom_config,
            show_context=show_context,
            show_debug=show_debug,
            use_remote=True,
            remote_method=remote_method
        )
        processing_time = time.time() - start_time
        
        logger.info(f"Multi-source Remote response in {processing_time:.2f} seconds")
        
        # Savining in history
        chat_entry = {
            'query': query,
            'response': result.get("response", ""),
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(processing_time, 2),
            'source': 'remote_multisource',
            'remote_method': remote_method,
            'success': result["success"],
            'config_used': result.get("config_used", {})
        }
        
        chat_sessions[session_id]['history'].append(chat_entry)
        
        if result["success"]:
            response_data = {
                "success": True,
                "response": result["response"],
                "response_source": result.get("response_source", "remote"),
                "session_id": session_id,
                "remote_method": remote_method,
                "multi_source_stats": result.get("retrieval_info", {}),
                "config_used": result.get("config_used", {}),
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
        else:
            response_data = {
                "success": False,
                "error": result.get("error", "Άγνωστο σφάλμα"),
                "session_id": session_id,
                "remote_method": remote_method,
                "timestamp": datetime.now().isoformat()
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in multi-source remote chat endpoint: {e}")
        return jsonify({
            "success": False,
            "error": f"Error remote chat: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/retry', methods=['POST'])
def retry_last_query():
    """Retry last query with remote LLM (multi-source)"""
    try:
        if not chatbot_status["loaded"] or chatbot is None:
            return jsonify({
                "success": False,
                "error": "Chatbot not available"
            }), 500
        
        data = request.get_json() or {}
        remote_method = data.get('remote_method', None)
        session_id = data.get('session_id', 'default')
        
        logger.info(f"Multi-source Retry with remote LLM (method: {remote_method})...")
        
        start_time = time.time()
        retry_result = chatbot.retry_with_remote_llm(method=remote_method)
        processing_time = time.time() - start_time
        
        # Helper function to convert sets to lists for JSON serialization
        def convert_sets_to_lists(obj):
            if isinstance(obj, dict):
                return {k: convert_sets_to_lists(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets_to_lists(item) for item in obj]
            elif isinstance(obj, set):
                return list(obj)
            else:
                return obj
        
        # Convert any sets to lists in retry_result
        retry_result = convert_sets_to_lists(retry_result)
        
        # Update session history if applicable
        if session_id in chat_sessions and retry_result["success"]:
            retry_entry = {
                'query': retry_result.get("query", ""),
                'response': retry_result["response"],
                'timestamp': datetime.now().isoformat(),
                'processing_time': round(processing_time, 2),
                'source': 'remote_retry_multisource',
                'remote_method': retry_result.get("method", remote_method),
                'success': True
            }
            chat_sessions[session_id]['history'].append(retry_entry)
        
        if retry_result["success"]:
            logger.info(f"Multi-source successful retry in {processing_time:.2f} seconds.")
            response_data = {
                "success": True,
                "response": retry_result["response"],
                "response_source": retry_result.get("response_source", "remote"),
                "method": retry_result.get("method", remote_method),
                "processing_time": round(processing_time, 2),
                "session_id": session_id,
                "query_result": convert_sets_to_lists(retry_result.get("query_result")),
                "timestamp": datetime.now().isoformat()
            }
        else:
            logger.warning(f"Multi-source Retry failed: {retry_result.get('error', 'Unknown')}")
            response_data = {
                "success": False,
                "error": retry_result.get("error", "Retry απέτυχε"),
                "fallback_response": retry_result.get("fallback_response"),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in multi-source retry endpoint: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"Error retry: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/test-remote', methods=['POST'])
def test_remote_connections():
    """Test connections to remote LLM services (multi-source)"""
    try:
        if not chatbot_status["loaded"] or chatbot is None:
            return jsonify({
                "success": False,
                "error": "Chatbot not available"
            }), 500
        
        logger.info("Trying multi-source remote LLM connections...")
        
        start_time = time.time()
        test_results = chatbot.remote_llm.test_connection()
        test_time = time.time() - start_time
        
        # Format results for readability
        formatted_results = {}
        working_methods = []
        failed_methods = []
        
        for method, result in test_results.items():
            formatted_results[method] = {
                "success": result["success"],
                "error": result.get("error", ""),
                "model": result.get("model", ""),
                "status": "Success" if result["success"] else f"Error: {result.get('error', 'Unknown')}"
            }
            
            if result["success"]:
                working_methods.append(method)
            else:
                failed_methods.append(method)
        
        logger.info(f"Multi-source Test completed in {test_time:.2f}s - Working: {len(working_methods)}, Failed: {len(failed_methods)}")
        
        return jsonify({
            "success": True,
            "results": formatted_results,
            "summary": {
                "working_methods": working_methods,
                "failed_methods": failed_methods,
                "total_tested": len(test_results),
                "test_time": round(test_time, 2)
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in multi-source test-remote endpoint: {e}")
        return jsonify({
            "success": False,
            "error": f"Error test: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/batch', methods=['POST'])
def batch_query():
    """Batch processing for multiple queries with multi-source system"""
    try:
        if not chatbot_status["loaded"] or chatbot is None:
            return jsonify({
                "success": False,
                "error": "Chatbot not available"
            }), 500
        
        data = request.get_json()
        if not data or 'queries' not in data or not isinstance(data['queries'], list):
            return jsonify({
                "success": False,
                "error": "Please provide a list of queries"
            }), 400
        
        queries = [q.strip() for q in data['queries'] if q.strip()]
        if not queries:
            return jsonify({
                "success": False,
                "error": "No valid queries provided"
            }), 400
        
        # Configuration for batch
        custom_config = None
        if 'config' in data:
            config_data = data['config']
            custom_config = QueryConfig(
                qa_weight=config_data.get('qa_weight', 0.5),
                notes_weight=config_data.get('notes_weight', 0.35),
                laws_weight=config_data.get('laws_weight', 0.15),
                max_results_per_source=config_data.get('max_results_per_source', 10),
                final_max_results=config_data.get('final_max_results', 20)
            )
        
        logger.info(f"Multi-source Batch processing {len(queries)} queries...")
        
        start_time = time.time()
        results = chatbot.batch_query(queries, config=custom_config)
        total_time = time.time() - start_time
        
        #  Summary statistics
        successful_queries = len([r for r in results if r.get('success', False)])
        failed_queries = len(results) - successful_queries
        
        logger.info(f"Multi-source Batch completed in {total_time:.2f}s - Success: {successful_queries}, Failed: {failed_queries}")
        
        return jsonify({
            "success": True,
            "results": results,
            "summary": {
                "total_queries": len(queries),
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "total_time": round(total_time, 2),
                "average_time_per_query": round(total_time / len(queries), 2)
            },
            "config_used": custom_config.__dict__ if custom_config else chatbot.default_config.__dict__,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch endpoint: {e}")
        return jsonify({
            "success": False,
            "error": f"Error batch processing: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """Load chat history for a specific session"""
    if session_id not in chat_sessions:
        return jsonify({
            "success": False,
            "error": "Session not found"
        }), 404
    
    history = chat_sessions[session_id]['history']
    
    # Limit history (latest 50 entries)
    limited_history = history[-50:] if len(history) > 50 else history
    
    return jsonify({
        "success": True,
        "session_id": session_id,
        "history": limited_history,
        "total_entries": len(history),
        "returned_entries": len(limited_history),
        "session_created": chat_sessions[session_id]['created_at']
    })

@app.route('/api/history/<session_id>/clear', methods=['DELETE'])
def clear_chat_history(session_id):
    """Clear chat history for a specific session"""
    if session_id in chat_sessions:
        chat_sessions[session_id]['history'] = []
        return jsonify({
            "success": True,
            "message": f"History for session {session_id} cleared"
        })
    else:
        return jsonify({
            "success": False,
            "error": "Session not found"
        }), 404

@app.route('/api/stats', methods=['GET'])
def get_comprehensive_stats():
    """Comprehensive statistics about the multi-source system and usage"""
    if not chatbot_status["loaded"] or chatbot is None:
        return jsonify({
            "error": "Multi-source chatbot not loaded",
            "status": chatbot_status
        }), 503
    
    try:
        # Multi-source system statistics
        multisource_stats = {}
        total_documents = 0
        
        for source_type, source_data in chatbot.query_system.sources.items():
            source_stats = {
                "documents": len(source_data['docs']),
                "has_tfidf": source_data['tfidf_vectorizer'] is not None,
                "has_indices": len(source_data['indices']) > 0,
                "embedding_ready": True  # Assuming embeddings are ready if loaded
            }
            multisource_stats[source_type] = source_stats
            total_documents += source_stats["documents"]
        
        # Chatbot statistics
        chatbot_stats = {
            "total_documents": total_documents,
            "sources_available": len(multisource_stats),
            "embedding_model": chatbot.query_system.embedding_model_name,
            "llm_model": chatbot.llm_model_id,
            "device": str(chatbot.device),
            "default_config": {
                "qa_weight": chatbot.default_config.qa_weight,
                "notes_weight": chatbot.default_config.notes_weight,
                "laws_weight": chatbot.default_config.laws_weight,
                "max_results": chatbot.default_config.final_max_results
            }
        }
        
        # Session statistics
        session_stats = {
            "total_sessions": len(chat_sessions),
            "total_conversations": sum(len(session['history']) for session in chat_sessions.values()),
            "active_sessions": len([s for s in chat_sessions.keys() if chat_sessions[s]['history']]),
        }
        
        # Performance statistics
        perf_stats = {
            "average_response_time": 0,
            "local_responses": 0,
            "remote_responses": 0,
            "failed_responses": 0,
            "multisource_responses": 0
        }
        
        all_entries = []
        for session in chat_sessions.values():
            all_entries.extend(session['history'])
        
        if all_entries:
            successful_entries = [e for e in all_entries if e.get('success', False)]
            if successful_entries:
                perf_stats["average_response_time"] = round(
                    sum(e.get('processing_time', 0) for e in successful_entries) / len(successful_entries), 2
                )
            
            perf_stats["local_responses"] = len([e for e in all_entries if e.get('source', '').startswith('local')])
            perf_stats["remote_responses"] = len([e for e in all_entries if e.get('source', '').startswith('remote')])
            perf_stats["failed_responses"] = len([e for e in all_entries if not e.get('success', False)])
            perf_stats["multisource_responses"] = len([e for e in all_entries if 'multisource' in e.get('source', '')])
        
        return jsonify({
            "multisource_stats": multisource_stats,
            "chatbot_stats": chatbot_stats,
            "session_stats": session_stats,
            "performance_stats": perf_stats,
            "system_status": chatbot_status,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting multi-source stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sources', methods=['GET'])
def get_source_info():
    """Detailed information about each document source in the multi-source system"""
    if not chatbot_status["loaded"] or chatbot is None:
        return jsonify({
            "success": False,
            "error": "Chatbot not available"
        }), 500
    
    try:
        sources_info = {}
        
        for source_type, source_data in chatbot.query_system.sources.items():
            # Basic info
            docs = source_data['docs']
            
            # metadata analysis
            metadata_analysis = {}
            for doc in docs:
                metadata = doc.get('metadata', {})
                for key, value in metadata.items():
                    if key not in metadata_analysis:
                        metadata_analysis[key] = set()
                    metadata_analysis[key].add(str(value))
            
            # Convert sets to counts
            for key in metadata_analysis:
                metadata_analysis[key] = len(metadata_analysis[key])
            
            # Sample documents (first 3)
            sample_docs = []
            for i, doc in enumerate(docs[:3]):
                sample_docs.append({
                    "index": i,
                    "content_preview": doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                    "metadata": doc.get('metadata', {})
                })
            
            sources_info[source_type] = {
                "total_documents": len(docs),
                "search_methods": {
                    "tfidf_available": source_data['tfidf_vectorizer'] is not None,
                    "semantic_available": len(source_data['indices']) > 0,
                    "indices_count": len(source_data['indices'])
                },
                "metadata_fields": metadata_analysis,
                "sample_documents": sample_docs,
                "status": "ready"
            }
        
        return jsonify({
            "success": True,
            "sources": sources_info,
            "total_sources": len(sources_info),
            "embedding_model": chatbot.query_system.embedding_model_name,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting sources info: {e}")
        return jsonify({
            "success": False,
            "error": f"Error loading data sources: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Comprehensive health check για multi-source system"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_type": "multi_source_rag",
        "chatbot_status": chatbot_status,
        "services": {
            "multisource_chatbot": "ready" if chatbot_status["loaded"] else "not_ready",
            "remote_llm": "available" if chatbot and hasattr(chatbot, 'remote_llm') else "not_available",
            "query_system": "available" if chatbot and hasattr(chatbot, 'query_system') else "not_available",
            "api_server": "running"
        }
    }
    
    # Detailed health checks
    if chatbot_status["loaded"] and chatbot:
        try:
            # Check multi-source system health
            source_health = {}
            for source_type, source_data in chatbot.query_system.sources.items():
                source_health[source_type] = {
                    "documents_loaded": len(source_data['docs']) > 0,
                    "tfidf_ready": source_data['tfidf_vectorizer'] is not None,
                    "semantic_ready": len(source_data['indices']) > 0,
                    "status": "healthy" if len(source_data['docs']) > 0 else "unhealthy"
                }
            
            health_status["source_health"] = source_health
            
            # Overall health assessment
            unhealthy_sources = [s for s, info in source_health.items() if info["status"] == "unhealthy"]
            if unhealthy_sources:
                health_status["status"] = "degraded"
                health_status["issues"] = f"Unhealthy sources: {unhealthy_sources}"
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
    
    # Overall health determination
    if not chatbot_status["loaded"]:
        health_status["status"] = "unhealthy"
        return jsonify(health_status), 503
    
    return jsonify(health_status)




# Flask server (flask_server.py)

@app.route('/api/chat/review', methods=['POST'])
def chat_review_mode():
    """Endpoint for review mode: Local LLM -> Remote Review"""
    try:
        if not chatbot_status["loaded"] or chatbot is None:
            return jsonify({
                "success": False,
                "error": "Chatbot not available",
                "status": "error"
            }), 500
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Please provide a query"
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "success": False,
                "error": "The query cannot be empty"
            }), 400
        
        show_context = data.get('show_context', False)
        show_debug = data.get('show_debug', False)
        remote_method = data.get('remote_method', None)
        session_id = data.get('session_id', 'default')
        
        # Custom configuration
        custom_config = None
        if 'config' in data:
            config_data = data['config']
            custom_config = QueryConfig(
                qa_weight=config_data.get('qa_weight', 0.5),
                notes_weight=config_data.get('notes_weight', 0.35),
                laws_weight=config_data.get('laws_weight', 0.15),
                max_results_per_source=config_data.get('max_results_per_source', 10),
                final_max_results=config_data.get('final_max_results', 20)
            )
        
        logger.info(f"Review mode request (method: {remote_method}): {query[:100]}...")
        
        # Session initialization
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                'history': [],
                'created_at': datetime.now().isoformat()
            }
        
        # Call review mode
        start_time = time.time()
        result = chatbot.enhanced_chat(
            query=query,
            config=custom_config,
            show_context=show_context,
            show_debug=show_debug,
            use_remote=False,
            use_review_mode=True,
            remote_method=remote_method
        )
        processing_time = time.time() - start_time
        
        logger.info(f"Review mode completed in {processing_time:.2f} seconds")
        
        # Save to history
        chat_entry = {
            'query': query,
            'response': result.get("response", ""),
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(processing_time, 2),
            'source': 'review_mode',
            'remote_method': remote_method,
            'success': result["success"],
            'config_used': result.get("config_used", {})
        }
        
        if result.get("review_info"):
            chat_entry['review_info'] = result["review_info"]
        
        chat_sessions[session_id]['history'].append(chat_entry)
        
        if result["success"]:
            response_data = {
                "success": True,
                "response": result["response"],
                "response_source": result.get("response_source", "review"),
                "session_id": session_id,
                "remote_method": remote_method,
                "multi_source_stats": result.get("retrieval_info", {}),
                "config_used": result.get("config_used", {}),
                "review_info": result.get("review_info", {}),
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
        else:
            response_data = {
                "success": False,
                "error": result.get("error", "Review mode failed"),
                "session_id": session_id,
                "remote_method": remote_method,
                "timestamp": datetime.now().isoformat()
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Review mode error: {e}")
        return jsonify({
            "success": False,
            "error": f"Review mode error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/retry-review', methods=['POST'])
def retry_review_mode():
    """Retry last query in review mode with remote LLM"""
    try:
        if not chatbot_status["loaded"] or chatbot is None:
            return jsonify({
                "success": False,
                "error": "Chatbot not available"
            }), 500
        
        data = request.get_json() or {}
        remote_method = data.get('remote_method', None)
        session_id = data.get('session_id', 'default')
        
        logger.info(f"Retry review mode (method: {remote_method})...")
        
        start_time = time.time()
        retry_result = chatbot.retry_with_remote_llm(method=remote_method, use_review_mode=True)
        processing_time = time.time() - start_time
        
        # Convert sets to lists for JSON serialization
        def convert_sets_to_lists(obj):
            if isinstance(obj, dict):
                return {k: convert_sets_to_lists(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets_to_lists(item) for item in obj]
            elif isinstance(obj, set):
                return list(obj)
            else:
                return obj
        
        retry_result = convert_sets_to_lists(retry_result)
        
        # Update session history
        if session_id in chat_sessions and retry_result["success"]:
            retry_entry = {
                'query': retry_result.get("query", ""),
                'response': retry_result["response"],
                'timestamp': datetime.now().isoformat(),
                'processing_time': round(processing_time, 2),
                'source': 'retry_review_mode',
                'remote_method': retry_result.get("method", remote_method),
                'success': True,
                'review_info': retry_result.get("review_info", {})
            }
            chat_sessions[session_id]['history'].append(retry_entry)
        
        if retry_result["success"]:
            logger.info(f"Review retry successful in {processing_time:.2f} seconds")
            response_data = {
                "success": True,
                "response": retry_result["response"],
                "response_source": retry_result.get("response_source", "retry_review"),
                "method": retry_result.get("method", remote_method),
                "processing_time": round(processing_time, 2),
                "session_id": session_id,
                "query_result": convert_sets_to_lists(retry_result.get("query_result")),
                "review_info": retry_result.get("review_info", {}),
                "timestamp": datetime.now().isoformat()
            }
        else:
            logger.warning(f"Review retry failed: {retry_result.get('error', 'Unknown')}")
            response_data = {
                "success": False,
                "error": retry_result.get("error", "Review retry failed"),
                "fallback_response": retry_result.get("fallback_response"),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Review retry error: {e}")
        return jsonify({
            "success": False,
            "error": f"Review retry error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500



# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "system": "multi_source_rag",
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "system": "multi_source_rag", 
        "timestamp": datetime.now().isoformat()
    }), 500

@app.errorhandler(503)
def service_unavailable(error):
    return jsonify({
        "error": "Multi-source service temporarily unavailable",
        "status": "Please wait for system initialization",
        "system": "multi_source_rag",
        "timestamp": datetime.now().isoformat()
    }), 503

if __name__ == '__main__':
    print("Starting Enhanced Multi-Source Flask API Server...")
    print("UI available in: http://localhost:5000")
    print(" Multi-Source API endpoints:")
    print("   • Chat (local): POST /api/chat")
    print("   • Chat (remote): POST /api/chat/remote") 
    print("   • Retry: POST /api/retry")
    print("   • Batch: POST /api/batch")
    print("   • Config: GET/POST /api/config")
    print("   • Test Remote: POST /api/test-remote")
    print("   • Status: GET /api/status")
    print("   • Stats: GET /api/stats")
    print("   • Sources: GET /api/sources")
    print("   • Health: GET /api/health")
    print("   • History: GET /api/history/<session_id>")
    print(" Multi-Source Features:")
    print("   • QA Database Integration")
    print("   • Notes Database Integration") 
    print("   • Laws Database Integration")
    print("   • Dynamic Weight Configuration")
    print("   • Hybrid Search (TF-IDF + Semantic)")
    print("-" * 60)
    print("Waiting for system to initialize...")
    
    # Flask server run
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )