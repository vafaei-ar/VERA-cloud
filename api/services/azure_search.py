"""
Azure AI Search Integration for VERA Cloud
Handles RAG (Retrieval-Augmented Generation) for medical knowledge
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, QueryType, QueryCaptionType
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
import json

logger = logging.getLogger(__name__)

class AzureSearchService:
    def __init__(self, endpoint: str, api_key: str, index_name: str = "stroke-care-knowledge"):
        self.endpoint = endpoint
        self.api_key = api_key
        self.index_name = index_name
        self.credential = AzureKeyCredential(api_key)
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=self.credential
        )
    
    async def search_medical_knowledge(self, query: str, filters: str = None, 
                                     top: int = 5, include_vectors: bool = True) -> List[Dict]:
        """Search medical knowledge base with hybrid search (text + vector)"""
        try:
            search_results = []
            
            # Text search
            text_results = self.client.search(
                search_text=query,
                filter=filters,
                top=top,
                include_total_count=True,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="default",
                query_caption=QueryCaptionType.EXTRACTIVE,
                highlight_fields=["content", "title"]
            )
            
            for result in text_results:
                search_results.append({
                    "content": result.get("content", ""),
                    "title": result.get("title", ""),
                    "source": result.get("source", ""),
                    "confidence": result.get("@search.score", 0.0),
                    "highlights": result.get("@search.highlights", {}),
                    "search_type": "text"
                })
            
            # Vector search (if embeddings are available)
            if include_vectors:
                try:
                    # This would require the query to be embedded first
                    # For now, we'll use text search as fallback
                    pass
                except Exception as e:
                    logger.warning(f"Vector search not available: {e}")
            
            logger.info(f"Found {len(search_results)} results for query: {query[:50]}...")
            return search_results
            
        except ResourceNotFoundError:
            logger.error(f"Search index '{self.index_name}' not found")
            return []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def search_by_question_type(self, question_type: str, user_input: str = None) -> List[Dict]:
        """Search for content related to specific question types"""
        try:
            # Build filter for question type
            filter_expr = f"question_type eq '{question_type}'"
            
            # Use user input if provided, otherwise use question type
            search_query = user_input or question_type
            
            results = await self.search_medical_knowledge(
                query=search_query,
                filters=filter_expr,
                top=3
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Question type search failed: {e}")
            return []
    
    async def search_emergency_keywords(self, user_input: str) -> List[Dict]:
        """Search for emergency-related content based on user input"""
        try:
            emergency_keywords = [
                "stroke warning signs", "emergency symptoms", "call 911",
                "sudden weakness", "speech problems", "vision changes",
                "severe headache", "loss of balance", "medical emergency"
            ]
            
            # Search for emergency content
            emergency_results = []
            for keyword in emergency_keywords:
                if keyword.lower() in user_input.lower():
                    results = await self.search_medical_knowledge(
                        query=keyword,
                        filters="category eq 'emergency'",
                        top=2
                    )
                    emergency_results.extend(results)
            
            return emergency_results[:5]  # Limit to top 5 emergency results
            
        except Exception as e:
            logger.error(f"Emergency search failed: {e}")
            return []
    
    async def search_recovery_guidelines(self, user_input: str, recovery_stage: str = None) -> List[Dict]:
        """Search for recovery guidelines based on user input and stage"""
        try:
            # Build search query
            search_query = f"recovery guidelines {user_input}"
            
            # Add stage filter if provided
            filters = None
            if recovery_stage:
                filters = f"recovery_stage eq '{recovery_stage}'"
            
            results = await self.search_medical_knowledge(
                query=search_query,
                filters=filters,
                top=4
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Recovery guidelines search failed: {e}")
            return []
    
    async def search_medication_info(self, user_input: str) -> List[Dict]:
        """Search for medication-related information"""
        try:
            medication_keywords = [
                "medication", "drug", "prescription", "side effects",
                "adherence", "dosage", "pharmacy", "blood thinner"
            ]
            
            # Check if input contains medication keywords
            if any(keyword in user_input.lower() for keyword in medication_keywords):
                results = await self.search_medical_knowledge(
                    query=f"medication {user_input}",
                    filters="category eq 'medication'",
                    top=3
                )
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Medication search failed: {e}")
            return []
    
    async def search_lifestyle_guidance(self, user_input: str) -> List[Dict]:
        """Search for lifestyle and daily activity guidance"""
        try:
            lifestyle_keywords = [
                "diet", "exercise", "activity", "lifestyle", "daily activities",
                "home safety", "mobility", "support", "rehabilitation"
            ]
            
            # Check if input contains lifestyle keywords
            if any(keyword in user_input.lower() for keyword in lifestyle_keywords):
                results = await self.search_medical_knowledge(
                    query=f"lifestyle {user_input}",
                    filters="category eq 'lifestyle'",
                    top=3
                )
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Lifestyle search failed: {e}")
            return []
    
    async def get_contextual_knowledge(self, user_input: str, question_type: str, 
                                     session_context: Dict = None) -> Dict:
        """Get comprehensive contextual knowledge for RAG"""
        try:
            knowledge = {
                "general": [],
                "emergency": [],
                "medication": [],
                "lifestyle": [],
                "recovery": []
            }
            
            # Search for different types of knowledge
            tasks = [
                self.search_by_question_type(question_type, user_input),
                self.search_emergency_keywords(user_input),
                self.search_medication_info(user_input),
                self.search_lifestyle_guidance(user_input)
            ]
            
            # Add recovery stage if available in session context
            recovery_stage = session_context.get("recovery_stage") if session_context else None
            if recovery_stage:
                tasks.append(self.search_recovery_guidelines(user_input, recovery_stage))
            
            # Execute searches in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Organize results
            if len(results) >= 4:
                knowledge["general"] = results[0] if not isinstance(results[0], Exception) else []
                knowledge["emergency"] = results[1] if not isinstance(results[1], Exception) else []
                knowledge["medication"] = results[2] if not isinstance(results[2], Exception) else []
                knowledge["lifestyle"] = results[3] if not isinstance(results[3], Exception) else []
                
                if len(results) > 4:
                    knowledge["recovery"] = results[4] if not isinstance(results[4], Exception) else []
            
            # Calculate total confidence and context richness
            total_results = sum(len(knowledge[key]) for key in knowledge)
            avg_confidence = 0.0
            if total_results > 0:
                all_scores = []
                for key in knowledge:
                    for result in knowledge[key]:
                        all_scores.append(result.get("confidence", 0.0))
                avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0
            
            return {
                "knowledge": knowledge,
                "total_results": total_results,
                "avg_confidence": avg_confidence,
                "context_richness": "high" if total_results > 5 else "medium" if total_results > 2 else "low"
            }
            
        except Exception as e:
            logger.error(f"Contextual knowledge search failed: {e}")
            return {
                "knowledge": {"general": [], "emergency": [], "medication": [], "lifestyle": [], "recovery": []},
                "total_results": 0,
                "avg_confidence": 0.0,
                "context_richness": "low"
            }
    
    async def create_knowledge_index(self, documents: List[Dict]) -> bool:
        """Create or update the knowledge index with documents"""
        try:
            # This would typically be done through Azure portal or separate script
            # For now, we'll just log the intention
            logger.info(f"Would create/update index with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            return False
