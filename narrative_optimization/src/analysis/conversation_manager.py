"""
Conversational Context Manager

Tracks conversation state across interactions to enable progressive narrative understanding.
Maintains user responses, context history, and refined predictions.
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import uuid


class ConversationManager:
    """
    Manages multi-turn conversations for narrative analysis refinement.
    
    Tracks:
    - Questions asked
    - User responses
    - Context evolution
    - Prediction refinements
    - Conversation history
    """
    
    def __init__(self):
        self.conversations = {}  # conversation_id -> conversation_state
    
    def start_conversation(
        self,
        text_a: str,
        text_b: str,
        initial_question: str = '',
        comparison_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Start a new conversation for a comparison.
        
        Parameters
        ----------
        text_a : str
            First text to compare
        text_b : str
            Second text to compare
        initial_question : str
            User's initial question
        comparison_data : dict
            Initial comparison results
        
        Returns
        -------
        conversation : dict
            New conversation state with ID
        """
        conversation_id = str(uuid.uuid4())
        
        conversation = {
            'id': conversation_id,
            'created_at': datetime.now().isoformat(),
            'texts': {
                'text_a': text_a,
                'text_b': text_b
            },
            'initial_question': initial_question,
            'turns': [],
            'context': {},
            'refined_predictions': [],
            'comparison_data': comparison_data or {},
            'status': 'active'
        }
        
        self.conversations[conversation_id] = conversation
        return conversation
    
    def add_turn(
        self,
        conversation_id: str,
        questions: List[str],
        responses: Optional[Dict[str, str]] = None,
        refined_analysis: Optional[Dict] = None
    ):
        """
        Add a conversation turn (questions + responses).
        
        Parameters
        ----------
        conversation_id : str
            Conversation identifier
        questions : list of str
            Questions asked in this turn
        responses : dict
            User responses {question: answer}
        refined_analysis : dict
            Updated analysis based on responses
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        turn = {
            'turn_number': len(self.conversations[conversation_id]['turns']) + 1,
            'timestamp': datetime.now().isoformat(),
            'questions': questions,
            'responses': responses or {},
            'refined_analysis': refined_analysis
        }
        
        self.conversations[conversation_id]['turns'].append(turn)
        
        # Update context with new information
        if responses:
            for question, answer in responses.items():
                self._update_context(conversation_id, question, answer)
    
    def _update_context(self, conversation_id: str, question: str, answer: str):
        """Update conversation context with new information."""
        conv = self.conversations[conversation_id]
        
        # Extract context variables from Q&A
        context_updates = self._extract_context_from_qa(question, answer)
        conv['context'].update(context_updates)
    
    def _extract_context_from_qa(self, question: str, answer: str) -> Dict[str, Any]:
        """Extract structured context from question-answer pair."""
        context = {}
        
        q_lower = question.lower()
        
        # Sports context
        if 'record' in q_lower or 'win' in q_lower:
            context['performance_info'] = answer
        if 'injury' in q_lower or 'injured' in q_lower:
            context['injury_status'] = answer
        if 'home' in q_lower or 'away' in q_lower:
            context['location'] = answer
        if 'recent' in q_lower or 'lately' in q_lower:
            context['recent_form'] = answer
        
        # Stakes and context
        if 'stakes' in q_lower or 'important' in q_lower:
            context['stakes_level'] = answer
        if 'rivalry' in q_lower or 'history' in q_lower:
            context['historical_context'] = answer
        if 'momentum' in q_lower or 'streak' in q_lower:
            context['momentum'] = answer
        
        # Product context
        if 'price' in q_lower or 'cost' in q_lower:
            context['price_info'] = answer
        if 'target' in q_lower or 'audience' in q_lower:
            context['target_audience'] = answer
        
        # Profile context
        if 'goal' in q_lower or 'looking for' in q_lower:
            context['goals'] = answer
        if 'value' in q_lower:
            context['values'] = answer
        
        # General context
        if 'frame' in q_lower or 'see' in q_lower or 'view' in q_lower:
            context['narrative_framing'] = answer
        
        # Store raw Q&A too
        context[f'qa_{len(context)}'] = {'q': question, 'a': answer}
        
        return context
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation state by ID."""
        return self.conversations.get(conversation_id)
    
    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get accumulated context for a conversation."""
        conv = self.conversations.get(conversation_id)
        return conv['context'] if conv else {}
    
    def get_all_responses(self, conversation_id: str) -> Dict[str, str]:
        """Get all user responses across all turns."""
        conv = self.conversations.get(conversation_id)
        if not conv:
            return {}
        
        all_responses = {}
        for turn in conv['turns']:
            all_responses.update(turn.get('responses', {}))
        
        return all_responses
    
    def get_history_summary(self, conversation_id: str) -> str:
        """Get a summary of conversation history for AI context."""
        conv = self.conversations.get(conversation_id)
        if not conv:
            return ""
        
        summary_parts = [
            f"Comparing: '{conv['texts']['text_a'][:100]}...' vs '{conv['texts']['text_b'][:100]}...'"
        ]
        
        if conv['initial_question']:
            summary_parts.append(f"User asked: '{conv['initial_question']}'")
        
        for turn in conv['turns']:
            summary_parts.append(f"\nTurn {turn['turn_number']}:")
            for q, a in turn.get('responses', {}).items():
                summary_parts.append(f"  Q: {q}")
                summary_parts.append(f"  A: {a}")
        
        if conv['context']:
            summary_parts.append(f"\nAccumulated context: {json.dumps(conv['context'], indent=2)}")
        
        return '\n'.join(summary_parts)
    
    def should_continue_conversation(self, conversation_id: str, max_turns: int = 3) -> bool:
        """Determine if conversation should continue or conclude."""
        conv = self.conversations.get(conversation_id)
        if not conv:
            return False
        
        # Stop after max turns
        if len(conv['turns']) >= max_turns:
            return False
        
        # Stop if context is well-populated
        if len(conv['context']) >= 5:
            return False
        
        return True
    
    def mark_complete(self, conversation_id: str):
        """Mark conversation as complete."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]['status'] = 'complete'
    
    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Remove conversations older than specified hours."""
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for conv_id, conv in self.conversations.items():
            created = datetime.fromisoformat(conv['created_at'])
            if created < cutoff:
                to_remove.append(conv_id)
        
        for conv_id in to_remove:
            del self.conversations[conv_id]
        
        return len(to_remove)


# Global conversation manager instance
conversation_manager = ConversationManager()

