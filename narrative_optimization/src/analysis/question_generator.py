"""
Intelligent Question Generator

Generates domain-specific clarifying questions to improve narrative analysis.
Uses AI to create natural, relevant questions based on context and missing information.
"""

from typing import List, Dict, Any, Optional
import re
import json


class IntelligentQuestionGenerator:
    """
    Generates clarifying questions for narrative analysis refinement.
    
    Creates questions that:
    - Identify missing critical variables
    - Clarify narrative framing
    - Uncover context and stakes
    - Enable better predictions
    """
    
    def __init__(self, openai_client=None):
        self.openai_client = openai_client
        
        # Domain-specific question templates
        self.question_templates = {
            'sports': {
                'context': [
                    "What are the stakes of this matchup? (championship, playoff, rivalry, regular season)",
                    "How have these teams/athletes performed recently?",
                    "Are there any key injuries or roster changes?",
                    "Is there a historical rivalry or significance to this matchup?",
                    "What's the momentum going into this event?"
                ],
                'framing': [
                    "How would you frame this competition? (David vs Goliath, clash of titans, redemption story)",
                    "What narrative is most important here? (skill, heart, strategy, momentum)",
                    "Is this more about individual talent or team dynamics?"
                ],
                'prediction': [
                    "Are you asking about immediate outcome or long-term performance?",
                    "What time horizon matters most? (single game, series, season)"
                ]
            },
            
            'products': {
                'context': [
                    "What's the price range for these products?",
                    "Who is the target customer for each?",
                    "What's most important: features, value, brand, or innovation?",
                    "Are these competing in the same market segment?"
                ],
                'framing': [
                    "How would you characterize each product's positioning? (premium, value, innovative, reliable)",
                    "What matters more: technical specs or user experience?"
                ],
                'prediction': [
                    "Are you comparing quality, sales potential, or customer satisfaction?",
                    "What defines 'better' in this context?"
                ]
            },
            
            'profiles': {
                'context': [
                    "What are the goals or intentions of each person?",
                    "What type of connection or relationship is being evaluated?",
                    "What values or characteristics matter most?"
                ],
                'framing': [
                    "How would you describe each person's narrative? (adventurer, thoughtful, ambitious)",
                    "Is this about compatibility, complementarity, or something else?"
                ],
                'prediction': [
                    "What kind of outcome are you predicting? (compatibility, success, satisfaction)",
                    "Short-term chemistry or long-term compatibility?"
                ]
            },
            
            'brands': {
                'context': [
                    "What's the target market for each brand?",
                    "What's the company size and market position?",
                    "How long have these brands been established?"
                ],
                'framing': [
                    "How would you characterize each brand's identity? (innovative, traditional, disruptive)",
                    "What matters more: mission, execution, or market position?"
                ],
                'prediction': [
                    "Are you comparing brand strength, market performance, or customer loyalty?",
                    "What success metric matters most?"
                ]
            },
            
            'general': {
                'context': [
                    "What additional context would help understand this comparison?",
                    "Are there any relevant constraints or circumstances?",
                    "What's the broader situation or environment?"
                ],
                'framing': [
                    "How do you see this comparison? What lens are you using?",
                    "What aspect is most important to you?"
                ],
                'prediction': [
                    "What outcome are you trying to predict or understand?",
                    "What would 'success' or 'better' mean in this context?"
                ]
            }
        }
    
    def generate_questions(
        self,
        text_a: str,
        text_b: str,
        domain: str,
        comparison_data: Dict,
        user_question: str = '',
        context: Optional[Dict] = None,
        max_questions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate intelligent follow-up questions.
        
        Parameters
        ----------
        text_a, text_b : str
            Texts being compared
        domain : str
            Detected domain
        comparison_data : dict
            Initial comparison results
        user_question : str
            User's initial question
        context : dict
            Already known context
        max_questions : int
            Maximum questions to generate
        
        Returns
        -------
        questions : list of dict
            Generated questions with metadata
        """
        context = context or {}
        
        # Start with template-based questions
        template_questions = self._get_template_questions(domain, context, max_questions)
        
        # If AI is available, enhance with AI-generated questions
        if self.openai_client:
            try:
                ai_questions = self._generate_ai_questions(
                    text_a, text_b, domain, comparison_data, user_question, context
                )
                # Merge, prioritizing AI questions
                all_questions = ai_questions + template_questions
            except Exception as e:
                print(f"AI question generation failed: {e}")
                all_questions = template_questions
        else:
            all_questions = template_questions
        
        # Deduplicate and limit
        unique_questions = self._deduplicate_questions(all_questions)
        return unique_questions[:max_questions]
    
    def _get_template_questions(
        self,
        domain: str,
        context: Dict,
        max_questions: int
    ) -> List[Dict[str, Any]]:
        """Get template-based questions for domain."""
        domain_templates = self.question_templates.get(domain, self.question_templates['general'])
        
        questions = []
        
        # Get context questions (most important)
        for q in domain_templates.get('context', []):
            if self._is_question_relevant(q, context):
                questions.append({
                    'question': q,
                    'category': 'context',
                    'priority': 'high',
                    'source': 'template'
                })
        
        # Get framing questions
        for q in domain_templates.get('framing', []):
            if self._is_question_relevant(q, context):
                questions.append({
                    'question': q,
                    'category': 'framing',
                    'priority': 'medium',
                    'source': 'template'
                })
        
        # Get prediction clarification questions
        for q in domain_templates.get('prediction', []):
            if self._is_question_relevant(q, context):
                questions.append({
                    'question': q,
                    'category': 'prediction_goal',
                    'priority': 'medium',
                    'source': 'template'
                })
        
        return questions
    
    def _is_question_relevant(self, question: str, context: Dict) -> bool:
        """Check if question is still relevant given current context."""
        q_lower = question.lower()
        
        # Skip if we already have this information
        if 'stakes' in q_lower and 'stakes_level' in context:
            return False
        if 'injury' in q_lower and 'injury_status' in context:
            return False
        if 'price' in q_lower and 'price_info' in context:
            return False
        if 'goal' in q_lower and 'goals' in context:
            return False
        if 'frame' in q_lower and 'narrative_framing' in context:
            return False
        
        return True
    
    def _generate_ai_questions(
        self,
        text_a: str,
        text_b: str,
        domain: str,
        comparison_data: Dict,
        user_question: str,
        context: Dict
    ) -> List[Dict[str, Any]]:
        """Use AI to generate contextual questions."""
        if not self.openai_client:
            return []
        
        # Build prompt
        prompt = self._build_question_prompt(
            text_a, text_b, domain, comparison_data, user_question, context
        )
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert narrative analyst. Generate 3-5 highly relevant clarifying questions that would significantly improve prediction accuracy. Focus on missing context, stakes, framing, and critical variables. Return ONLY a JSON array of question objects."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse JSON
            import json
            if '```json' in ai_response:
                json_str = ai_response.split('```json')[1].split('```')[0].strip()
            elif '```' in ai_response:
                json_str = ai_response.split('```')[1].split('```')[0].strip()
            else:
                json_str = ai_response.strip()
            
            questions_data = json.loads(json_str)
            
            # Format questions
            formatted = []
            for q in questions_data:
                if isinstance(q, dict):
                    formatted.append({
                        'question': q.get('question', q.get('text', '')),
                        'category': q.get('category', 'context'),
                        'priority': q.get('priority', 'high'),
                        'source': 'ai',
                        'reasoning': q.get('reasoning', '')
                    })
                elif isinstance(q, str):
                    formatted.append({
                        'question': q,
                        'category': 'context',
                        'priority': 'high',
                        'source': 'ai'
                    })
            
            return formatted
        
        except Exception as e:
            print(f"AI question generation error: {e}")
            return []
    
    def _build_question_prompt(
        self,
        text_a: str,
        text_b: str,
        domain: str,
        comparison_data: Dict,
        user_question: str,
        context: Dict
    ) -> str:
        """Build prompt for AI question generation."""
        prompt = f"""Generate 3-5 clarifying questions to improve this narrative analysis.

TEXT A: "{text_a[:300]}"
TEXT B: "{text_b[:300]}"

DOMAIN: {domain}
USER QUESTION: "{user_question if user_question else 'General comparison'}"

CURRENT ANALYSIS:
- Overall similarity: {comparison_data.get('overall_similarity', 0):.2%}
- Most different dimension: {comparison_data.get('most_different_dimension', 'unknown')}

CONTEXT ALREADY KNOWN:
{json.dumps(context, indent=2) if context else 'None yet'}

Generate questions that would:
1. Clarify missing critical context (stakes, circumstances, constraints)
2. Understand narrative framing ("How do you see this?")
3. Identify what success/outcome means
4. Uncover temporal factors (immediate vs long-term)
5. Reveal what truly matters in this comparison

Return JSON array format:
[
  {{
    "question": "Clear, specific question text",
    "category": "context|framing|prediction_goal|temporal",
    "priority": "high|medium",
    "reasoning": "Why this question matters"
  }}
]"""
        
        return prompt
    
    def _deduplicate_questions(self, questions: List[Dict]) -> List[Dict]:
        """Remove duplicate or very similar questions."""
        unique = []
        seen_text = set()
        
        for q in questions:
            q_text = q['question'].lower()
            # Simple deduplication
            if q_text not in seen_text:
                unique.append(q)
                seen_text.add(q_text)
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        unique.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 1))
        
        return unique
    
    def generate_stakes_questions(self, text_a: str, text_b: str, domain: str) -> List[str]:
        """Generate questions specifically about stakes and context weight."""
        questions = []
        
        if domain == 'sports':
            questions = [
                "What's at stake in this matchup? (championship, playoff berth, pride, seeding)",
                "Is this a high-stakes rivalry or a routine game?",
                "What's the timing context? (elimination game, season opener, mid-season)",
                "Is there meaningful momentum or a streak involved?"
            ]
        elif domain == 'products':
            questions = [
                "What's at stake for buyers? (major investment, minor purchase)",
                "Is this a high-stakes decision or low-risk choice?",
                "What's the context? (business critical, personal use, gift)"
            ]
        elif domain == 'profiles':
            questions = [
                "What's at stake in this comparison? (life partner, casual connection, project collaborator)",
                "Is this a high-stakes decision or exploratory?",
                "What matters most? (long-term compatibility, immediate chemistry, specific goals)"
            ]
        else:
            questions = [
                "What's at stake in this comparison?",
                "How important is this decision?",
                "What context makes this comparison matter?"
            ]
        
        return questions[:3]


def create_question_generator(openai_client=None):
    """Factory function to create question generator."""
    return IntelligentQuestionGenerator(openai_client)

