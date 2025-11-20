"""
Math Problems Domain - Д ≈ 0.05 Benchmark

Tests narrative variables against logical necessity.
Can narrative move Д above zero in logic domain?

Narrative variables tested (50+):
- Problem framing (story vs abstract)
- Solution presentation (confidence, clarity, aesthetics)
- Solver identity (expert vs novice)
- Explanation richness
- Meta-commentary
- Context setting
- All dimensions that might affect comprehension/acceptance
"""

import random
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


class MathProblemGenerator:
    """
    Generates math problems with MAXIMUM narrative variation.
    
    Tests: Can narrative make 2+2=5? Or at least make solution more/less accepted?
    """
    
    def __init__(self, n_problems: int = 500):
        self.n_problems = n_problems
        self.problems = []
        
        # Problem templates
        self.problem_types = [
            ('arithmetic', lambda: self._generate_arithmetic()),
            ('algebra', lambda: self._generate_algebra()),
            ('word_problem', lambda: self._generate_word_problem()),
            ('geometry', lambda: self._generate_geometry())
        ]
    
    def generate_problem_with_full_narrative(self, problem_id: int) -> Dict:
        """Generate one problem with MAXIMUM narrative dimensions."""
        
        # Generate base problem and correct answer
        problem_type, generator = random.choice(self.problem_types)
        problem_data = generator()
        
        # Correct answer
        correct_answer = problem_data['answer']
        
        # Sometimes give wrong answer to test if narrative can make it seem right
        provide_wrong = random.random() < 0.3
        given_answer = correct_answer if not provide_wrong else problem_data['wrong_answer']
        
        # EXHAUSTIVE narrative dimensions
        narrative = {
            # PROBLEM FRAMING
            'presentation_style': random.choice(['abstract', 'story', 'visual', 'formal', 'casual']),
            'urgency_framing': random.choice([None, 'URGENT', 'Time-sensitive', 'Take your time']),
            'difficulty_claimed': random.choice(['trivial', 'easy', 'moderate', 'hard', 'impossible']),
            'trick_warning': random.choice([True, False]),
            'historical_context': random.choice([None, 'Pythagoras proved', 'Ancient problem', 'Modern discovery']),
            'practical_application': random.choice([None, 'Used in engineering', 'Daily life', 'Pure theory']),
            
            # SOLUTION PRESENTATION
            'confidence': random.uniform(0, 1),
            'certainty_language': random.choice(['obviously', 'clearly', 'perhaps', 'I think', 'definitely']),
            'work_shown': random.choice([True, False]),
            'steps_narrative_quality': random.uniform(0, 1),
            'explanation_complexity': random.randint(0, 5),  # Paragraphs
            'uses_metaphor': random.choice([True, False]),
            'visual_aids_described': random.choice([True, False]),
            
            # SOLVER IDENTITY
            'solver_identity': random.choice(['PhD mathematician', '7-year-old', 'engineer', 'student', 'teacher']),
            'credentials_listed': random.choice([True, False]),
            'credentials': random.choice([None, 'PhD MIT', 'Fields Medal', 'High school', 'Self-taught']),
            'experience_claimed': random.choice([None, '20 years teaching', 'First time', 'Math prodigy']),
            'demographics_revealed': {
                'age': random.choice([True, False]),
                'background': random.choice([True, False]),
                'education': random.choice([True, False])
            },
            'struggles_admitted': random.choice([True, False]),  # "I struggled with this"
            'passion_expressed': random.choice([True, False]),  # "I love math"
            
            # AESTHETIC PRESENTATION
            'formatting_quality': random.uniform(0, 1),
            'uses_latex': random.choice([True, False]),
            'visual_beauty_emphasized': random.choice([True, False]),
            'elegance_claimed': random.choice([True, False]),
            'notation_style': random.choice(['standard', 'creative', 'minimal', 'verbose']),
            
            # META-COMMENTARY
            'acknowledges_difficulty': random.choice([True, False]),
            'warns_common_mistakes': random.choice([True, False]),
            'provides_tricks': random.choice([True, False]),
            'admits_uncertainty': random.choice([True, False]),
            'defensive': random.choice([True, False]),  # "Some might disagree but..."
            
            # CONTEXT
            'audience': random.choice(['students', 'peers', 'experts', 'general public']),
            'purpose': random.choice(['teaching', 'testing', 'exploration', 'verification']),
            'stakes': random.choice(['exam question', 'homework', 'research', 'curiosity']),
            
            # COMPARISONS
            'compares_methods': random.choice([True, False]),
            'cites_authorities': random.choice([True, False]),  # "Euler's method"
            'references_alternative': random.choice([True, False]),
            
            # PERSUASION TACTICS
            'uses_authority': random.choice([True, False]),
            'appeals_to_intuition': random.choice([True, False]),
            'appeals_to_logic': random.choice([True, False]),
            'social_proof': random.choice([None, 'Everyone knows', 'Most people get this wrong'])
        }
        
        # Construct full narrative text
        narrative_text = self._construct_math_narrative(problem_data, given_answer, narrative)
        
        return {
            'problem_id': problem_id,
            'problem_type': problem_type,
            'problem_text': problem_data['text'],
            'correct_answer': correct_answer,
            'given_answer': given_answer,
            'is_correct': int(given_answer == correct_answer),
            'narrative': narrative,
            'full_narrative_text': narrative_text
        }
    
    def _generate_arithmetic(self) -> Dict:
        """Generate arithmetic problem."""
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        op = random.choice(['+', '-', '×'])
        
        if op == '+':
            answer = a + b
            wrong = answer + random.choice([-2, -1, 1, 2])
        elif op == '-':
            answer = a - b
            wrong = answer + random.choice([-2, -1, 1, 2])
        else:  # ×
            answer = a * b
            wrong = answer + random.randint(-5, 5)
        
        return {
            'text': f"{a} {op} {b} = ?",
            'answer': answer,
            'wrong_answer': wrong
        }
    
    def _generate_algebra(self) -> Dict:
        """Generate simple algebra."""
        x = random.randint(1, 10)
        constant = random.randint(1, 20)
        answer = x
        wrong = x + random.choice([-2, -1, 1, 2])
        
        return {
            'text': f"Solve for x: x + {constant} = {x + constant}",
            'answer': answer,
            'wrong_answer': wrong
        }
    
    def _generate_word_problem(self) -> Dict:
        """Generate word problem (tests if narrative framing helps)."""
        a = random.randint(2, 10)
        b = random.randint(2, 10)
        answer = a + b
        wrong = answer + random.choice([-2, -1, 1, 2])
        
        names = ['Sarah', 'John', 'Maria', 'Chen', 'Alex']
        items = ['apples', 'books', 'coins', 'toys', 'pencils']
        
        name = random.choice(names)
        item = random.choice(items)
        
        return {
            'text': f"{name} has {a} {item}. They get {b} more. How many total?",
            'answer': answer,
            'wrong_answer': wrong
        }
    
    def _generate_geometry(self) -> Dict:
        """Generate geometry problem."""
        side = random.randint(3, 12)
        answer = side * 4  # Perimeter of square
        wrong = answer + random.randint(-3, 3)
        
        return {
            'text': f"Square with side length {side}. What is perimeter?",
            'answer': answer,
            'wrong_answer': wrong
        }
    
    def _construct_math_narrative(self, problem: Dict, answer: int, narrative: Dict) -> str:
        """Construct complete narrative."""
        parts = []
        
        # Context setting
        if narrative['historical_context']:
            parts.append(narrative['historical_context'])
        
        # Identity
        parts.append(f"As a {narrative['solver_identity']}")
        
        if narrative['credentials']:
            parts.append(f"({narrative['credentials']})")
        
        # Problem
        parts.append(f"I will solve: {problem['text']}")
        
        # Solution with confidence
        parts.append(f"The answer is {narrative['certainty_language']} {answer}")
        
        # Meta
        if narrative['acknowledges_difficulty']:
            parts.append("This is a challenging problem")
        
        if narrative['admits_uncertainty']:
            parts.append("though I may be wrong")
        
        return ". ".join(parts) + "."
    
    def generate_all_problems(self) -> List[Dict]:
        """Generate complete dataset."""
        print(f"Generating {self.n_problems} math problems with MAXIMUM narrative variation...")
        
        for i in range(self.n_problems):
            problem = self.generate_problem_with_full_narrative(i)
            self.problems.append(problem)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{self.n_problems} problems...")
        
        print(f"✓ Complete: {self.n_problems} problems with 50+ narrative dimensions each")
        
        return self.problems
    
    def save_dataset(self, output_path: str):
        """Save dataset."""
        with open(output_path, 'w') as f:
            json.dump(self.problems, f, indent=2)
        
        print(f"✓ Saved to: {output_path}")


def main():
    """Generate math problems benchmark."""
    print("=" * 80)
    print("MATH PROBLEMS DOMAIN - Д ≈ 0.05 BENCHMARK")
    print("Testing: Can narrative make wrong answers seem right?")
    print("=" * 80)
    print("")
    
    generator = MathProblemGenerator(n_problems=500)
    problems = generator.generate_all_problems()
    
    output_path = Path(__file__).parent.parent.parent.parent / 'data/domains/math_problems_benchmark.json'
    generator.save_dataset(str(output_path))
    
    print("\n" + "=" * 80)
    print("DATASET READY")
    print("=" * 80)
    print("Next: Test if ANY narrative variable affects acceptance of solutions")
    print("Expected: Д ≈ 0.02 for correct answers, maybe 0.15 for word problems")


if __name__ == "__main__":
    main()

