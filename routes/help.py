"""
Help and Documentation Routes

Glossaries, examples, and plain English explanations.
"""

from flask import Blueprint, render_template

help_bp = Blueprint('help', __name__)

@help_bp.route('/metrics')
def metrics_glossary():
    """Display metrics glossary with plain English definitions."""
    return render_template('metrics_glossary.html')

@help_bp.route('/features/<transformer_name>')
def feature_inspector(transformer_name):
    """Display feature explanations for a transformer."""
    return render_template('feature_inspector.html', transformer_name=transformer_name)


