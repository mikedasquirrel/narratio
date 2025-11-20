"""
Ships Routes
"""

from flask import Blueprint, render_template
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))

from domains.ships.data_loader import ShipDataLoader

ships_bp = Blueprint('ships', __name__)

_data = None

def _load():
    global _data
    if _data is None:
        loader = ShipDataLoader()
        _data = loader.load_ships()
    return _data

@ships_bp.route('/')
def dashboard():
    ships = _load()
    return render_template('ships/dashboard.html', n_ships=len(ships))

@ships_bp.route('/browser')
def browser():
    ships = _load()
    return render_template('ships/browser.html', ships=ships[:100])

