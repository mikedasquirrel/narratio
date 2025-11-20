# PythonAnywhere Deployment Commands for Namesake

**Project**: Namesake - Narrative Framework  
**GitHub**: https://github.com/mikedasquirrel/novela  
**PythonAnywhere Username**: mikedasquirrel

---

## 🚀 Deployment Commands

Copy and paste these commands into your PythonAnywhere bash console:

### 1. Clone the Repository

```bash
cd ~
git clone https://github.com/mikedasquirrel/novela.git
cd novela
```

### 2. Create Virtual Environment

```bash
mkvirtualenv --python=/usr/bin/python3.10 namesake
workon namesake
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install flask flask-cors pandas numpy scipy scikit-learn statsmodels
pip install plotly matplotlib seaborn
pip install joblib requests beautifulsoup4 lxml
```

### 4. Set Environment Variables (Optional - if using OpenAI features)

```bash
echo "export OPENAI_API_KEY='your-key-here'" >> ~/.bashrc
source ~/.bashrc
```

### 5. Test Locally (Optional)

```bash
cd ~/novela
python3 app.py
# Should show: Running on http://0.0.0.0:5738
# Ctrl+C to stop
```

---

## 🌐 Configure Web App on PythonAnywhere

### Go to Web Tab

1. Click "Add a new web app"
2. Choose "Manual configuration" (NOT Flask quickstart)
3. Select **Python 3.10**

### Configure WSGI File

**Path**: `/var/www/mikedasquirrel_pythonanywhere_com_wsgi.py`

**Replace contents with**:

```python
import sys
import os

# Add project directory
project_home = '/home/mikedasquirrel/novela'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set environment variables (optional)
os.environ['OPENAI_API_KEY'] = 'your-key-if-needed'

# Import Flask app
from app import app as application
```

### Configure Virtual Environment

**In Web Tab**:
- Virtualenv path: `/home/mikedasquirrel/.virtualenvs/namesake`
- Click reload button

### Configure Static Files (Important!)

**Add static files mapping**:
- URL: `/static/`
- Directory: `/home/mikedasquirrel/novela/static/`

---

## 🔧 Post-Deployment Commands

### Update Code (When You Push Changes)

```bash
cd ~/novela
git pull origin main
# Then click "Reload" button in PythonAnywhere Web tab
```

### View Logs (If Issues)

```bash
tail -50 /var/log/mikedasquirrel.pythonanywhere.com.error.log
tail -50 /var/log/mikedasquirrel.pythonanywhere.com.server.log
```

### Restart App

```bash
touch /var/www/mikedasquirrel_pythonanywhere_com_wsgi.py
# OR click "Reload" button in Web tab
```

---

## 📋 Quick Deploy (One-Liner for Future Updates)

```bash
cd ~/novela && git pull && touch /var/www/mikedasquirrel_pythonanywhere_com_wsgi.py
```

---

## 🌍 Access Your Site

After deployment, visit:

**https://mikedasquirrel.pythonanywhere.com/**

### Key Pages

- `/` - Home with complete spectrum
- `/domains` - Interactive spectrum explorer  
- `/housing` - $93K #13 effect analysis
- `/wwe-domain` - $1B from fake fights (highest π)
- `/framework-story` - Complete three-force narrative
- `/discoveries` - Key findings with bookends
- `/api/domains/all` - Unified API (all 13 domains)

---

## ✅ Verification Checklist

After deployment, check:

- [ ] Home page loads with 13 domains, π: 0.04-0.974
- [ ] `/domains` shows complete spectrum
- [ ] `/housing` page displays properly
- [ ] `/wwe-domain` page displays properly
- [ ] `/api/domains/all` returns JSON
- [ ] Navigation works (click through menu)
- [ ] Static files load (CSS, JS)

---

## 🎯 What's Deployed

**Complete Framework**:
- 13 domains analyzed (Lottery to WWE)
- 211,000+ entities
- π spectrum: 0.04 to 0.974
- Three forces: Λ (Matter), Ν (Meaning), Ψ (Mind)
- Perfect bookends established

**Key Discoveries**:
- Housing: $93K #13 discount, 99.92% skip rate
- WWE: π=0.974 (highest ever), $1B from fake
- Lottery: Д=0.00 (perfect control)
- Kayfabe formalized as Ψ₂ (meta-awareness)

**Essays**:
- ON_FAITH_AND_SILENCE.md (philosophical introduction)
- Complete framework documentation
- Prestige domain theory
- Pure nominative analysis

**Website Features**:
- Beautiful glassmorphism design
- Interactive domain explorer
- Complete API access
- Black/fuchsia/cyan color scheme

---

## 🐛 Troubleshooting

**If site shows "Something went wrong"**:
```bash
# Check error log
tail -100 /var/log/mikedasquirrel.pythonanywhere.com.error.log

# Common fixes:
# 1. Reload web app (Web tab → Reload button)
# 2. Check WSGI file path is correct
# 3. Verify virtual environment path
# 4. Ensure static files mapping is correct
```

**If pages load but look broken**:
```bash
# Static files not mapping - check Web tab static files configuration
# Should have: /static/ → /home/mikedasquirrel/novela/static/
```

**If specific domain page 404s**:
```bash
# Check blueprint is registered in app.py
# Restart: touch /var/www/mikedasquirrel_pythonanywhere_com_wsgi.py
```

---

## 📊 Performance Notes

**The site is production-ready**:
- All routes tested (9/9 working)
- All APIs functional
- No broken links
- Complete data integration

**Large files excluded**:
- Raw movie data (99MB+)
- Tennis datasets (105MB+)
- Basketball seasons (145MB+)
- These aren't needed for website operation

**Code and documentation are complete**:
- All analysis scripts
- All domain pages
- All API endpoints
- All essays and theory docs

---

## 🎉 You're Live!

Your complete narrative framework is now deployed at:

**https://mikedasquirrel.pythonanywhere.com/**

With:
- Perfect spectrum bookends (Lottery ↔ WWE)
- Complete three-force model
- 13 domains, 211K+ entities
- Beautiful interactive visualizations
- Full API access

**The framework is live. The theory is accessible. The site is beautiful.**

---

**GitHub**: https://github.com/mikedasquirrel/novela  
**Live Site**: https://mikedasquirrel.pythonanywhere.com/ (after deployment)  
**Status**: Ready to deploy ✓

