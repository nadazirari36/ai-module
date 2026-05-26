# 🚀 STREAMLIT APP - INSTALLATION & USAGE

## STEP 1: INSTALL STREAMLIT (2 minutes)

```bash
cd E:\ai_module

# Activate virtual environment
venv\Scripts\activate

# Install Streamlit
pip install streamlit
```

---

## STEP 2: COPY THE FILE

Copy `streamlit_app.py` to your project folder:
```
E:\ai_module\streamlit_app.py
```

Make sure file structure looks like:
```
E:\ai_module\
├── streamlit_app.py          ← NEW FILE (put here!)
├── api.py                    ← Already have
├── inference.py              ← Already have
├── ai_module/                ← Folder with your code
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── model_lstm.py
│   ├── model_ncf.py
│   ├── model_autoencoder.py
│   ├── evaluate.py
│   └── saved_models/         ← Your trained models here!
└── saved_models/             ← Models should be here
    ├── autoencoder.keras
    ├── ncf.keras
    ├── lstm.keras
    └── mappings.pkl
```

---

## STEP 3: RUN THE APP (1 minute)

```bash
# In terminal, in E:\ai_module folder:
streamlit run streamlit_app.py
```

**What happens:**
1. ✅ Streamlit starts
2. ✅ Browser opens automatically (http://localhost:8501)
3. ✅ App loads with interface
4. ✅ You see the web interface!

---

## STEP 4: TEST IT (2 minutes)

In the web interface:
1. Default customer ID is "17850" (keep it)
2. Default items are "85123A,71053" (keep it)
3. Click "🚀 Get Recommendations"
4. **See the results appear!**

---

## FOR THE PRESENTATION

### BEFORE CLASS:

```bash
# Terminal: Start the app
cd E:\ai_module
streamlit run streamlit_app.py

# Browser opens automatically!
# Keep this terminal open
```

### DURING PRESENTATION:

1. **Show Slides (5 min)**
   - Beamer presentation slides 1-8

2. **Show App Demo (7 min)**
   - Browser shows the Streamlit interface
   - Enter customer ID → Get recommendations
   - Show different examples
   - Show model comparison charts

3. **Show More Slides (5 min)**
   - Slides 9-14 (lessons, conclusion)

4. **Q&A (3 min)**
   - Answer questions about the system

---

## TROUBLESHOOTING

### Problem: "Module not found"
```
Error: ModuleNotFoundError: No module named 'inference'
```

**Solution:**
Make sure `inference.py` is in the same folder as `streamlit_app.py`

### Problem: "Models not found"
```
Error: FileNotFoundError: saved_models/lstm.keras
```

**Solution:**
Make sure you have trained the models:
```bash
python train_pipeline.py
```

### Problem: Streamlit not installing
```bash
# Try:
pip install --upgrade pip
pip install streamlit
```

### Problem: Port 8501 already in use
```bash
# Use different port:
streamlit run streamlit_app.py --server.port 8502
```

---

## WHAT THE PROFESSOR SEES

### The App Interface:

```
╔═══════════════════════════════════════════════════════════════╗
║  🎯 Intelligent Product Recommendation System                 ║
║     Ensemble Deep Learning (AutoEncoder + NCF + LSTM)        ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  🛍️ Get Product Recommendations                              ║
║                                                               ║
║  Input Section:                                               ║
║  ┌─────────────────────────────────────────────────────────┐ ║
║  │ Customer ID: [17850]                                    │ ║
║  │ Previously Purchased: [85123A, 71053]                   │ ║
║  │ Number of Recs: [5]                                     │ ║
║  │                                                         │ ║
║  │ [🚀 Get Recommendations] (button)                       │ ║
║  └─────────────────────────────────────────────────────────┘ ║
║                                                               ║
║  Results Section:                                             ║
║  ┌─────────────────────────────────────────────────────────┐ ║
║  │ ✅ Recommendations generated successfully!              │ ║
║  │                                                         │ ║
║  │ 📋 Recommendations | Strategy: ensemble_...            │ ║
║  │                                                         │ ║
║  │ 1. REGENCY CAKESTAND 3 TIER                  0.320 ▓▓▓│ ║
║  │    ID: 22423                                           │ ║
║  │                                                         │ ║
║  │ 2. REX CASH+CARRY JUMBO SHOPPER             0.318 ▓▓▓│ ║
║  │    ID: 21034                                           │ ║
║  │                                                         │ ║
║  │ [More recommendations...]                              │ ║
║  └─────────────────────────────────────────────────────────┘ ║
║                                                               ║
║  📊 Model Performance                                         ║
║  ┌─────────────────────────────────────────────────────────┐ ║
║  │ LSTM: 32.23%  Ensemble: 31.82%  NCF: 17.20%  AE: 2.71% │ ║
║  │                                                         │ ║
║  │ [Bar chart showing comparison]                         │ ║
║  └─────────────────────────────────────────────────────────┘ ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## PRESENTATION SCRIPT

```
ÉTAPE 1 - Show Slides (0-5 min)
─────────────────────────────
"Voici notre système de recommandation basé sur 3 modèles ML.
 Données: 397K transactions, 4K clients, 3.6K produits.
 Résultat: 32% precision - 6.4x mieux que le hasard!

 [Show slides 1-8]"

ÉTAPE 2 - Show Streamlit App (5-12 min)
─────────────────────────────────────
"Maintenant je vais montrer le système en action.

 [Click browser with Streamlit]
 
 Voyez l'interface. Entrez un customer ID et cliquez 'Get Recommendations'.
 
 [Enter: 17850]
 [Click: Get Recommendations]
 
 Regardez! Les recommendations apparaissent:
 - CAKSTAND (score 0.320)
 - SHOPPER (score 0.318)
 
 Cela montre:
 ✓ Le système fonctionne en temps réel
 ✓ Les prédictions sont basées sur les modèles ML
 ✓ LSTM capture les patterns temporels
 
 [Try another customer]
 
 Vous voyez aussi la comparaison des modèles:
 - LSTM: 32.23% (meilleur)
 - Ensemble: 31.82% (presque identique)
 - NCF: 17.2%
 - AutoEncoder: 2.71%"

ÉTAPE 3 - Show More Slides (12-18 min)
──────────────────────────────────
"Voici ce que nous avons appris:
 - LSTM captures temporal patterns
 - Ensemble fusion works well
 - System is 6.4x better than random
 
 [Show slides 9-14]"

ÉTAPE 4 - Q&A (18-20 min)
─────────────────
"Questions?"
```

---

## TIPS FOR DEMO

### Before Presentation:
```
1. Test the app once: streamlit run streamlit_app.py
2. Try different customer IDs
3. Make sure it responds quickly
4. Close the app: Press Ctrl+C
```

### During Presentation:
```
1. Start app: streamlit run streamlit_app.py
2. Let browser open
3. Wait for "App is ready" message
4. Then start your presentation
5. At demo time, switch to browser and show interface
6. After demo, switch back to slides
```

### Example Customer IDs to Try:
```
17850  → Lighting/Home items
15168  → Home décor items
12792  → Party supplies
18150  → Chalkboards/Kitchen items
```

---

## SUCCESS CRITERIA

The app is working correctly if:

✅ App starts without errors
✅ Interface loads in browser
✅ You can enter customer ID
✅ "Get Recommendations" button works
✅ Recommendations appear with scores
✅ Model comparison chart shows
✅ Response time is <1 second

---

## WHAT'S INSIDE THE APP

The Streamlit app includes:

✅ **Input Section**
   - Customer ID field
   - Previously purchased items
   - Number of recommendations slider

✅ **Results Section**
   - Recommendations with scores
   - Progress bars
   - Product descriptions

✅ **Model Performance Section**
   - Metric cards (LSTM, Ensemble, NCF, AutoEncoder)
   - Bar chart comparison
   - Performance metrics

✅ **System Information Sidebar**
   - Model status
   - Dataset statistics
   - Performance metrics

✅ **Help/Info Sections**
   - How it works (expandable)
   - Model details (expandable)
   - Evaluation metrics (expandable)

---

## TOTAL TIME TO SETUP

```
Install Streamlit:    2 minutes
Copy file:           1 minute
Test app:            2 minutes
Practice demo:       5 minutes

TOTAL:               10 minutes!
```

---

## YOU'RE READY! 🚀

1. Copy `streamlit_app.py` to `E:\ai_module\`
2. Run: `streamlit run streamlit_app.py`
3. Browser opens → App loads
4. Click "Get Recommendations"
5. See results appear!

**C'est tout!** The app is production-ready and impressive! 💪
