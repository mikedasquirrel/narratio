# Get Real UFC Data from Kaggle

## Required: Kaggle API Credentials

To pull the UFC scraper kernel, you need Kaggle API credentials.

### Setup Steps

1. **Go to Kaggle.com and login**
   - Visit: https://www.kaggle.com/

2. **Get API Token**
   - Go to: https://www.kaggle.com/settings/account
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json`

3. **Place credentials**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Pull UFC data**
   ```bash
   cd /Users/michaelsmerconish/Desktop/RandomCode/novelization
   kaggle kernels pull asaniczka/scraper-to-scrape-ufcstats-com
   ```

This will download the UFC scraper notebook which contains or can generate real UFC data.

## Alternative: Direct Dataset Download

Or download the dataset directly:
```bash
kaggle datasets download -d mdabbert/ultimate-ufc-dataset
unzip ultimate-ufc-dataset.zip -d data/domains/
```

## After Getting Data

Once you have real UFC data:
```bash
# Run rigorous analysis
python3 narrative_optimization/domains/ufc/analyze_ufc_rigorous.py

# Test residual effects
python3 narrative_optimization/domains/ufc/test_narrative_residuals.py
```

This will give us REAL correlations with REAL fight outcomes!

