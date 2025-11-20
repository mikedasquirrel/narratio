#!/bin/bash
# Process remaining 6 domains sequentially with genome features

cd /Users/michaelsmerconish/Desktop/RandomCode/novelization

echo "======================================================================"
echo "PROCESSING REMAINING DOMAINS WITH GENOME FEATURES"
echo "======================================================================"
echo "Started: $(date)"
echo ""

# Array of domains to process
domains=("nfl:3000" "nba:5000" "tennis:5000" "golf:5000" "movies:2000" "startups:258")

for domain_spec in "${domains[@]}"; do
  IFS=':' read -r domain size <<< "$domain_spec"
  
  echo ""
  echo "======================================================================"
  echo "PROCESSING: $domain ($size samples)"
  echo "======================================================================"
  echo ""
  
  python3 -c "
import sys
sys.path.insert(0, 'narrative_optimization')
from universal_domain_processor import UniversalDomainProcessor

processor = UniversalDomainProcessor(
    results_dir='narrative_optimization/results/domains_genome',
    use_transformers=True,
    fast_mode=False,
    enable_cross_domain=True
)

result = processor.process_domain('$domain', sample_size=$size, save_results=True)
print(f'✓ $domain complete: {result.get(\"n_patterns\")} patterns')
"
  
  if [ $? -eq 0 ]; then
    echo "✅ $domain completed successfully"
  else
    echo "❌ $domain failed"
  fi
done

echo ""
echo "======================================================================"
echo "BATCH PROCESSING COMPLETE"
echo "======================================================================"
echo "Completed: $(date)"
ls -lh narrative_optimization/results/domains_genome/*/*.json 2>/dev/null | wc -l

