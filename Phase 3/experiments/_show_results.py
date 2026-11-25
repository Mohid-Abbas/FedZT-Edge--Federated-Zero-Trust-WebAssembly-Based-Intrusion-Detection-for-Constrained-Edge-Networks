import json, sys
sys.path.insert(0, '.')

with open('results/raw_logs/baselines_unsw_nb15.json') as f: unsw = json.load(f)
with open('results/raw_logs/baselines_cicids2017.json') as f: cicids = json.load(f)
with open('results/raw_logs/adversarial_robustness.json') as f: adv = json.load(f)
with open('results/raw_logs/heterogeneity_sweep.json') as f: het = json.load(f)

print('UNSW-NB15 Live Results:')
for k,v in unsw.items():
    print('  %-24s F1=%.4f Acc=%.4f AUC=%.4f' % (k, v['f1'], v['accuracy'], v['roc_auc']))

print()
print('CICIDS-2017 Live Results:')
for k,v in cicids.items():
    print('  %-24s F1=%.4f Acc=%.4f AUC=%.4f' % (k, v['f1'], v['accuracy'], v['roc_auc']))

print()
print('Adversarial (FedZT vs NoDef F1):')
for i, r in enumerate(adv['rates']):
    print('  rate=%.1f  FedZT=%.4f  NoDef=%.4f  FedProx=%.4f' % (
        r, adv['FedZT_Edge'][i], adv['B2_no_defense'][i], adv['B3_FedProx'][i]))
