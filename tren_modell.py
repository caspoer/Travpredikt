import ml_model, json

def log(msg):
    print(msg)

print("=== TRENER FORBEDRET MODELL ===")
meta = ml_model.train(log_fn=log)

print()
print("=== RESULTAT ===")
mtype = meta["model_type"]
tr    = meta["train_auc"]
te    = meta["test_auc"]
gap   = tr - te
gbm   = meta["gbm_test_auc"]
lr    = meta["lr_test_auc"]
sd    = meta["split_date"]

print(f"Modell:     {mtype}")
print(f"Train AUC:  {tr}")
print(f"Test AUC:   {te}")
print(f"Gap:        {gap:.3f}")
print(f"GBM test:   {gbm}")
print(f"LR test:    {lr}")
print(f"Split dato: {sd}")
print()
print("Feature-viktighet:")
imps = sorted(meta["importances"].items(), key=lambda x: -x[1])
for k, v in imps:
    bar = "#" * int(v * 40)
    print(f"  {k:<22} {v*100:5.1f}%  {bar}")
