import os
import json

notebooks = [
    "AdaBoost.ipynb",
    "Decision_Tree.ipynb",
    "KNN.ipynb",
    "K_Means.ipynb",
    "Linear_Regression.ipynb",
    "Logistic_Regression.ipynb",
    "Naive_Bayes.ipynb",
    "PCA.ipynb",
    "Random_Forest.ipynb",
    "SVM.ipynb",
    "XGBoost.ipynb"
]

models_dir = "models"
all_passed = True

for nb in notebooks:
    filepath = os.path.join(models_dir, nb)
    if not os.path.exists(filepath):
        print(f"[-] Missing file: {filepath}")
        all_passed = False
        continue
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Verify structure
        if "cells" not in data or "metadata" not in data or "nbformat" not in data:
            print(f"[-] Invalid notebook format in {nb}: missing top-level keys")
            all_passed = False
            continue
            
        cell_types = [c.get("cell_type") for c in data["cells"]]
        if not cell_types:
            print(f"[-] Notebook {nb} is empty")
            all_passed = False
            continue
            
        print(f"[+] {nb} loaded successfully. Found {len(data['cells'])} cells (Markdown: {cell_types.count('markdown')}, Code: {cell_types.count('code')})")
        
    except json.JSONDecodeError as e:
        print(f"[-] JSON decode failed for {nb}: {e}")
        all_passed = False
    except Exception as e:
        print(f"[-] Error reading {nb}: {e}")
        all_passed = False

if all_passed:
    print("\n[SUCCESS] All notebooks passed standard checks!")
else:
    print("\n[FAILURE] One or more notebooks failed checks.")
