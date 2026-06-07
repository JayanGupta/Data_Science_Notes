import os
import json
import traceback
import sys

# Crucial step: Use headless Agg backend to prevent matplotlib from trying to open windows
import matplotlib
matplotlib.use('Agg')

notebooks = [
    "Linear_Regression.ipynb",
    "Ridge_Regression.ipynb",
    "Lasso_Regression.ipynb",
    "Logistic_Regression.ipynb",
    "Decision_Tree.ipynb",
    "KNN.ipynb",
    "SVM.ipynb",
    "Naive_Bayes.ipynb",
    "Random_Forest.ipynb",
    "AdaBoost.ipynb",
    "XGBoost.ipynb",
    "K_Means.ipynb",
    "PCA.ipynb"
]

models_dir = "models"
all_passed = True

print("=== Starting Jupyter Notebook Execution Tests ===")

for nb in notebooks:
    filepath = os.path.join(models_dir, nb)
    print(f"\nTesting: {nb}...")
    
    if not os.path.exists(filepath):
        print(f"[-] File not found: {filepath}")
        all_passed = False
        continue
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb_data = json.load(f)
            
        # We will execute all code cells in a single global/local namespace to simulate cell execution order
        namespace = {}
        
        # Execute standard imports in namespace first
        code_cells = [cell for cell in nb_data.get("cells", []) if cell.get("cell_type") == "code"]
        
        print(f"Found {len(code_cells)} code cells. Executing...")
        
        cell_index = 0
        for cell in code_cells:
            cell_index += 1
            source = "".join(cell.get("source", []))
            
            # Skip empty cells
            if not source.strip():
                continue
                
            try:
                # Execute the cell source in our simulated notebook environment
                exec(source, namespace)
            except Exception as cell_err:
                print(f"\n[ERROR] Exception in {nb} at code cell #{cell_index}:")
                print("-" * 50)
                print(source)
                print("-" * 50)
                traceback.print_exc(file=sys.stdout)
                print("-" * 50)
                raise cell_err
                
        print(f"[SUCCESS] {nb} executed completely with no errors!")
        
    except Exception as e:
        print(f"[-] {nb} failed execution check.")
        all_passed = False

print("\n" + "=" * 50)
if all_passed:
    print("[FINAL RESULT] All 13 notebooks executed with ZERO errors!")
else:
    print("[FINAL RESULT] One or more notebooks failed execution. See errors above.")
print("=" * 50)
