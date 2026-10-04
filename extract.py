import json

with open('F:/Capstone_Kidney_Project/Copy_of_Capstone_for_report.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

with open('F:/Capstone_Kidney_Project/extracted_code.py', 'w', encoding='utf-8') as f:
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            f.write(source + '\n\n# --- End of Cell ---\n\n')
print("Extracted all code cells to extracted_code.py")
