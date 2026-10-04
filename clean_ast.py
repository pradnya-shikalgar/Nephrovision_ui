import ast

with open('F:/Capstone_Kidney_Project/extracted_code.py', 'r', encoding='utf-8') as f:
    source = f.read()

tree = ast.parse(source)

clean_source = ""
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
        clean_source += ast.unparse(node) + "\n\n"

with open('F:/Capstone_Kidney_Project/backend/model_def.py', 'w', encoding='utf-8') as f:
    f.write(clean_source)

print("Clean model definitions extracted!")
