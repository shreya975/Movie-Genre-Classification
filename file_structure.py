import os

structure = {
    "movie-genre-classification": {
        "data": {
            "raw": {},
            "processed": {}
        },
        "notebooks": {
            "01_eda.ipynb": "",
            "02_baseline_model.ipynb": "",
            "03_bert_experiments.ipynb": ""
        },
        "src": {
            "data_loader.py": "",
            "preprocess.py": "",
            "feature_engineering.py": "",
            "train_ml.py": "",
            "train_bert.py": "",
            "evaluate.py": "",
            "predict.py": ""
        },
        "models": {
            "tfidf_vectorizer.pkl": "",
            "best_ml_model.pkl": "",
            "bert_model": {}
        },
        "reports": {
            "confusion_matrix.png": "",
            "classification_report.txt": ""
        },
        "app": {
            "app.py": ""
        },
        "requirements.txt": "",
        "Dockerfile": "",
        "README.md": "",
        ".gitignore": ""
    }
}

def create_structure(base_path, data):
    for name, content in data.items():
        path = os.path.join(base_path, name)

        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, "w") as f:
                f.write("")

create_structure(".", structure)
print("Folder structure created successfully!")