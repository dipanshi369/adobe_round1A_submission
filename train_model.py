import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import joblib
import os


os.makedirs("app/model", exist_ok=True)


with open("training_data.json", "r", encoding="utf-8") as f:
    training_examples = json.load(f)


embedder = SentenceTransformer("all-MiniLM-L6-v2")


texts = []
font_sizes = []
y0s = []
labels = []


for example in training_examples:
    example_data = example["data"]
    for item in example_data:
        texts.append(item["text"])
        font_sizes.append(item["font_size"])
        y0s.append(item["y0"])
        labels.append(item["label"])

print(f"📊 Total training samples: {len(texts)}")
print(f"📊 Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")


print("🔄 Generating text embeddings...")
text_embeddings = embedder.encode(texts)


layout_features = np.array([[fs, y] for fs, y in zip(font_sizes, y0s)])
X = np.hstack([text_embeddings, layout_features])
y = np.array(labels)

print(f"✅ Feature matrix shape: {X.shape}")
print(f"✅ Labels shape: {y.shape}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📈 Training samples: {len(y_train)}")
print(f"📈 Test samples: {len(y_test)}")


print("🔄 Training classifier...")
clf = LogisticRegression(
    max_iter=1000, 
    multi_class="multinomial", 
    solver="lbfgs",
    class_weight="balanced"  # Handle potential class imbalance
)
clf.fit(X_train, y_train)


print("🔄 Evaluating model...")
y_pred = clf.predict(X_test)
unique_labels = sorted(list(set(y)))
report = classification_report(y_test, y_pred, target_names=unique_labels, output_dict=True)

print("\n📊 Classification Report:")
print(json.dumps({k: v for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}, indent=2))
print(f"\n🎯 Overall Accuracy: {report['accuracy']:.3f}")
print(f"🎯 Macro F1-Score: {report['macro avg']['f1-score']:.3f}")
print(f"🎯 Weighted F1-Score: {report['weighted avg']['f1-score']:.3f}")


feature_names = [f"text_emb_{i}" for i in range(text_embeddings.shape[1])] + ["font_size", "y_position"]
print(f"\n📋 Most important features for classification:")
for class_idx, class_name in enumerate(clf.classes_):
    top_features = np.argsort(np.abs(clf.coef_[class_idx]))[-5:][::-1]
    print(f"  {class_name}: {[feature_names[i] for i in top_features]}")


evaluation_data = {
    "classification_report": report,
    "test_samples": len(y_test),
    "training_samples": len(y_train),
    "total_examples": len(training_examples),
    "feature_dimensions": X.shape[1],
    "classes": list(clf.classes_),
    "model_parameters": {
        "max_iter": 1000,
        "multi_class": "multinomial",
        "solver": "lbfgs",
        "class_weight": "balanced"
    }
}

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_data, f, indent=2)
print("✅ Evaluation results saved to evaluation_results.json")


model_data = {
    "classifier": clf,
    "embedder_name": "all-MiniLM-L6-v2",
    "feature_names": feature_names,
    "classes": list(clf.classes_),
    "embedding_dim": text_embeddings.shape[1]
}

joblib.dump(model_data, "app/model/heading_classifier.joblib")
print("✅ Model saved to app/model/heading_classifier.joblib")


inference_config = {
    "embedder_model": "all-MiniLM-L6-v2",
    "feature_order": ["text_embeddings", "font_size", "y_position"],
    "classes": list(clf.classes_),
    "embedding_dimension": text_embeddings.shape[1],
    "usage_example": {
        "input_format": {
            "text": "Chapter 1: Introduction",
            "font_size": 22,
            "y0": 95
        },
        "output_format": {
            "predicted_label": "H1",
            "confidence": 0.95
        }
    }
}

with open("app/model/inference_config.json", "w", encoding="utf-8") as f:
    json.dump(inference_config, f, indent=2)
print("✅ Inference configuration saved to app/model/inference_config.json")

print("\n🎉 Training completed successfully!")
print(f"📁 Model files saved in: app/model/")
print(f"📈 Final model accuracy: {report['accuracy']:.1%}")
