import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    words = jieba.cut(text)
    stopwords = set()
    words = [w for w in words if w not in stopwords]
    return " ".join(words)


def read_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data, labels = [], []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            data.append(parts[0])
            labels.append(int(parts[1]))
    return data, labels


with open("class.txt", "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

train_texts, train_labels = read_dataset("train.txt")
val_texts,   val_labels   = read_dataset("val.txt")
test_texts,  test_labels  = read_dataset("test.txt")

train_texts = [preprocess(t) for t in train_texts]
val_texts   = [preprocess(t) for t in val_texts]
test_texts  = [preprocess(t) for t in test_texts]

tfidf = TfidfVectorizer()
tfidf.fit(train_texts + val_texts)

X_train = tfidf.transform(train_texts)
X_val   = tfidf.transform(val_texts)
X_test  = tfidf.transform(test_texts)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":         MultinomialNB(),
    "MLP":                 MLPClassifier(hidden_layer_sizes=(100,), max_iter=300),
}

results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, train_labels)

    val_pred  = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    val_acc  = accuracy_score(val_labels,  val_pred)
    test_acc = accuracy_score(test_labels, test_pred)
    results[name] = {"val_acc": val_acc, "test_acc": test_acc}

    print(f"\n[{name}]")
    print(f"  Validation Accuracy : {val_acc:.4f}")
    print(f"  Test Accuracy       : {test_acc:.4f}")
    print(f"\n  Validation Classification Report:")
    print(classification_report(val_labels, val_pred, target_names=classes, zero_division=0))
    print(f"  Test Classification Report:")
    print(classification_report(test_labels, test_pred, target_names=classes, zero_division=0))

print("\n--- Summary ---")
print(f"{'Classifier':<25} {'Val Acc':>10} {'Test Acc':>10}")
for name, res in results.items():
    print(f"{name:<25} {res['val_acc']:>10.4f} {res['test_acc']:>10.4f}")