from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report


def read_ner_data(file_path):
    sentences, labels = [], []
    cur_sent, cur_labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.strip() == '':
                if cur_sent:
                    sentences.append(cur_sent)
                    labels.append(cur_labels)
                    cur_sent, cur_labels = [], []
            else:
                parts = line.split()
                if len(parts) == 2:
                    # CoNLL format: word label
                    cur_sent.append(parts[0])
                    cur_labels.append(parts[1])
                elif len(parts) == 1 and cur_sent:
                    # fallback: label-only line
                    cur_labels.append(parts[0])
    if cur_sent:
        sentences.append(cur_sent)
        labels.append(cur_labels)
    return sentences, labels


def prepare_ner_data(data):
    # data is already (sentences, labels) tuple from read_ner_data
    return data


def word2features(sent, i):
    word = sent[i]
    features = {
        'bias':           1.0,
        'word.lower()':   word.lower(),
        'word[-3:]':      word[-3:],
        'word[-2:]':      word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.len':       len(word),
    }

    if i > 0:
        prev = sent[i - 1]
        features.update({
            '-1:word.lower()':   prev.lower(),
            '-1:word.istitle()': prev.istitle(),
            '-1:word.isupper()': prev.isupper(),
            '-1:word.isdigit()': prev.isdigit(),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        nxt = sent[i + 1]
        features.update({
            '+1:word.lower()':   nxt.lower(),
            '+1:word.istitle()': nxt.istitle(),
            '+1:word.isupper()': nxt.isupper(),
            '+1:word.isdigit()': nxt.isdigit(),
        })
    else:
        features['EOS'] = True

    if i > 1:
        prev2 = sent[i - 2]
        features.update({
            '-2:word.lower()':   prev2.lower(),
            '-2:word.istitle()': prev2.istitle(),
        })

    if i < len(sent) - 2:
        nxt2 = sent[i + 2]
        features.update({
            '+2:word.lower()':   nxt2.lower(),
            '+2:word.istitle()': nxt2.istitle(),
        })

    return {k: str(v) for k, v in features.items()}


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


train_data = read_ner_data('train.txt')
test_data  = read_ner_data('test.txt')
val_data   = read_ner_data('val.txt')

X_train, y_train = train_data
X_test,  y_test  = test_data
X_val,   y_val   = val_data

X_train = [sent2features(s) for s in X_train]
X_test  = [sent2features(s) for s in X_test]
X_val   = [sent2features(s) for s in X_val]

crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

y_val_pred  = crf.predict(X_val)
y_test_pred = crf.predict(X_test)

print("--- Validation Set Report ---")
print(flat_classification_report(y_val, y_val_pred, digits=4))

print("--- Test Set Report ---")
print(flat_classification_report(y_test, y_test_pred, digits=4))

print("Top 10 most likely transitions:")
for (label_from, label_to), weight in sorted(
    crf.transition_features_.items(), key=lambda x: -x[1]
)[:10]:
    print(f"  {label_from:>10} -> {label_to:<10}  {weight:.4f}")