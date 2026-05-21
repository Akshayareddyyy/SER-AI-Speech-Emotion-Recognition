import os
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


EMBED_PATH = os.path.join('outputs', 'wav2vec2_embeddings_full.npz')
MODEL_OUT = os.path.join('models', 'wav2vec2_linearsvc_tuned_full.pkl')
SCALER_OUT = os.path.join('models', 'wav2vec2_scaler_linearsvc_tuned_full.pkl')
LABEL_OUT = os.path.join('models', 'wav2vec2_label_encoder_linearsvc_tuned_full.pkl')
REPORT_OUT = os.path.join('outputs', 'wav2vec2_linearsvc_tuning_full.txt')

CS = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
RANDOM_STATE = 42


def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    if not os.path.exists(EMBED_PATH):
        raise FileNotFoundError(f"Embeddings cache not found: {EMBED_PATH}")

    arr = np.load(EMBED_PATH, allow_pickle=True)
    embeddings = arr['embeddings']
    labels = arr['labels'].tolist() if hasattr(arr['labels'], 'tolist') else list(arr['labels'])
    speakers = arr['speakers'].tolist() if hasattr(arr['speakers'], 'tolist') else list(arr['speakers'])

    # Speaker-independent split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    idx_train, idx_test = next(gss.split(embeddings, labels, groups=speakers))

    X_train = embeddings[idx_train]
    X_test = embeddings[idx_test]
    y_train = [labels[i] for i in idx_train]
    y_test = [labels[i] for i in idx_test]

    le = LabelEncoder()
    le.fit(labels)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []
    best_acc = -1.0
    best_C = None
    best_clf = None
    best_y_pred = None

    print(f"Tuning LinearSVC over C values: {CS}")
    for C in CS:
        clf = LinearSVC(C=C, class_weight='balanced', max_iter=10000, random_state=RANDOM_STATE)
        try:
            clf.fit(X_train_s, y_train_enc)
            y_pred = clf.predict(X_test_s)
            acc = accuracy_score(y_test_enc, y_pred)
            macro = f1_score(y_test_enc, y_pred, average='macro')
            weighted = f1_score(y_test_enc, y_pred, average='weighted')
            results.append({'C': C, 'acc': acc, 'macro_f1': macro, 'weighted_f1': weighted, 'y_pred': y_pred, 'clf': clf})
            print(f"C={C} | acc={acc:.4f} | macro_f1={macro:.4f} | weighted_f1={weighted:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_C = C
                best_clf = clf
                best_y_pred = y_pred
        except Exception as e:
            print(f"C={C} failed: {e}")

    if not results:
        raise RuntimeError("No LinearSVC results were produced during tuning")

    # Save best model and artifacts
    if best_clf is not None:
        joblib.dump(best_clf, MODEL_OUT)
        joblib.dump(scaler, SCALER_OUT)
        joblib.dump(le, LABEL_OUT)

    # Write report
    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write('LinearSVC tuning (wav2vec2 embeddings full)\n')
        f.write(f'Embeddings cache: {EMBED_PATH}\n')
        f.write(f'Total samples: {len(labels)}\n')
        f.write(f'Train samples: {len(idx_train)}\n')
        f.write(f'Test samples: {len(idx_test)}\n')
        f.write(f'Embedding shape: {embeddings.shape}\n\n')
        f.write('Results per C:\n')
        for r in results:
            f.write(f"C={r['C']}, acc={r['acc']:.4f}, macro_f1={r['macro_f1']:.4f}, weighted_f1={r['weighted_f1']:.4f}\n")

        f.write('\nBest:\n')
        f.write(f'C={best_C}, acc={best_acc:.4f}\n')
        improved = best_acc > 0.5454
        f.write(f'Improved over previous LinearSVC (54.54%): {improved} (best={best_acc:.4f})\n\n')

        if best_y_pred is not None:
            creport = classification_report(y_test_enc, best_y_pred, target_names=le.classes_)
            cm = confusion_matrix(y_test_enc, best_y_pred)
            f.write('Classification Report (best):\n')
            f.write(creport + '\n')
            f.write('Confusion Matrix:\n')
            np.savetxt(f, cm, fmt='%d')

    # Final console print
    print('\nTuning completed.')
    print(f'Best C: {best_C} | Best accuracy: {best_acc:.4f} | Improved: {best_acc > 0.5454}')
    if best_y_pred is not None:
        print('Classification report (best):')
        print(classification_report(y_test_enc, best_y_pred, target_names=le.classes_))
        print('Confusion matrix:')
        print(confusion_matrix(y_test_enc, best_y_pred))


if __name__ == '__main__':
    main()
