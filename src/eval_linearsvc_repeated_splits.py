import os
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import statistics

EMBED_PATH = os.path.join('outputs', 'wav2vec2_embeddings_full.npz')
REPORT_OUT = os.path.join('outputs', 'wav2vec2_repeated_splits_report.txt')

RANDOM_STATES = [21, 42, 84, 123, 202]
TEST_SIZE = 0.2
RANDOM_STATE_SEED = 0


def run_once(embeddings, labels, speakers, rs):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=rs)
    idx_train, idx_test = next(gss.split(embeddings, labels, groups=speakers))

    X_train = embeddings[idx_train]
    X_test = embeddings[idx_test]
    y_train = [labels[i] for i in idx_train]
    y_test = [labels[i] for i in idx_test]

    le = LabelEncoder(); le.fit(labels)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler(); scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LinearSVC(C=0.01, class_weight='balanced', max_iter=10000, random_state=RANDOM_STATE_SEED)
    clf.fit(X_train_s, y_train_enc)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test_enc, y_pred)
    macro = f1_score(y_test_enc, y_pred, average='macro')
    weighted = f1_score(y_test_enc, y_pred, average='weighted')

    return {
        'random_state': rs,
        'acc': acc,
        'macro_f1': macro,
        'weighted_f1': weighted,
        'y_test_enc': y_test_enc,
        'y_pred': y_pred,
        'label_encoder': le,
        'confusion_matrix': confusion_matrix(y_test_enc, y_pred),
        'classification_report': classification_report(y_test_enc, y_pred, target_names=le.classes_)
    }


def main():
    if not os.path.exists(EMBED_PATH):
        raise FileNotFoundError(f"Embeddings cache not found: {EMBED_PATH}")

    arr = np.load(EMBED_PATH, allow_pickle=True)
    embeddings = arr['embeddings']
    labels = arr['labels'].tolist() if hasattr(arr['labels'], 'tolist') else list(arr['labels'])
    speakers = arr['speakers'].tolist() if hasattr(arr['speakers'], 'tolist') else list(arr['speakers'])

    results = []
    for rs in RANDOM_STATES:
        res = run_once(embeddings, labels, speakers, rs)
        results.append(res)
        print(f"RS={rs} | acc={res['acc']:.4f} | macro_f1={res['macro_f1']:.4f} | weighted_f1={res['weighted_f1']:.4f}")

    accs = [r['acc'] for r in results]
    macros = [r['macro_f1'] for r in results]
    mean_acc = statistics.mean(accs)
    std_acc = statistics.pstdev(accs)
    mean_macro = statistics.mean(macros)
    std_macro = statistics.pstdev(macros)

    best_idx = int(np.argmax(accs))
    worst_idx = int(np.argmin(accs))

    # Write report
    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write('Repeated speaker-independent splits evaluation\n')
        f.write(f'Embeddings cache: {EMBED_PATH}\n')
        f.write(f'Random states: {RANDOM_STATES}\n')
        f.write(f'Test size: {TEST_SIZE}\n')
        f.write(f'Model: LinearSVC(C=0.01, class_weight="balanced", max_iter=10000)\n')
        f.write(f'Total samples: {len(labels)}\n')
        f.write(f'Embedding shape: {embeddings.shape}\n\n')

        f.write('Per-split results:\n')
        for r in results:
            f.write(f"RS={r['random_state']}, acc={r['acc']:.4f}, macro_f1={r['macro_f1']:.4f}, weighted_f1={r['weighted_f1']:.4f}\n")
        f.write('\nSummary:\n')
        f.write(f'Mean accuracy: {mean_acc:.4f}\n')
        f.write(f'Std accuracy: {std_acc:.4f}\n')
        f.write(f'Mean macro F1: {mean_macro:.4f}\n')
        f.write(f'Std macro F1: {std_macro:.4f}\n')
        f.write(f'Best split: RS={results[best_idx]["random_state"]} (acc={results[best_idx]["acc"]:.4f})\n')
        f.write(f'Worst split: RS={results[worst_idx]["random_state"]} (acc={results[worst_idx]["acc"]:.4f})\n\n')

        f.write('Best split classification report:\n')
        f.write(results[best_idx]['classification_report'] + '\n')
        f.write('Best split confusion matrix:\n')
        np.savetxt(f, results[best_idx]['confusion_matrix'], fmt='%d')

    # Final console summary
    print('\nRepeated splits completed.')
    print(f'Mean acc={mean_acc:.4f} | Std acc={std_acc:.4f} | Mean macro={mean_macro:.4f} | Std macro={std_macro:.4f}')
    print(f'Best RS={results[best_idx]["random_state"]} acc={results[best_idx]["acc"]:.4f}')


if __name__ == '__main__':
    main()
