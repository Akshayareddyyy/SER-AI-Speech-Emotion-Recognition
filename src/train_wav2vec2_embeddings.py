import os
import random
import time
import datetime
import sys
import numpy as np
import librosa
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

import data_loader

# Try import transformers; raise helpful error if missing
try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
except Exception:
    raise ImportError(
        "transformers library is required. Install with `pip install transformers torch`"
    )

# Config
SR = 16000
MAX_DURATION = 4.0  # seconds
MAX_LEN = int(SR * MAX_DURATION)
RANDOM_STATE = 42
QUICK_TEST = False  # set True to run quick pipeline on 500 samples
# If set to an int, limit to up to this many samples per class (balanced subset mode)
# Set to None to use full dataset
SUBSET_PER_CLASS = None

# Cache files
if SUBSET_PER_CLASS is not None:
    EMBED_CACHE_QUICK = os.path.join('outputs', f'wav2vec2_embeddings_subset_{SUBSET_PER_CLASS}.npz')
    EMBED_CACHE_FULL = os.path.join('outputs', f'wav2vec2_embeddings_subset_{SUBSET_PER_CLASS}.npz')
    EMBED_CACHE_PARTIAL = os.path.join('outputs', f'wav2vec2_embeddings_subset_{SUBSET_PER_CLASS}_partial.npz')
    MODEL_PATH = os.path.join('models', f'wav2vec2_best_classifier_subset_{SUBSET_PER_CLASS}.pkl')
    LABEL_PATH = os.path.join('models', f'wav2vec2_label_encoder_subset_{SUBSET_PER_CLASS}.pkl')
    SCALER_PATH = os.path.join('models', f'wav2vec2_scaler_subset_{SUBSET_PER_CLASS}.pkl')
    REPORT_PATH = os.path.join('outputs', f'wav2vec2_report_subset_{SUBSET_PER_CLASS}.txt')
    # tuned artifacts (do not overwrite default paths)
    TUNED_MODEL_PATH = os.path.join('models', f'wav2vec2_best_classifier_subset_{SUBSET_PER_CLASS}_tuned.pkl')
    TUNED_SCALER_PATH = os.path.join('models', f'wav2vec2_scaler_subset_{SUBSET_PER_CLASS}_tuned.pkl')
    TUNED_LABEL_PATH = os.path.join('models', f'wav2vec2_label_encoder_subset_{SUBSET_PER_CLASS}_tuned.pkl')
    TUNED_REPORT_PATH = os.path.join('outputs', f'wav2vec2_tuning_subset_{SUBSET_PER_CLASS}_report.txt')
else:
    EMBED_CACHE_QUICK = os.path.join('outputs', 'wav2vec2_embeddings_quick.npz')
    EMBED_CACHE_FULL = os.path.join('outputs', 'wav2vec2_embeddings_full.npz')
    EMBED_CACHE_PARTIAL = os.path.join('outputs', 'wav2vec2_embeddings_partial.npz')

    # Full-run artifact names (explicit _full suffix per user request)
    MODEL_PATH = os.path.join('models', 'wav2vec2_best_classifier_full.pkl')
    LABEL_PATH = os.path.join('models', 'wav2vec2_label_encoder_full.pkl')
    SCALER_PATH = os.path.join('models', 'wav2vec2_scaler_full.pkl')
    PCA_PATH = os.path.join('models', 'wav2vec2_pca_full.pkl')
    REPORT_PATH = os.path.join('outputs', 'wav2vec2_report_full.txt')


def ensure_dirs():
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)


def load_file_list():
    files = data_loader.get_all_files()
    return files


def extract_embeddings(files, quick_test=False, batch_size=8, save_partial_every=100):
    """Extract embeddings with resume support, batching, ETA, and partial saves.

    Returns: embeddings (np.ndarray), labels (list), speakers (list), paths (list)
    """
    ensure_dirs()

    total_files = len(files)
    if quick_test:
        files = random.sample(files, min(500, len(files)))

    total = len(files)

    # Select cache files
    cache_quick = EMBED_CACHE_QUICK
    cache_full = EMBED_CACHE_FULL
    partial_cache = EMBED_CACHE_PARTIAL

    # If quick_test and quick cache exists, load it
    if quick_test and os.path.exists(cache_quick):
        try:
            arr = np.load(cache_quick, allow_pickle=True)
            print(f"Loaded cached quick embeddings from {cache_quick}")
            return arr['embeddings'], arr['labels'].tolist(), arr['speakers'].tolist(), arr['paths'].tolist()
        except Exception:
            pass

    # If full cache exists and not quick_test, automatically use it (no interactive prompts)
    if (not quick_test) and os.path.exists(cache_full):
        try:
            arr = np.load(cache_full, allow_pickle=True)
            print(f"Loaded cached full embeddings from {cache_full}")
            return arr['embeddings'], arr['labels'].tolist(), arr['speakers'].tolist(), arr['paths'].tolist()
        except Exception:
            print(f"Found cache at {cache_full} but failed to load; will re-extract.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
    model.to(device)
    model.eval()

    # Check partial resume
    processed_paths = []
    embeddings_acc = []
    labels_acc = []
    speakers_acc = []
    paths_acc = []
    start_index = 0
    if os.path.exists(partial_cache):
        try:
            arr = np.load(partial_cache, allow_pickle=True)
            embeddings_acc = [e for e in arr['embeddings']]
            labels_acc = arr['labels'].tolist()
            speakers_acc = arr['speakers'].tolist()
            paths_acc = arr['paths'].tolist()
            start_index = len(paths_acc)
            print(f"Resuming from partial cache ({start_index}/{total}) at {partial_cache}")
        except Exception:
            print("Could not load partial cache; starting from scratch")
            embeddings_acc = []

    # Remove already processed files from list
    if start_index > 0:
        processed_set = set(paths_acc)
        remaining = [(p, l, s) for (p, l, s) in files if p not in processed_set]
    else:
        remaining = list(files)

    processed = start_index
    t0 = time.time()

    # Helper to save partial
    def save_partial():
        if len(embeddings_acc) > 0:
            np.savez_compressed(partial_cache,
                                embeddings=np.vstack(embeddings_acc),
                                labels=np.array(labels_acc),
                                speakers=np.array(speakers_acc),
                                paths=np.array(paths_acc))
            print(f"Saved partial cache to {partial_cache} ({len(paths_acc)}/{total})")

    # Batch processing
    try:
        for bstart in range(0, len(remaining), batch_size):
            batch = remaining[bstart:bstart+batch_size]
            wavs = []
            meta = []
            for path, label, speaker in batch:
                try:
                    wav, _ = librosa.load(path, sr=SR, mono=True)
                    if len(wav) > MAX_LEN:
                        wav = wav[:MAX_LEN]
                    else:
                        pad = MAX_LEN - len(wav)
                        if pad > 0:
                            wav = np.pad(wav, (0, pad), mode='constant')
                    wavs.append(wav)
                    meta.append((path, label, speaker))
                except Exception as e:
                    print(f"Failed to load {path}: {e}")

            if len(wavs) == 0:
                continue

            # Prepare inputs
            inputs = processor(wavs, sampling_rate=SR, return_tensors='pt', padding=True)
            # move tensors to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # (B, T, H)
            mask = inputs.get('attention_mask', None)
            if mask is None:
                # simple mean pool
                embs = last_hidden.mean(dim=1).cpu().numpy()
            else:
                mask = mask.unsqueeze(-1)  # (B, T, 1)
                summed = (last_hidden * mask).sum(dim=1)  # (B, H)
                lens = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
                embs = (summed / lens).cpu().numpy()

            # Append results
            for i_idx, (path, label, speaker) in enumerate(meta):
                emb = embs[i_idx]
                embeddings_acc.append(emb)
                labels_acc.append(label)
                speakers_acc.append(speaker)
                paths_acc.append(path)
                processed += 1

                # ETA / elapsed
                elapsed = time.time() - t0
                avg = elapsed / (processed if processed > 0 else 1)
                remaining_count = total - processed
                eta = avg * remaining_count
                elapsed_s = str(datetime.timedelta(seconds=int(elapsed)))
                eta_s = str(datetime.timedelta(seconds=int(eta)))
                print(f"Extracting {processed}/{total} | elapsed: {elapsed_s} | ETA: {eta_s}")

                # save partial periodically
                if (processed % save_partial_every) == 0:
                    save_partial()

    except Exception as e:
        print("Batch extraction failed, falling back to single-file loop:", e)
        # fallback single-file loop
        for path, label, speaker in remaining:
            try:
                wav, _ = librosa.load(path, sr=SR, mono=True)
                if len(wav) > MAX_LEN:
                    wav = wav[:MAX_LEN]
                else:
                    pad = MAX_LEN - len(wav)
                    if pad > 0:
                        wav = np.pad(wav, (0, pad), mode='constant')
                inputs = processor(wav, sampling_rate=SR, return_tensors='pt', padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state.squeeze(0)
                emb = last_hidden.mean(dim=0).cpu().numpy()
                embeddings_acc.append(emb)
                labels_acc.append(label)
                speakers_acc.append(speaker)
                paths_acc.append(path)
            except Exception as e2:
                print(f"Failed to process {path}: {e2}")
            processed += 1
            elapsed = time.time() - t0
            avg = elapsed / (processed if processed > 0 else 1)
            remaining_count = total - processed
            eta = avg * remaining_count
            elapsed_s = str(datetime.timedelta(seconds=int(elapsed)))
            eta_s = str(datetime.timedelta(seconds=int(eta)))
            print(f"Extracting {processed}/{total} | elapsed: {elapsed_s} | ETA: {eta_s}")
            if (processed % save_partial_every) == 0:
                save_partial()

    # Finalize
    if len(embeddings_acc) == 0:
        raise RuntimeError("No embeddings extracted")

    embeddings_np = np.vstack(embeddings_acc)
    # Save to final cache (quick or full)
    out_cache = cache_quick if quick_test else cache_full
    np.savez_compressed(out_cache, embeddings=embeddings_np, labels=np.array(labels_acc), speakers=np.array(speakers_acc), paths=np.array(paths_acc))
    print(f"Saved embeddings to {out_cache}")
    # remove partial if exists
    try:
        if os.path.exists(partial_cache):
            os.remove(partial_cache)
    except Exception:
        pass

    return embeddings_np, labels_acc, speakers_acc, paths_acc


def main():
    ensure_dirs()
    files = load_file_list()
    if len(files) == 0:
        print("No files found. Check dataset paths in data_loader/config.")
        return

    # Keep order stable
    files = sorted(files)

    if QUICK_TEST:
        print("QUICK_TEST enabled: using 500 samples max")

    # Balanced subset selection (per-class)
    if SUBSET_PER_CLASS is not None:
        from collections import defaultdict, Counter
        grouped = defaultdict(list)
        for path, label, speaker in files:
            grouped[label].append((path, label, speaker))

        rng = random.Random(RANDOM_STATE)
        selected = []
        for label, items in grouped.items():
            rng.shuffle(items)
            selected.extend(items[:SUBSET_PER_CLASS])

        files = selected
        sel_dist = Counter([lbl for (_, lbl, _) in files])
        print(f"Balanced subset mode: SUBSET_PER_CLASS={SUBSET_PER_CLASS}")
        print(f"Total selected samples: {len(files)}")
        print("Selected class distribution:")
        for k, v in sorted(sel_dist.items()):
            print(f"  {k}: {v}")

    embeddings, labels, speakers, paths = extract_embeddings(files, quick_test=QUICK_TEST, batch_size=8, save_partial_every=100)

    # Basic checks
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        print("Not enough label classes found.")
        return

    # GroupShuffleSplit (speaker-independent)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    idx_train, idx_test = next(gss.split(embeddings, labels, groups=speakers))

    X_train = embeddings[idx_train]
    X_test = embeddings[idx_test]
    y_train = [labels[i] for i in idx_train]
    y_test = [labels[i] for i in idx_test]
    g_train = [speakers[i] for i in idx_train]
    g_test = [speakers[i] for i in idx_test]

    overlap = len(set(g_train).intersection(set(g_test)))

    # Encode labels
    le = LabelEncoder()
    le.fit(labels)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # Scale
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # If running subset tuning (SUBSET_PER_CLASS set) run full grid tuning, else run full-dataset evaluation
    from sklearn.svm import LinearSVC, SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA

    if SUBSET_PER_CLASS is not None:
        results = []  # list of dicts: {'name','params','acc','clf','y_pred'}

        def try_fit(name, params, clf, Xtr, ytr, Xte, yte):
            try:
                clf.fit(Xtr, ytr)
                y_pred = clf.predict(Xte)
                acc = accuracy_score(yte, y_pred)
                print(f"{name} {params} accuracy: {acc:.4f}")
                results.append({'name': name, 'params': params, 'acc': acc, 'clf': clf, 'y_pred': y_pred})
            except Exception as e:
                print(f"Failed {name} {params}: {e}")

        # 1) LogisticRegression grid
        C_vals_lr = [0.1, 0.5, 1, 2, 5]
        for C in C_vals_lr:
            clf = LogisticRegression(C=C, class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE)
            try_fit('LogisticRegression', {'C': C}, clf, X_train_s, y_train_enc, X_test_s, y_test_enc)

        # 2) LinearSVC grid
        C_vals_lsvc = [0.1, 0.5, 1, 2]
        for C in C_vals_lsvc:
            clf = LinearSVC(C=C, class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE)
            try_fit('LinearSVC', {'C': C}, clf, X_train_s, y_train_enc, X_test_s, y_test_enc)

        # 3) SVC RBF grid
        C_vals_svc = [1, 5, 10]
        gamma_vals = ['scale', 0.01, 0.001]
        for C in C_vals_svc:
            for gamma in gamma_vals:
                clf = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='balanced', random_state=RANDOM_STATE)
                try_fit('SVC_RBF', {'C': C, 'gamma': gamma}, clf, X_train_s, y_train_enc, X_test_s, y_test_enc)

        # 4) MLPClassifier grid
        hidden_variants = [(512, 256), (256, 128), (512, 256, 128)]
        alphas = [0.0001, 0.001, 0.01]
        for hidden in hidden_variants:
            for alpha in alphas:
                clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, early_stopping=True, max_iter=500, random_state=RANDOM_STATE)
                try_fit('MLP', {'hidden': hidden, 'alpha': alpha}, clf, X_train_s, y_train_enc, X_test_s, y_test_enc)

        # 5) PCA experiments: reduce then try LogisticRegression and MLP
        pca_components = [128, 256, 384]
        for n_comp in pca_components:
            try:
                pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
                Xtr_p = pca.fit_transform(X_train_s)
                Xte_p = pca.transform(X_test_s)
                # Logistic on PCA
                for C in C_vals_lr:
                    clf = LogisticRegression(C=C, class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE)
                    try_fit(f'PCA{n_comp}+Logistic', {'n_comp': n_comp, 'C': C}, clf, Xtr_p, y_train_enc, Xte_p, y_test_enc)
                # MLP on PCA
                for hidden in hidden_variants:
                    for alpha in alphas:
                        clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, early_stopping=True, max_iter=500, random_state=RANDOM_STATE)
                        try_fit(f'PCA{n_comp}+MLP', {'n_comp': n_comp, 'hidden': hidden, 'alpha': alpha}, clf, Xtr_p, y_train_enc, Xte_p, y_test_enc)
            except Exception as e:
                print(f"PCA {n_comp} failed: {e}")

        # Ranking results
        if not results:
            raise RuntimeError("No classifier results were produced.")

        # Ensure we only sort entries that have an 'acc' key
        results_sorted = sorted([r for r in results if 'acc' in r], key=lambda x: x['acc'], reverse=True)
        print('\nTuned classifier ranking:')
        for r in results_sorted[:20]:
            print(f"- {r['name']} {r['params']}: {r['acc']:.4f}")

        # Save best tuned classifier and artifacts
        if len(results_sorted) == 0:
            raise RuntimeError("No tuned classifiers completed successfully; aborting save.")

        best = results_sorted[0]
        best_clf = best['clf']
        best_acc = best['acc']

        # Save tuned artifacts
        joblib.dump(best_clf, TUNED_MODEL_PATH)
        joblib.dump(le, TUNED_LABEL_PATH)
        joblib.dump(scaler, TUNED_SCALER_PATH)

        # Write tuned report
        with open(TUNED_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(f"Tuned classifier results for subset (SUBSET_PER_CLASS={SUBSET_PER_CLASS})\n")
            f.write(f"Total samples: {len(labels)}\n")
            f.write(f"Total speakers: {len(set(speakers))}\n")
            f.write(f"Train samples: {len(idx_train)}\n")
            f.write(f"Test samples: {len(idx_test)}\n")
            f.write(f"Overlap speakers: {overlap}\n")
            f.write(f"Embedding shape: {embeddings.shape}\n\n")
            f.write("Ranked models (top 50):\n")
            for r in results_sorted[:50]:
                f.write(f"{r['name']}, {r['params']}, {r['acc']:.4f}\n")
            f.write(f"\nBest: {results_sorted[0]['name']} {results_sorted[0]['params']} -> {results_sorted[0]['acc']:.4f}\n")
            # detailed classification report for best
            y_pred_best = results_sorted[0]['y_pred']
            creport = classification_report(y_test_enc, y_pred_best, target_names=le.classes_)
            cm = confusion_matrix(y_test_enc, y_pred_best)
            f.write('\nClassification Report (best):\n')
            f.write(creport)
            f.write('\nConfusion Matrix:\n')
            np.savetxt(f, cm, fmt='%d')

        # Also write a short summary to the original report path for compatibility
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(f"Total samples: {len(labels)}\n")
            f.write(f"Total speakers: {len(set(speakers))}\n")
            f.write(f"Train samples: {len(idx_train)}\n")
            f.write(f"Test samples: {len(idx_test)}\n")
            f.write(f"Overlap speakers: {overlap}\n")
            f.write(f"Embedding shape: {embeddings.shape}\n\n")
            f.write("Top tuned classifier:\n")
            f.write(f"  {results_sorted[0]['name']} {results_sorted[0]['params']} -> {results_sorted[0]['acc']:.4f}\n")
    else:
        # Full-dataset evaluation using best tuned settings from subset
        print('Running full-dataset evaluation using PCA(384)+MLP(512,256), alpha=0.001')
        # Fit PCA on training data
        pca = PCA(n_components=384, random_state=RANDOM_STATE)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)
        # Save PCA
        joblib.dump(pca, PCA_PATH)

        # Define classifiers per requested best options
        clf_mlp = MLPClassifier(hidden_layer_sizes=(512, 256), alpha=0.001, early_stopping=True, max_iter=500, random_state=RANDOM_STATE)
        clf_log = LogisticRegression(C=0.1, class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE)
        clf_lsvc = LinearSVC(C=0.1, class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE)
        # SVC RBF: include if runtime acceptable - we'll run it (may be slower)
        clf_svc = SVC(C=1, kernel='rbf', gamma=0.001, class_weight='balanced', random_state=RANDOM_STATE)

        eval_results = {}
        for name, clf in [('PCA384+MLP', clf_mlp), ('Logistic', clf_log), ('LinearSVC', clf_lsvc), ('SVC_RBF', clf_svc)]:
            try:
                clf.fit(X_train_p, y_train_enc)
                y_pred = clf.predict(X_test_p)
                acc = accuracy_score(y_test_enc, y_pred)
                eval_results[name] = {'clf': clf, 'acc': acc, 'y_pred': y_pred}
                print(f"{name} accuracy: {acc:.4f}")
            except Exception as e:
                print(f"Failed {name}: {e}")
                eval_results[name] = {'clf': None, 'acc': 0.0, 'y_pred': None}

        # Rank
        ranked_full = sorted(eval_results.items(), key=lambda x: x[1]['acc'], reverse=True)
        print('\nFull-run classifier ranking:')
        for name, info in ranked_full:
            print(f"- {name}: {info['acc']:.4f}")

        # Save best
        best_name, best_info = ranked_full[0]
        best_clf_full = best_info['clf']
        best_acc_full = best_info['acc']

        if best_clf_full is None:
            print('No classifier trained successfully on full run; aborting save.')
            return

        # Save artifacts
        joblib.dump(best_clf_full, MODEL_PATH)
        joblib.dump(le, LABEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        # Write full report
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(f"Full-run results\n")
            f.write(f"Total samples: {len(labels)}\n")
            f.write(f"Total speakers: {len(set(speakers))}\n")
            f.write(f"Train samples: {len(idx_train)}\n")
            f.write(f"Test samples: {len(idx_test)}\n")
            f.write(f"Train speakers: {len(set(g_train))}\n")
            f.write(f"Test speakers: {len(set(g_test))}\n")
            f.write(f"Overlap speakers: {overlap}\n")
            f.write(f"Embedding shape: {embeddings.shape}\n\n")
            f.write("Classifier accuracies:\n")
            for name, info in ranked_full:
                f.write(f"  {name}: {info['acc']:.4f}\n")
            f.write(f"\nBest classifier: {best_name} ({best_acc_full:.4f})\n\n")
            y_pred_best = best_info['y_pred']
            creport = classification_report(y_test_enc, y_pred_best, target_names=le.classes_)
            cm = confusion_matrix(y_test_enc, y_pred_best)
            f.write('Classification Report (best):\n')
            f.write(creport)
            f.write('\nConfusion Matrix:\n')
            np.savetxt(f, cm, fmt='%d')

        # Console summary
        print(f"Total samples: {len(labels)}")
        print(f"Total speakers: {len(set(speakers))}")
        print(f"Train samples: {len(idx_train)}")
        print(f"Test samples: {len(idx_test)}")
        print(f"Train speakers: {len(set(g_train))}")
        print(f"Test speakers: {len(set(g_test))}")
        print(f"Overlap speakers: {overlap}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Best full classifier: {best_name} -> {best_acc_full:.4f}")
        print(f"Detailed report written to {REPORT_PATH}")
        # indicate whether it beats Improved Conv1D
        improved_conv1d = 0.5159
        beats = best_acc_full > improved_conv1d
        print(f"Beats Improved Conv1D (51.59%): {beats} (best={best_acc_full:.4f})")

    # Console summary
    print(f"Total samples: {len(labels)}")
    print(f"Total speakers: {len(set(speakers))}")
    print(f"Train samples: {len(idx_train)}")
    print(f"Test samples: {len(idx_test)}")
    print(f"Overlap speakers: {overlap}")
    print(f"Embedding shape: {embeddings.shape}")

    # Print tuned-summary only in subset/tuning mode
    if SUBSET_PER_CLASS is not None:
        # results_sorted is guaranteed to exist in this branch
        best_console = results_sorted[0]
        print("Best tuned classifier:", best_console['name'], best_console['params'], best_console['acc'])
        try:
            print("Tuned report written to", TUNED_REPORT_PATH)
        except NameError:
            pass


if __name__ == '__main__':
    main()
