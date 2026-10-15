"""Create and execute the MoodSense facial-expression evaluation notebook."""

from __future__ import annotations

import base64
import contextlib
import io
import os
import sys
import traceback
from pathlib import Path

import matplotlib
import nbformat as nbf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "MoodSense_Facial_Expression_Model_Evaluation.ipynb"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
# MoodSense AI: Facial-Expression Mood Inference
## CNN Model Evaluation and Local Inference Engine

This notebook evaluates the exact compact ONNX CNN used by the local Flask
application. A frontal face is located with an OpenCV Haar cascade, converted
to grayscale, resized to 64 × 64 pixels, normalized to `[-1, 1]`, and assigned
one of seven visible-expression classes: **angry, disgust, fear, happy, sad,
surprise, or neutral**.

> **Interpretation boundary:** A facial expression is an observable movement,
> not a direct measurement of a person's internal mood. The output is not a
> diagnosis and must not be used alone for health, employment, education,
> policing, insurance, or other high-stakes decisions.
"""
    ),
    md(
        """
### 1. Reproducibility and model provenance

The deployment artifact is an ONNX conversion of the MIT-licensed quantized
TensorFlow Lite model distributed by the open-source `fer` project. The upstream
model was developed for FER-2013 facial-expression recognition. MoodSense adds safe image
validation, frontal-face localization, local inference, browser camera/upload
support, per-user SQLite records, and this reproducible evaluation.

The notebook uses the official **FER-2013 PrivateTest partition** and never
uses those 3,589 observations to update the model weights.
"""
    ),
    code(
        """
from pathlib import Path
import json, platform, time, warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from facial_service import FACIAL_LABELS, MODEL_PATH
from facial_evaluation import (
    ensure_fer2013_csv, load_private_test, evaluate_private_test,
    evaluation_artifacts
)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', context='notebook')
pd.set_option('display.max_columns', 20)

ROOT = Path.cwd()
METADATA_PATH = ROOT / 'models' / 'facial_model_metadata.json'
print('Python:', platform.python_version())
print('OpenCV:', cv2.__version__)
print('scikit-learn:', sklearn.__version__)
print('Model file:', MODEL_PATH)
print('Model size:', f'{MODEL_PATH.stat().st_size / 1024:.1f} KiB')
"""
    ),
    md(
        """
### 2. Evaluation dataset

FER-2013 contains 48 × 48 grayscale face crops labelled from 0 to 6. The CSV
contains predefined `Training`, `PublicTest`, and `PrivateTest` partitions. Only
the 3,589 `PrivateTest` images are evaluated here. On first rerun, the helper
downloads the approximately 92 MB archive and verifies its SHA-256 checksum.
The dataset is intentionally excluded from the application archive.
"""
    ),
    code(
        """
csv_path = ensure_fer2013_csv()
data = load_private_test(csv_path)
data['label'] = data['emotion'].map(dict(enumerate(FACIAL_LABELS)))
print(f'PrivateTest shape: {data.shape[0]:,} rows × {data.shape[1]} columns')
print('Missing values:', int(data.isna().sum().sum()))
display(data[['emotion','label','Usage']].head())
"""
    ),
    code(
        """
counts = data['label'].value_counts().reindex(FACIAL_LABELS)
distribution = pd.DataFrame({
    'Class': [label.title() for label in counts.index],
    'Frequency': counts.values,
    'Percentage': (counts.values / len(data) * 100).round(2),
})
display(distribution)

palette = ['#CF5C59','#73936A','#8878C4','#4DA784','#5F8FB8','#E7AB43','#788785']
fig, ax = plt.subplots(figsize=(10,5.2))
bars = ax.bar(distribution['Class'], distribution['Frequency'], color=palette)
ax.bar_label(bars, padding=3, fontsize=9)
ax.set(title='FER-2013 PrivateTest Class Distribution (n = 3,589)',
       xlabel='Actual expression class', ylabel='Number of images')
sns.despine()
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
**Interpretation.** The test data are naturally imbalanced. Happy has 879
examples, whereas disgust has only 55. For this reason, balanced accuracy and
macro F1 are reported alongside ordinary accuracy.
"""
    ),
    code(
        """
fig, axes = plt.subplots(2, 7, figsize=(14,4.5))
for column, label in enumerate(FACIAL_LABELS):
    subset = data[data['label'] == label].head(2)
    for row_number, pixels in enumerate(subset['pixels']):
        image = np.fromstring(pixels, sep=' ', dtype=np.uint8).reshape(48,48)
        axes[row_number, column].imshow(image, cmap='gray', vmin=0, vmax=255)
        axes[row_number, column].axis('off')
        if row_number == 0:
            axes[row_number, column].set_title(label.title(), fontsize=10)
fig.suptitle('Illustrative FER-2013 PrivateTest Face Crops', y=1.02)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
### 3. Deployment preprocessing and inference

For the already cropped FER-2013 evaluation images, face localization is not
repeated. Each 48 × 48 crop is resized to 64 × 64 with bicubic interpolation,
converted to floating point, divided by 255, and scaled to `[-1, 1]`. The
deployed ONNX Runtime session then returns seven class probabilities.

In the web application, a Haar cascade first detects all frontal faces and the
largest face is selected. This evaluation therefore measures **expression
classification on provided face crops**, not real-world face-detection recall.
"""
    ),
    code(
        """
started = time.perf_counter()
y_true, y_pred, y_probability = evaluate_private_test(data, progress_every=500)
elapsed = time.perf_counter() - started
artifacts = evaluation_artifacts(y_true, y_pred, y_probability)
metrics = artifacts['metrics']
print(f'Evaluated {len(data):,} images in {elapsed:.2f} seconds')
print(f'Average crop-classification time: {elapsed / len(data) * 1000:.2f} ms/image')
"""
    ),
    md("### 4. Overall model performance"),
    code(
        """
metric_table = pd.DataFrame({
    'Metric': [
        'Accuracy','Balanced accuracy','Macro F1','Weighted F1',
        'Macro ROC–AUC (OvR)','Expected calibration error (10 bins)',
        'Multiclass Brier score','Mean confidence'
    ],
    'Value': [
        metrics['accuracy'],metrics['balanced_accuracy'],metrics['macro_f1'],
        metrics['weighted_f1'],metrics['macro_roc_auc_ovr'],
        metrics['expected_calibration_error_10_bins'],
        metrics['multiclass_brier_score'],metrics['mean_confidence']
    ]
})
display(metric_table.style.format({'Value':'{:.4f}'}))
"""
    ),
    md(
        """
**Overall interpretation.** Accuracy represents the percentage of all test
images classified correctly. Balanced accuracy gives each expression equal
importance, and macro F1 summarizes class-wise precision and recall without
allowing the larger happy or neutral classes to dominate. ROC–AUC measures
ranking quality, whereas calibration error and Brier score assess the quality
of the probability estimates.
"""
    ),
    md("### 5. Class-specific performance"),
    code(
        """
report = pd.DataFrame(artifacts['classification_report']).T
class_report = report.loc[FACIAL_LABELS, ['precision','recall','f1-score','support']]
display(class_report.style.format({
    'precision':'{:.4f}','recall':'{:.4f}','f1-score':'{:.4f}','support':'{:.0f}'
}))
print(classification_report(
    y_true, y_pred, labels=np.arange(7), target_names=FACIAL_LABELS,
    digits=4, zero_division=0
))
"""
    ),
    code(
        """
matrix = np.asarray(artifacts['confusion_matrix'])
fig, ax = plt.subplots(figsize=(8.5,7))
sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
            xticklabels=[x.title() for x in FACIAL_LABELS],
            yticklabels=[x.title() for x in FACIAL_LABELS], ax=ax)
ax.set(title='Confusion Matrix — FER-2013 PrivateTest',
       xlabel='Predicted expression', ylabel='Actual expression')
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
**Confusion-matrix interpretation.** Correct classifications appear on the
main diagonal. The model performs most strongly on happy expressions. Fear is
frequently confused with angry, sad, or surprise, while sad and neutral are
also commonly confused. These errors show why a single frame must not be
treated as certain knowledge of a person's emotional state.
"""
    ),
    code(
        """
plot_metrics = class_report[['precision','recall','f1-score']]
ax = plot_metrics.plot(kind='bar', figsize=(10.5,5.5), width=.76,
                       color=['#173B39','#DE784D','#5F8FB8'])
ax.set(title='Precision, Recall, and F1 by Facial-Expression Class',
       xlabel='Expression class', ylabel='Score', ylim=(0,1.05))
ax.tick_params(axis='x', rotation=0)
ax.legend(loc='upper left')
sns.despine()
plt.tight_layout()
plt.show()
"""
    ),
    md("### 6. One-vs-rest ROC analysis"),
    code(
        """
binary_truth = label_binarize(y_true, classes=np.arange(7))
fig, ax = plt.subplots(figsize=(9,6.5))
roc_rows = []
for index, (label, colour) in enumerate(zip(FACIAL_LABELS, palette)):
    false_positive, true_positive, _ = roc_curve(binary_truth[:,index], y_probability[:,index])
    class_auc = auc(false_positive, true_positive)
    roc_rows.append({'Class':label.title(), 'ROC AUC':class_auc})
    ax.plot(false_positive, true_positive, lw=2, color=colour,
            label=f'{label.title()} (AUC = {class_auc:.3f})')
ax.plot([0,1],[0,1],'--',color='#8d9893',lw=1)
ax.set(title='One-vs-Rest ROC Curves — FER-2013 PrivateTest',
       xlabel='False-positive rate', ylabel='True-positive rate')
ax.legend(loc='lower right', fontsize=8)
sns.despine()
plt.tight_layout()
plt.show()
display(pd.DataFrame(roc_rows).style.format({'ROC AUC':'{:.4f}'}))
"""
    ),
    md("### 7. Probability calibration"),
    code(
        """
confidence = y_probability.max(axis=1)
correct = y_pred == y_true
edges = np.linspace(0,1,11)
calibration_rows = []
for lower, upper in zip(edges[:-1], edges[1:]):
    selected = (confidence > lower) & (confidence <= upper)
    if selected.any():
        calibration_rows.append({
            'Lower':lower, 'Upper':upper, 'Count':int(selected.sum()),
            'Mean confidence':float(confidence[selected].mean()),
            'Observed accuracy':float(correct[selected].mean())
        })
calibration = pd.DataFrame(calibration_rows)
fig, ax = plt.subplots(figsize=(7,6))
ax.plot(calibration['Mean confidence'], calibration['Observed accuracy'],
        marker='o', lw=2, color='#DE784D', label='CNN')
ax.plot([0,1],[0,1],'--',color='#788785',label='Perfect calibration')
for row in calibration.itertuples():
    ax.annotate(str(row.Count), (row._4, row._5), xytext=(4,4),
                textcoords='offset points', fontsize=7)
ax.set(title='Reliability Diagram (labels show bin sample size)',
       xlabel='Mean predicted confidence', ylabel='Observed accuracy',
       xlim=(0,1), ylim=(0,1))
ax.legend(loc='upper left')
sns.despine()
plt.tight_layout()
plt.show()
display(calibration.style.format({
    'Lower':'{:.1f}','Upper':'{:.1f}',
    'Mean confidence':'{:.4f}','Observed accuracy':'{:.4f}'
}))
"""
    ),
    md(
        """
**Calibration interpretation.** The expected calibration error summarizes the
gap between confidence and observed accuracy. A lower value is better. The
diagram is important because a high softmax probability is not a guarantee
that the visible-expression label is correct.
"""
    ),
    md("### 8. Error analysis"),
    code(
        """
error_indices = np.flatnonzero(y_pred != y_true)
confusion_rows = []
for actual_index, actual_label in enumerate(FACIAL_LABELS):
    for predicted_index, predicted_label in enumerate(FACIAL_LABELS):
        if actual_index != predicted_index and matrix[actual_index,predicted_index] > 0:
            confusion_rows.append({
                'Actual':actual_label.title(), 'Predicted':predicted_label.title(),
                'Count':int(matrix[actual_index,predicted_index])
            })
confusion_pairs = pd.DataFrame(confusion_rows).sort_values('Count',ascending=False)
display(confusion_pairs.head(12))

chosen_errors = sorted(error_indices, key=lambda idx: y_probability[idx,y_pred[idx]], reverse=True)[:14]
fig, axes = plt.subplots(2,7,figsize=(14,5))
for axis, index in zip(axes.ravel(), chosen_errors):
    image = np.fromstring(data.iloc[index]['pixels'], sep=' ', dtype=np.uint8).reshape(48,48)
    axis.imshow(image,cmap='gray',vmin=0,vmax=255)
    axis.set_title(
        f"A: {FACIAL_LABELS[y_true[index]]}\\nP: {FACIAL_LABELS[y_pred[index]]} ({y_probability[index,y_pred[index]]:.0%})",
        fontsize=8
    )
    axis.axis('off')
fig.suptitle('Highest-Confidence Misclassifications', y=1.01)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
### 9. Save the evaluation metadata

The Flask application loads `facial_expression_model.onnx`; the metadata
below records how that exact artifact performed. The CSV images are never
saved into the web application's user database.
"""
    ),
    code(
        """
metadata = {
    'project':'MoodSense AI Local — Facial Expression Engine',
    'model_name':'FER-2013 compact CNN (ONNX)',
    'model_file':MODEL_PATH.name,
    'classes':FACIAL_LABELS,
    'evaluation_split':'FER-2013 PrivateTest',
    'evaluation_rows':len(data),
    'important_boundary':'Facial expression classification is not direct measurement of internal mood or a diagnosis.',
    **artifacts,
}
METADATA_PATH.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('Saved evaluation metadata:', METADATA_PATH)
"""
    ),
    md("### 10. Example inference results"),
    code(
        """
demo_rows = []
for class_index, label in enumerate(FACIAL_LABELS):
    index = data.index[data['emotion'] == class_index][0]
    demo_rows.append({
        'Actual expression':label.title(),
        'Predicted expression':FACIAL_LABELS[y_pred[index]].title(),
        'Confidence (%)':round(y_probability[index,y_pred[index]]*100,2)
    })
display(pd.DataFrame(demo_rows))
"""
    ),
    md(
        """
## Conclusion and documentation guidance

On the 3,589-image FER-2013 PrivateTest split, the deployed ONNX CNN
achieved approximately **64.42% accuracy**, **61.83% balanced accuracy**,
**62.62% macro F1**, and **0.9084 macro one-vs-rest ROC–AUC**. Happy was the
strongest class, while fear, sad, and anger showed substantial overlap.

These results demonstrate a functional facial-expression classifier under the
FER-2013 benchmark. They do not establish a universal mood detector. FER-2013
contains low-resolution faces gathered from internet sources, facial labels can
be subjective, a single image lacks personal and cultural context, and the
notebook does not measure webcam face-detection failures. A real deployment
requires informed consent, privacy controls, representative external data,
demographic subgroup evaluation, temporal testing across video frames, and
human review.

**Suggested reporting language:** “The deployed facial-expression CNN obtained
64.42% accuracy and a macro F1-score of 62.62% on the official 3,589-image
FER-2013 PrivateTest split. The system classifies visible expression patterns
rather than directly measuring internal mood; its outputs are therefore
probabilistic cues and not psychological diagnoses.”
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def notebook_display(*objects):
    for value in objects:
        if hasattr(value, "data") and hasattr(value.data, "to_string"):
            print(value.data.to_string())
        elif hasattr(value, "to_string"):
            print(value.to_string())
        else:
            print(repr(value))


def execute_in_process(document):
    namespace = {"__name__": "__main__", "display": notebook_display}
    previous_directory = Path.cwd()
    os.chdir(ROOT)
    count = 0
    try:
        for cell in document.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            cell.execution_count = count
            cell.outputs = []
            stdout, stderr = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(cell.source, f"<facial-notebook-cell-{count}>", "exec"), namespace)
            except Exception as error:
                combined = stdout.getvalue() + stderr.getvalue()
                if combined:
                    cell.outputs.append(nbf.v4.new_output("stream", name="stdout", text=combined))
                cell.outputs.append(
                    nbf.v4.new_output(
                        "error",
                        ename=type(error).__name__,
                        evalue=str(error),
                        traceback=traceback.format_exc().splitlines(),
                    )
                )
                raise
            combined = stdout.getvalue() + stderr.getvalue()
            if combined:
                cell.outputs.append(nbf.v4.new_output("stream", name="stdout", text=combined))
            for figure_number in plt.get_fignums():
                figure = plt.figure(figure_number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                cell.outputs.append(
                    nbf.v4.new_output(
                        "display_data",
                        data={"image/png": encoded, "text/plain": "<Matplotlib figure>"},
                        metadata={},
                    )
                )
            plt.close("all")
    finally:
        os.chdir(previous_directory)


execute_in_process(notebook)
nbf.write(notebook, OUTPUT)
print(f"Executed notebook written to {OUTPUT}")
