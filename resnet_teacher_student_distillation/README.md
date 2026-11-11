# ResNet Teacher–Student Distillation with Rotation Self-Distillation

We trained a compact residual network for the CIFAR-10 challenge for our Deep Learning course. Our final student has **4,896,418 trainable parameters** and achieved **87.90% private accuracy** and **90.40% public accuracy** on Kaggle.

We first trained a larger teacher from scratch, then used its predictions to guide a smaller student. We also gave the student a rotation task and used those predictions for self-distillation. Both models used only the permitted competition training data. The teacher was used during training and was not part of the submitted network.

This folder contains this experiment, its executed notebooks, and its saved results.

## Final result

| Measurement | Result |
| --- | --- |
| Student | WRN-34, width factor 3.25, with a rotation head |
| Trainable parameters, including the rotation head | 4,896,418 |
| Parameter limit | 5,000,000 |
| Submitted file | `submission_final_ema_multi.csv` |
| Kaggle private accuracy | **87.90%** |
| Kaggle public accuracy | **90.40%** |
| Validation accuracy for this candidate | 97.70% |

The validation and Kaggle scores come from different image sets. Our validation accuracy was much higher than our hidden-test accuracy, so it was not a reliable estimate of the final competition result.

## Our approach

### Teacher guidance

We trained a WRN-28-10 teacher for 300 epochs on 45,000 images and kept 5,000 for validation, with 500 images per class. The teacher has 36,479,194 parameters. It is only a training aid and is not included in the submitted network.

We then trained the WRN-34 student from scratch for 330 epochs using the same split. The student learned from both the correct labels and the frozen teacher’s class probabilities.

We used a distillation temperature of 4. The teacher-loss weight started at 0.88 and fell to 0.60 over the last 120 epochs, giving the correct labels more influence near the end.

### Rotation learning and self-distillation

We made four views of each student training image, rotated by 0, 90, 180, and 270 degrees. An extra output layer predicted 40 combinations of object class and rotation.

We averaged the class scores associated with the correct rotation across these views. Their probabilities guided the student’s normal classification output. Gradients did not flow back through these target probabilities during that update.

This let the student learn from its own rotation predictions as well as from the larger teacher.

The rotation-label and self-distillation losses each had a weight of 0.5. We increased their contribution gradually over the first ten epochs.

Test predictions used the normal ten-class output. We still included the extra rotation head in the full parameter count.

### Training and prediction

We used SGD with momentum, weight decay, a short learning-rate warm-up, and cosine decay.

Image augmentation included random crops, horizontal flips, TrivialAugmentWide, cutout, occasional grayscale, and mild blur, JPEG compression, and color changes. Mixup was disabled in this experiment.

We kept an exponential moving average (EMA) of the model weights. We also averaged late EMA checkpoints as a separate prediction option.

We compared these weights with flip, multi-view, and cardinal test-time augmentation (TTA). TTA averages predictions across different views of an image without adding model parameters.

Several settings changed from our earlier experiments, so these results do not measure the effect of self-distillation alone.

## Training settings

| Setting | Teacher | Student |
| --- | --- | --- |
| Architecture | WRN-28-10 | WRN-34, width 3.25 |
| Epochs | 300 | 330 |
| Batch size | 128 | 64 original images, expanded to four views |
| Peak learning rate | 0.1 | 0.04625 |
| Optimizer | SGD, momentum 0.9 | SGD, momentum 0.9 |
| Weight decay | 0.0005 | 0.0005 |
| Warm-up | 5 epochs | 5 epochs |
| Schedule | Cosine decay | Cosine decay |
| EMA decay | 0.999 | 0.999 |
| Model seed | 17 | 41 |
| Training and validation images | 45,000 / 5,000 | Same split |

## Validation results

The best EMA validation scores seen during training were:

- **Teacher:** 97.24% at epoch 258.
- **Student:** 97.38% at epoch 300.

These best-epoch scores are separate from the fixed final weights used to generate the submission files.

After training, we compared six prediction options:

| Student weights | TTA | Validation accuracy |
| --- | --- | --- |
| Tail average | Flip | 97.72% |
| Final EMA | Multi | 97.70% |
| Final EMA | Cardinal | 97.68% |
| Tail average | Cardinal | 97.68% |
| Tail average | Multi | 97.60% |
| Final EMA | Flip | 97.54% |

The top two differed by only one of the 5,000 validation images. Our submitted file used **final EMA with multi-view TTA**. The Kaggle scores at the top of this README belong to that file.

## Graphs

### Training curves

![Teacher training curves](figures/teacher_training_curves.png)

![Student training curves](figures/student_training_curves.png)

Both losses fell steadily, and validation accuracy improved quickly before leveling off. EMA made the validation curves smoother.

Training accuracy was measured on harder augmented images, which helps explain why it was below clean-image validation accuracy. The student combines several losses, so its loss value is not directly comparable with the teacher’s.

### Comparing prediction methods

![Prediction methods](figures/student_prediction_methods.png)

The methods performed very similarly on validation. More image views did not always improve accuracy. The horizontal axis is shortened to make the small differences visible.

### Teacher and student validation accuracy

![Teacher and student validation accuracy](figures/teacher_student_validation.png)

The student slightly exceeded the teacher’s best EMA validation score. Each curve belongs to a separate training stage, so equal epoch numbers do not mean equal elapsed time.

A matching student run without distillation would be needed to measure how much teacher guidance helped.

### Predicted classes

![Predicted class distribution](figures/predicted_class_distribution.png)

This shows how many test images were assigned to each class by `submission_tail_avg_flip.csv`, the validation-leading option chosen by the graph notebook.

It is a count of predictions, not per-class accuracy.

## Folder contents

| Path | Contents |
| --- | --- |
| `notebooks/01_training_results.ipynb` | Training code with saved execution output |
| `notebooks/02_results_graphs.ipynb` | Graph code with saved figures |
| `results/teacher/history.csv` | Teacher training history |
| `results/student/` | Student weights, history, report, prediction CSVs, and logits |
| `figures/` | PNG graphs |

## Run the training notebook

The training notebook was written for **Google Colab**. It uses a GPU, a Google Drive mount, and `/content/` paths. Running it in another environment requires changes to the paths and file-upload steps.

1. Open `notebooks/01_training_results.ipynb` in Colab and select a GPU runtime.
2. Upload the permitted dataset as `data-DL.zip` when prompted, or place it at `MyDrive/data-DL.zip` first.
3. Run the notebook and authorize Google Drive access.
4. With the default two-stage settings, the teacher trains first and the student follows. Both start from scratch if their output folders have no existing checkpoints.

Outputs are saved under:

```text
/content/drive/MyDrive/Operation90_TwoStage/
```

The teacher and student have separate run folders. The full run takes hours, depending on the GPU and storage. Student training processes four rotated views for each image and also runs the frozen teacher.

The notebook uses PyTorch, torchvision, NumPy, pandas, Pillow, tqdm, and Colab utilities. The graph notebook also uses Matplotlib.

## Make the graphs without training

No GPU or model weights are needed for the saved-history graphs.

1. Download `results/teacher/history.csv` and the `results/student/` folder.
2. Compress the student folder into a ZIP that keeps the folder inside the archive. For example, it should contain `student/history.csv` and `student/report.json`.
3. Open `notebooks/02_results_graphs.ipynb` in Colab with a CPU runtime.
4. Run the first cell and upload the teacher history and student-folder ZIP together.
5. Run the second cell.

The notebook displays the figures and downloads `all_training_graphs.zip`.

The smaller `student_submissions.zip` in the repository contains history, report, and prediction files at the archive root. The current graph notebook expects a folder inside the ZIP, so use the folder ZIP described above.

## Weights and checkpoints

The teacher weights and resume checkpoints are available in [release v1.0-weights](https://github.com/NeelGaji/Prometheus/releases/tag/v1.0-weights).

| File | Location | Use |
| --- | --- | --- |
| `student_final_ema.pth` | `results/student/` | Final EMA student weights |
| `student_tail_avg.pth` | `results/student/` | Late-averaged student weights |
| Teacher and student `history.csv` | Their results folders | Per-epoch training measurements |
| `report.json` | `results/student/` | Parameter count and validation scores |
| `submission_*.csv` | `results/student/` | Test predictions |
| `test_logits_*.npy` | `results/student/` | Saved test outputs |
| `student_submissions.zip` | `results/student/` | Packaged history, report, and prediction CSVs |
| `teacher_final_ema.pth` | Release | Teacher weights for separate student training |
| `teacher_last.pth` | Release | Optional checkpoint to resume teacher training |
| `student_last.pth` | Release | Optional checkpoint to resume student training |

### Train only the student

Download the teacher’s EMA weights into Google Drive. Set these values in the training notebook, using the file’s actual path:

```python
STAGE = 'student'
TEACHER_CHECKPOINT = '/content/drive/MyDrive/teacher_final_ema.pth'
```

This skips teacher training and uses its frozen weights to guide the student. Keep the original dataset and validation split.

A fresh student run needs an output folder without an existing student checkpoint.

### Resume an interrupted run

The release gives the two `last.pth` files different names so they can be distinguished.

1. Download the checkpoint for the stage you want to resume.
2. Rename it to `last.pth`.
3. Put it in a separate Drive folder, such as `MyDrive/resume_student`.
4. Select one stage and set the source folder:

```python
STAGE = 'student'
RESUME_SOURCE_DIR = '/content/drive/MyDrive/resume_student'
TEACHER_CHECKPOINT = '/content/drive/MyDrive/teacher_final_ema.pth'
```

For teacher training, use `STAGE = 'teacher'` and leave `TEACHER_CHECKPOINT` empty.

Keep the original settings when resuming, including the epoch count. This import step copies the source folder into the matching run folder and requires that destination not to exist. Rename an older destination folder if necessary to keep it safe.

If the original run folder is already in Drive, leave `RESUME_SOURCE_DIR` empty and rerun with the same settings to resume directly.

Student training also needs the teacher weights at the configured path.

## References

- Zagoruyko and Komodakis. [Wide Residual Networks](https://arxiv.org/abs/1605.07146), 2016.
- Hinton, Vinyals, and Dean. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), 2015.
- Lee, Hwang, and Shin. [Self-supervised Label Augmentation via Input Transformations](https://arxiv.org/abs/1910.05872), 2020.

Further references and our experiment comparisons are included in the project report.
