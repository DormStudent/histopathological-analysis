# histopathological-analysis
Making a learning model in Python that analyzes various tissue samples and gives a probability of occurrence of malignant growth

## Extracting a UNI embedding from a patch

`analyze_patch.py` uses the gated [UNI model](https://huggingface.co/MahmoodLab/UNI)
to turn one histology patch image into a 1,024-dimensional feature vector. It
does not predict malignancy; that requires a separate classifier trained for
the target task.

Install PyTorch, timm, and Pillow in your Python environment:

```bash
python -m pip install torch timm Pillow
```

Request access to the UNI model on Hugging Face, agree to its terms, and log in
with an account that has been granted access:

```bash
hf auth login
```

Call the function with a patch image path. It returns a 1,024-dimensional
NumPy array; optionally pass an output path to save it:

```python
from analyze_patch import analyze_patch

embedding = analyze_patch("path/to/patch.png")
# Or save it as a NumPy file while extracting:
embedding = analyze_patch("path/to/patch.png", "output/patch_embedding.npy")
```

You can also use the command-line wrapper, which saves the embedding next to
the image by default:

```bash
python analyze_patch.py path/to/patch.png
```

To choose a different output path:

```bash
python analyze_patch.py path/to/patch.png --output output/patch_embedding.npy
```
