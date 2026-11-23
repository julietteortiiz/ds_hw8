# Provided HW08 data

`digits.npz`: X (1797, 64), float intensities 0..16; y (1797,), digit labels.
Exported without reordering or filtering from sklearn.datasets.load_digits
(scikit-learn 1.6.1). Each row is an 8 by 8 image flattened in row-major order.
Labels are for visualization only. This is the UCI Optical Recognition of
Handwritten Digits test subset; attribution: E. Alpaydin and C. Kaynak.
Source: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
Loader documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
The upstream dataset description is preserved in digits_DESCRIPTION.txt.

`svd_image.npy`: float 128 by 128 disk, generated for this assignment:

```python
axis = np.linspace(-1.0, 1.0, 128)
xx, yy = np.meshgrid(axis, axis)
image = ((xx**2 + yy**2) <= 0.65**2).astype(float)
```

Load these through data_utils.py. No network connection is needed.
