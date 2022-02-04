# GA_Feature_Selection
GA_Feature_Selection

# How to use

## Run SVM baseline:
```
python3 baseline_svm.py 
```

## Run SVM+GA benchmark:
```
python3 start.py 
```

## Dataset:
- The benchmark is based on the UCI breast cancer dataset, which contains 569 instances and 30 features.
- In wdbc.data:
```
    - The first feature represents ID which you can ignore in this benchmark.
    - The second feature represents Diagnosis (M = malignant, B = benign). In other words, this feature is the label.
```
