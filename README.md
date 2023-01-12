# Models
- 10.pkl
  - A 10 feature Viola Jones Classifier
- 50.pkl
  - A 50 feature Viola Jones classifier
- 200.pkl
  - A 200 feature Viola Jones classifier
- cascade.pkl
  - An Attentional Cascade of classifiers looking at 1 feature, 2 features, 5 features, 10 features, and 50 features.
- cascade1.pkl
  - An Attentional Cascade of classifiers looking at 1 feature, 2 features, and 3 features.
- cascade2.pkl
  - An Attentional Cascade of classifiers looking at 1 feature, 10 features, 50 features, and 100 features.

# To Run
- python .\main.py --path "solvay.jpg" --cascade "cascade"

