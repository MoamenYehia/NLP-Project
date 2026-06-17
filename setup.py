"""
Run this once before starting the app to download required NLTK data.
  python setup.py
"""
import nltk

packages = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "vader_lexicon",
]

print("Downloading NLTK data...")
for pkg in packages:
    nltk.download(pkg, quiet=False)

print("\nAll done! You can now run: python app.py")
