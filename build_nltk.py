"""
Vercel build hook — downloads required NLTK data at build time.
This runs during `vercel build` so the data is bundled into the deployment.
"""
import nltk
import os

# Download to a path that will be included in the deployment
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

nltk.download("vader_lexicon", download_dir=nltk_data_dir)
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("punkt_tab", download_dir=nltk_data_dir)
nltk.download("averaged_perceptron_tagger", download_dir=nltk_data_dir)

print(f"✅ NLTK data downloaded to {nltk_data_dir}")
