from datasets import load_dataset
import spacy
from nltk.corpus import wordnet as wn

ds = load_dataset("wikitext", "wikitext-103-v1")
print("wikitext train size:", len(ds["train"]))

nlp = spacy.load("en_core_web_sm")
print("spacy sentences:", [s.text for s in nlp("Hello world. This is a test.").sents])

print("wordnet synsets:", len(list(wn.all_synsets())))
