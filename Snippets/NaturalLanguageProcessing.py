# Natural Language Processing
"""from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
filtered = [word for word in stopwords.words('english') if word != "not"]

corpus = []
for review in range(0, 1000):
    ps = PorterStemmer()

    review = re.sub("[^a-zA-Z]", " ", dataset["Review"]
                    [review]).lower().split()
    review = [ps.stem(word) for word in review if not word in filtered]
    review = " ".join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1]"""
