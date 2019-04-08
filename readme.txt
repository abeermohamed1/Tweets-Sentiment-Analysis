Please include in the folder  :
- GoogleNews-vectors-negative300.bin

This project is a ‘Tweets Sentiment Analysis’ project. The objective, in this project, is classifying with good accuracy the polarity of tweets either as ‘Positive’, ‘Negative’, and ‘Neutral’. In this report, we are presenting the classification model we built in order to reach this objective. There are a lot of papers in the literature addressing the problem of ‘Tweet Sentiment Analysis’, a lot of papers presented multiple processing techniques and features that can be used in the classification problem. We used some external papers beside the paper provided in the project to help us in understanding the problem, and approaching it. We used many techniques, features and algorithms to help us in building an effective classification model. We tested a lot of classification models using various combinations of techniques and features, and at the end we succeeded in building a model that is able to classify the polarity of the tweets with good accuracy.

These datasets are originally provided by the “International Workshop on Semantic Evaluation (SemEval)” on 2013, 2015, and 2016 [4]. We merged the three datasets to be used in our model as one training dataset. Each dataset consist of three main parts: tweet id, tweet polarity, and tweet. The polarity is either positive, negative or neutral.
The statistics

1.Tweets Cleaning and Preprocessing:
2.Document Vector Generation
3.Document Vector Preprocessing
4.Features Addition
5.Classification

1.Tweets Cleaning and Preprocessing:
1. Lowercasing
2. Stemming
3. Lemmatizing
4. Stop Word Removal
5. Emoticons Removal
6. Unicode Removal
7. Numbers Removal
8. URL Removal
9. Hashtag Removal
10. ‘@’ Sign Removal
11. Contraction Replacement
12. Punctuation Removal
13. Punctuation Replacement
a. Any repeated exclamation marks (by any number of repetitions) is replaced by the word “MultiExclamationMarks”.
b. Any repeated question marks (by any number of repetitions) is replaced by the word “MultiQuestionMarks”.
c. Any repeated stop marks (by any number of repetitions) is replaced by the word “MultiStopMarks”.
14. Slang/abbreviation Replacement: Replacing slang and abbreviation (uppercased or lowercased) by its corresponding complete sentence. For example: ‘BTW’ will be replaced by ‘by the way’. Slang lexicon is used in this technique.
15. Elongated Word Replacemen
16. Word Correction
17. Negation Replacement

2.Document Vector Generation
1. Count Model
2. TF-IDF Model
3. Word Embeddings Model

3- Document Vector Preprocessing Techniques
These are the techniques applied on the document vector. Two techniques were tested which are: dimensionality reduction, and normalization. Dimensionality reduction algorithms tested are: Chi2 and PCA (Principal Component Analysis).

4- Features Addition:
1. Words: Number of all the words in the tweet.
2. Elongated Words: Number of elongated words in the tweet (e.g. ‘booooring’, etc.).
3. Exclamation Marks: Number of exclamation marks in the tweet (either they are separated or not).
4. Question Marks: Number of question marks in the tweet (either they are separated or not).
5. Dots: Number of dots in the tweet (either they are separated or not).
6. Capitalized Words: Number of fully capitalized words in the tweet (e.g. TODAY).
7. Positive Words: Number of positive words in the tweet (the lexicon of positive words provided in the project is used in this feature).
8. Negative Words: Number of negative words in the tweet (the lexicon of negative words provided in the project is used in this feature).
9. Slang/abbreviation: Number of slang/abbreviations in the tweet (slang lexicon is used in this feature).
10. Positive Emoticons: Number of positive emoticons in the tweet (list of defined positive emoticons was used in this feature).
11. Negative Emoticons: Number of negative emoticons in the tweet (list of defined negative emoticons was used in this feature).
12. URLs: Number of URLs in the tweet. This feature counts URLs with different formats.
13. Hash tags: Number of hash tags (#) in the tweet.
14. ‘@’ Signs: Number of (@) signs in the tweet. This sign is used when person or page is mentioned in the tweet.
15. Nouns: Number of nouns in the tweet. POS tagging is used in this feature.
16. Verbs: Number of verbs in the tweet. POS tagging is used in this feature.
17. Adjectives: Number of adjectives in the tweet. POS tagging is used in this feature.
18. Adverbs: Number of adverbs in the tweet. POS tagging is used in this feature.
19. Vader Sentiment Analysis Vector

5- Classification
We chose 10 different classifiers to be used in testing the classification accuracy.
1. Naïve Bayes (Multinomial)
2. Random Tree
3. Random Search (advanced version of ‘Random Tree classifier’)
4. Logistic Regression
5. Random Forest
6. Decision Tree
7. XGB
8. SVM (‘rbf’ kernel)
9. Boosted Tree
10. Ensemble Learning Voting Classifier (This classifier uses multiple classifiers and vote on them)
