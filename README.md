# Complex Word Classification

The task of this challenge is to automatically classify whether or not a word in a given sentence is complex.

### Description:

Predicting which words are considered hard to understand for a given target population is a vital step in many Natural Language Processing applications such as text simplification. A system could simplify texts for second language learners, native speakers with low literacy levels, and people with reading disabilities. This task is commonly referred to as Complex Word Identification. Usually, this task is approached as a binary classification task in which systems predict a complexity value (complex vs. non-complex) for a set of target words in a text. In this challenge, the task is to predict the lexical complexity of a word in a sentence. A word which is considered to be complex has label 1, a word is considered to be simple (non-complex) has label 0.

### Data:

The data comes from three sources: biblical text, biomedical articles and proceedings of the European Parliament. These sources were selected as they contain a natural mixture of common language and difficult to understand expressions, whilst each containing vastly different domain-specific vocabulary. The training data consists of 7662 training examples, each training example is a row of the form (id, corpus, sentence, token, complex). The testing data consists of 1338 test examples, each test example is a row iof the form (id, corpus, sentence, token). Notice that you have to infer the label 0 (the token is not complex) or 1 (the token is complex).

**File Description:**

- **train.xlsx** - the training set
- **test.xlsx** - the test set
- **sample_submission.csv** - a sample submission file in the correct format, with some labels equal to 0, and some labels equal to 1.

### I have realised an AI that has an 0.78150 accuracity using Gaussian Naive Bayes and another one with 0.58 with KNN.