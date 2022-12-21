---
title: 30 Days of Machine Learning Engineering
author: Francesco Di Salvo
date: 2022-12-05
topics: [Machine Learning]
meta: 30 Days of Machine Learning Engineering
target: Expert
language: English
cover: ./cover.jpeg
published: true
layout: guide
---


# Thirty Days of Machine Learning Engineering

## Day 00 - What is this challenge about?

After completing the #66DaysOfDataChallenge, I decided to tackle a new challenge, namely the **#30DaysOfMachineLearningEngineering** challenge! 

During the last two years I learned a lot about theoretical concepts in Machine Learning, Deep Learning and several other related disciplines. However, I recognize that I still need to learn and to work on some of the “**best practices**” in the field. 

After some research, I found the great book “**Machine Learning Engineering**” by [Andriy Burkov](https://www.linkedin.com/in/andriyburkov/), and after I’ve read the foreword by [Cassie Kozyrkov](https://www.linkedin.com/in/kozyrkov/), I was convinced that it was the right one. 

Therefore, for the following **30 days**, I will share daily bites from the book and I would love to open some debates on the relative topics! 

I will convey all the daily posts on a **GitHub repository** and this **blog**!

P.S. I do truly believe on the values of this challenge, and great examples can be given by [Boris Giba](https://www.linkedin.com/in/borisgiba/) and [Tinham Tamang](https://www.linkedin.com/in/thinam-tamang/), just to name a few. 

Disclaimer: The book is distributed on the “read first, buy later” principle. Therefore you can  freely read the book too and then decide to buy it!

**Sources**:

- [Book](http://www.mlebook.com/wiki/doku.php?id=start)
- [Blog](https://staituned.com/learn/expert/30-days-of-machine-learning-engineering)
- [LinkedIn](https://www.linkedin.com/company/stai-tuned)
- Github: [#66DaysOfDataChallenge](https://github.com/francescodisalvo05/66DaysOfData)
- Github: [#30DaysOfMachineLearningEngineering](https://github.com/francescodisalvo05/30DaysOfMachineLearningEngineering)
- [Cover image](https://www.ocpathink.org/post/quality-not-seniority-stressed-for-teacher-pay)

## Day 01 - Machine Learning preliminaries

Today we went through the first chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

This chapter starts from the **initial definitions** of Machine Learning, Supervised vs Unsupervised learning, as in any ML book. 

However, there are two interesting sections devoted to “when to use” and “when to not use” Machine Learning. 

Ideally, Machine Learning **should be employed** when the problem cannot be hard-coded, when we expect a lot of variations over time, when it involves a perceptive problem, when there is little information about the topic, when it has a simple objective and finally, when it is cost effective! 

On the other hand, Machine Learning **should be avoided** when every action or behavior has to be explained, when the error cost it too high, when the data is too expansive or hard to obtain, when the problem is too complex and it has too many required outcomes and when it can be eventually solved through heuristics or lookup tables.


## Day 02 - Impact of a Machine Learning project

Today we went through the second chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

It covers the **preliminary stage of any ML project**. The key aspect of any project is the prioritization, based on estimated impact and costs. 

The **impact** can be measured on what it can replace from the broader engineering project and from the provided benefits of the predictions. Do they save time? Money? 

Here is reported the potential value of **inexpensive predictions**. If we are able to develop a model which predicts a large number of observations at nearly-zero cost, we would be able to automatize the process that before that, was probably done by hand. 

However, the mistakes have to be taken into account. There is always a trade-off to bear in mind. How many mistakes does it make on average? How much does a mistake cost to us?

## Day 03 - Cost of a Machine Learning project

Today we went through the second chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

It covers the **preliminary stage of any ML project**. The key aspect of any project is the prioritization, based on estimated impact and costs. 

The **cost** of a ML project is mainly characterized by the **difficulty** of the problem, the cost of the **data** and the required level of **accuracy**.  The latter is probably the trickiest because the relationship between cost and performances follows a **superlinear growth** 

The **first milestone** regards the difficulty of the problem, starting from already available solutions (if any) and ending with the requested computation power required. The **second milestone** is the data. It is important to understand whether the data is available and how does it cost to build the infrastructure, to label and to maintain the data.

## Day 04 - Complexity of a Machine Learning project

Today we went through the second chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

It covers the **preliminary stage of any ML project**. The key aspect of any project is the prioritization, based on estimated impact and costs.

Estimating the **complexity** of a ML project is everything but easy. To do so, the are different ways. 

- **Compare existing projects:** in can give an idea of the current state of the art
- **Compare human performances**: this can help in order to evaluate a human benchmark in terms of time and performances.
- **Benchmark on simpler scenarios**: try to simplify the problem and check whether if it is solvable. If so, you can start to make the model more complex in order to answer to broader requirements.

## Day 05 - **Why Machine Learning Projects Fail**

Today we went through the second chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

The author proposed **7 reasons why Machine Learning Projects fail** 

- **Lack of experienced talent**: this is still a fairly recent discipline, therefore it is not always easy to hire highly skilled talents, and moreover, even the way employers hire do not always maximize the change of getting the best candidate
- **Lack of support by the leadership**:  sometimes the responsible of the project is not an AI-expert, therefore the real technical limitations cannot easily understood.
- **Missing data infrastructure**: it is always true that “garbage in, garbage out”, therefore it is important to deal with clean and consisted data.
- **Data labeling**: dealing with labeled data is a gold mine nowadays, therefore it is something that has to be taken into account before diving into a new ML project.
- **Lack of collaboration**: siloed companies may have communication issues between their teams, obtaining as a consequence troubles on the integration of smaller pieces of the project
- **Technically infeasible projects**: being ambitious is always important, but the issues arrive when the constraints are not feasible at all.
- **Lack of alignment between Technical and business teams**: the business side has its own requirements, as well as the technical side. However, the meeting point may be hard to reach.

To this extend, we recently published an article called [**“Five reasons why my Machine Learning models failed”**](https://staituned.com/learn/expert/five-reasons-why-my-machine-learning-models-failed), reporting some real examples! 

Reference:

- [https://arxiv.org/pdf/2108.02497.pdf](https://arxiv.org/pdf/2108.02497.pdf)

## **Day 06 - Questions about the data**

Today we went through the third chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

Focusing on the data collection pipeline, the author suggests to **ask 5 questions about the data.**

1. **Is the data accessible?** While looking for some data, it is important to understand whether the **data** is already **available** and as a consequence, if it is protected under **copyright**. Moreover, it is also important to check if the data is sensitive and if there might be some related **privacy** issues.
2. **Is the data sizeable?** Once we have an idea about the data that we can use, we have to understand if the data is **enough** for our project. And if not, we have to check how frequently we can **generate** new data. A **rule of thumb** about the dataset says suggest that we need ~10 times the number of features, ~100 or 1000 times the number of classes and ~10 times the number of trainable parameters (in Deep Learning).
3. **Is the data usable?** We cannot escape from it: garbage in, garbage out! We have to deal with tidy data. Therefore it is important to spot missing values, potential imbalance, expired data and eventual incompleteness of the represented phenomenon. 
4. **Is the data understandable?** It is truly important to understand where each feature and instance comes from, in order to easily manipulate them, if needed.
5. **Is the data reliable?** While dealing with old and historical data, this is something to bear in mind. Do we have some expired data? Maybe some measurement comes from some broken and replaced devices. Moreover, is there any delay between measurement and collected labels?

## Day 07 - Bias on our data

Today we are going through the third chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

Here there is reported a comprehensive **list of biases** that we might encounter on our datasets. Together with this list, you can also find some ways that can reduce this likelihood. 

- The **selection bias** happens when we select data sources that are more likely to be easily available or cheap.
- The **self-selection bias** happens when the data comes from people that “volunteered” to provide it (e.g. reviews or polls).
- The **omitted variable bias** happens when we miss a fundamental feature for the problem at hand.
- The **sampling bias** occurs when the data distribution used in training does not reflect the distribution that will receive in production afterwards.
- The **labeling bias** typically occurs when the labeling is performed from a biased process or person.

This was just a few mentioned example from the book, if you want to know more, I invite you to read the third chapter, it is extremely interesting!

## Day 08 - Causes of data leakage

Today we are going through the third chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

**Data leakage** happens when the information from the validation or the test set are used to train the model. 

There are three main situations in which we may spot it.

The **target is a function of the features**. I guess you all experienced at least once. The simplest case is when you have a copy of your target on the feature space. 

The **features hides the target**. The target can be represented as a code or something similar and it can be directly associated with a perfect match with the outcome. 

We are dealing with **feature from the future**. At inference time we will get only the “current” scenario. However, we may have features that has to be collected in the future that cannot be modeled. The author proposed the example of the “Will pay loan” classification problem in which we have the features “Late Payment Reminders”. Since the prediction will be given before the user starts the loan, it will be always 0!

## Day 09 - Data augmentation in NLP

Today we are going through the third chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

**Data augmentation** is a fairly known technique, especially in Computer Vision, where it is possible to generate new examples by flipping the image, changing the brightness, the color and so on. 

However, in **Natural Language Processing**, this is less straightforward and we may see different interesting techniques. 

1. Replace some words with its closest **synonym** or **hypernym**. The latter is a word having a broader meaning. 
2. In word or document **embedding** applications, we may employ **Gaussian noise** to randomly chosen features, obtaining different similar words. 
3. Given a word to update, use KNN in order to get the K closest words and generate K new different sentences.
4. Given a sentence or a document, translate it into another language and then translate it again into the source language, obtaining a new different sample close to the initial one. This is called **back translation**.

## Day 10 - Data sampling strategies

Today we are going through the third chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

It is not always necessary to deal with all the dataset available. Therefore in these cases it is important to rely on a **meaningful samples**. To this extent, the author proposes three **sampling strategies**.

- **Simple Random Sampling** is probably the easiest. Every selected instance is randomly taken, no matter how. Therefore, every instance has the same probability of being selected (1/N). Despite its simplicity, it may miss some relevant and meaningful instances.
- **Systematic Sampling** starts from the definition of a list containing all the examples. From this list you select the starting point and then decide to pick one instance every “k” elements. In this way we are able to consider instances across the full range of possible ones. However there may be troubles with periodic examples.
- **Stratified Sampling** is probably one of the most used strategy. Here the instances will be sampled based on some defined groups based on the available features. For example we may want to pick 50% of males and 50% of females in order to balance the gender.

## Day 11 - How to convert categorical features to numbers

Today we are going through the fourth chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

Most of the machine learning algorithms are **not able to deal with categorical features**, therefore we have several workarounds to our disposal in order to overcome this limitation. 

- **One Hot Encoding** is extremely popular and it transforms a categorical feature into different binary ones. Hence, if the categorical feature has cardinality N, this will be translated into a N-dimensional binary vector, increasing a lot the final dimension.
- **Bin counting,** or mean encoding,  ****overcomes the size limitation introduced by OHE. It converts every value with its occurrence mean, calculated as the instance frequency/total frequency.
- **Odds ratio** or **log odds ratio** come from statistics and are used for binary classification problems. The odds ratio (OR) quantifies the strength if the association between the two events (A,B) and it is defined as the ratio of the odds of A in the presence of B and the odds of A in the absence of B.

## Day 12 - How to deal with time series

Today we are going through the fourth chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

**Time series data** are different from what we are used to. A time series is an **ordered sequence** of observations. 

Before feeding these data to a Machine Learning algorithm, this need some transformation. The **go-to strategy** involves three simple step: 

1. **Split** the whole time series into smaller segment of a given length 
2. Create a **training example** from each segment 
3. For each training example, calculate the **statistics** observed on the relative segment 

The involved statistics are mainly based on the **domain of interest**, and they can vary from average, spread, returns and so on.

## Day 13 - Four properties of good features

Today we are going through the fourth chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

Have you ever asked yourself whether your **features were “good” or not**? Well, in this chapter we can find four properties that might help answering that question. 

- First and foremost, the features need to have an **high predictive power**, therefore they have to be meaningful with respect to the problem.
- Second, that should be **fast to compute**. Bear in mind that time = money in production, therefore slow and costly features may drastically impact the final result.
- Third, the features have to be **reliable**. Our sources have to be reliable and always available when we need them. Reliability can be also translated into the correctness of the data and their quality.
- Finally, the features must be **uncorrelated**. Correlated features may provide redundant information and will not be beneficial for our model. If we put it in another way, we have one or more features to maintain (higher cost) with no advantages.

Some other bonus features may involve the represented values, based on outliers, distribution and many other factors.

## Day 14 - Discretize features via binning

Today we are going through the fourth chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

Many Machine Learning algorithms are developed or optimized for dealing with discrete features. Therefore it is a common practice su **discretize continuous features**. 

One of the most common methods for discretization is **binning** (or bucketing), which replaces the continuous range numbers with a categorical value. 

The easiest way is via a **uniform binning**, where we have bins having the same size. 

Another way is via **k-means,** where every bin is now a cluster, therefore we’ll have on every bean, the closest values. 

Finally, another way relies on **quantiles**, where each bin will have the same number of samples.

## Day 15 - Feature engineering best practices

Today we are going through the fourth chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

This chapter reports 11 feature engineering best practices, but for the sake of simplicity I will write the ones that I use the most. I invite you to read the chapter in order to discover the remaining ones! 

**Reducing the cardinality** is one of the most common. Sometimes we have to rely on categorical features with thousands of distinct elements. If they are not “crucial” information, it would be idea to reduce the cardinality by grouping similar ones, by removing less frequent cases, or eventually, remove the whole feature directly. 

**Make feature selection when necessary**. In order to make our models simpler, lighter and more explainable, it would be ideal to select the right number of features, removing the ones that we may consider “less useful” 

**Isolate feature extraction code**. This step of any MLE pipeline has to be independent from the rest. Further step must nos strictly rely on it.

## Day 16 - Choose the right baseline

Today we are going through the fifth chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

Any Machine Learning pipeline need a **baseline** to compare. This gives an analyst the feeling that the problem is solvable. 

We can take as a reference the outcome of some heuristic algorithm or the human performances. However, there are **two main ways** to approach a baseline: 

- **Random prediction**: we consider a random label between the possible alternatives.
- **Zero-rule algorithms**: this is every easier than the random one. In classification it always predict the most common class seen in training whereas in regression it always predict the average observed in training.

## Day 17 - How to select the learning algorithm

Today we are going through the fifth chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

A crucial aspect of any Machine Learning pipeline is the **choice of the learning algorithm**. Thanks to the No Free Launch Theorem, we know that there is no a “universal learner”, therefore we have to look for the right one for a given task. 

Instead of tying all the possible algorithms, we can start from our requirements:

- **Explainability**: we can choose between less performing but more explainable models and vice versa.
- **In memory vs out-of-memory**: If we cannot load all the data in memory, we might opt for incremental learning algorithms.
- **Number of features**: some algorithms are more sensitive to the dimensionality of the feature space.
- **Nonlinearity of the data**: based on the nonlinearity of the data, we can select different algorithms. The classical example relies on SVM with its linear or not-linear kernels.
- **Speed**: this has to be considered. Speed means money in a company. Therefore, we have to take into account how much the speed impact the overall project.

## Day 18 - Hyperparameters tuning

Today we are going through the fifth chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

Once you selected a model, the default configuration is not likely to produce satisfactory results. Therefore it is important to **evaluate different hyperparameters** and see how it behaves. 

There are three main strategies, two of which are well-known:

- **Grid search**:  defines a search space as a grid of hyperparameter values and evaluate every position in the grid (hence, every combination). It is typically performed when the training time is not so long.
- **Random search** defines a search space as a bounded domain of hyperparameter values and randomly sample points in that domain.
- **Coarse-to-fine search** is a combination of the previous two. It starts with a random search in order to find the regions with an higher potential. Then, once the region is spotted, it leverages a grid search in order to select the best combination.

## Day 19 - Distribution shift

Today we are going through the sixth chapter of the “Machine Learning Engineering” book by Andriy Burkov. 

Sometimes the data that we observe in production is different from what we used in training. Therefore, our model evaluations might be a bit misleading. This is called **distribution shift**. 

This is still an open problem in the research, but we can distinguish between three kind of distribution shifts:

- **Covariance shift** : shift in the values of features
- **Prior probability shift**: shift in the values of the target
- **Concept drift**: shift in the relationship between the features and the label

An approach commonly used to mitigate this phenomenon is the **adversarial validation**. This allows to estimate whether the training and test data are similar. 

To do so, we **halve** our training data. The **first half** will use the old label as an additional feature and it will be replaced with the label “**train**”. Then, a modified test set will do the same, using the **label** “**test**”.  Afterwards, we use this data settings for **training a binary classification algorithm** that should be able to re-identify them. 

Finally, we **apply** this model to the **second half** of the training set. From here, we **select** the **samples** which are more **similar** to the “**test set**” (in terms of scoring) and we’ll use them as a **validation** set for the original problem. Pretty cool, right?


<br />
<br />
