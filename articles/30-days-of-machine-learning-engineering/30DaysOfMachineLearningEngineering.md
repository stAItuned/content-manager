---
title: 30 Days of Machine Learning Engineering
author: Francesco Di Salvo
date: 2022-11-23
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

<br />
<br />