---
title: 30 Days of Machine Learning Engineering
author: Francesco Di Salvo
date: 2022-11-14
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


<br />
<br />