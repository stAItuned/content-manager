---
meta: Simulations are able to generate random variables with certain properties and hence being able to mimic real random phenomena.
date: 
target: Expert 
topics: [Bayesian Inference,MCMC,Simulation] 
cover: filename of the cover image, located next to the markdown file 
title: Simulation in Bayesian Inference
language: English 
author: Gabriele Cola
---

## Simulation in Bayesian Inference

The most of times Simulation= Monte Carlo, in particular this methods are a broad a class of computational algorithms that rely on repeated random sampling to obtain numerical results. They are often used when its difficult(or impossibile) to calculate a quantity of interest analytically. 

There are many applications of Simulation, for example In Bayesian Inference is used to approximate the posterior distribution. In many cases in order to to do it we require more powerful methods rather than the simple Monte Carlo methods, so we use MCMC( Monte Carlo Markov Chain).

### Posterior distribution simulated via MCMC
The MCMC requires to specify a lot of parameters , such as:
* Burn-In period: The preliminary steps, during which the chain moves from its unrepresentative initial value to the modal region of the posterior
*	Chains: MCMC sampling, values are drawn from a probability distribution, so this values forms chain.So, in order to  to evaluate the convergence of MCMC chains it is helpful to create multiple chains that have different starting values
*	Thinning: The parameter thin allows the user to specify if and how much the MCMC chains should be thinned out before storing them, in order to decrease the autocorrelations

The graph here, illustrate one of the main graphical tool used to see if the MCMC has converged, **”The trace plot”**. We adopt it in order to see if there is stability in the long run behaviour of this chain. In this particular case we want to see if there is a good approximation of the true posterior distribution of parameter **$\sigma$**.

<p align="center">
  <img src="https://user-images.githubusercontent.com/103529789/196060497-b689b975-eb99-4430-bf09-8f87641c86eb.png" width="550"/>
</p>

In this second graph, we are looking for the shape of the distribution of the **$\sigma$** parameter, so we use a **density plot**. Furthemore, we see the distribution for each increasing number of chains in order to see how it changes.

<p align="center">
  <img src="https://user-images.githubusercontent.com/103529789/196060679-2191a8c5-2967-49d4-8078-fc60935ab1a6.png" width="550"/>
</p>

### The role of simulation in Machine Learning
MCMC techniques are often applied to solve integration and optimisation problems in large dimensional spaces. These two types of problem play a fundamental role in machine learning.
Furthermore, Bayesian strategies help various machine learning algorithms in removing pivotal data from little informational indexes and taking care of missing information. They assume a significant function in an immense scope of territories from game improvement to sedate disclosure. 
Bayesian strategies empower the assessment of uncertainty in forecasts which demonstrates crucial for fields like medication. The techniques assist setting aside with timing and cash by permitting the pressure of deep learning algorithms a hundred folds and naturally tuning hyperparameters.



