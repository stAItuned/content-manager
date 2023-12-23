---
title: Increase your productivity with your own PyTorch template
author: Francesco Di Salvo
topics: [Deep Learning, PyTorch, Productivity]
target: Expert
language: English
cover: productivity.webp
meta: Enhance productivity with a custom PyTorch deep learning pipeline. Streamline model experimentation, secure reproducibility, and tailor to your needs.
date: 2023-02-21
published: true
---

# Increase your productivity with your own PyTorch template

Over the last years I've built several deep learning projects in PyTorch, either for work or for academic projects. What I realized was that **I was actually writing and writing the same things**, except for the dataset class, model architecture, custom losses and a few minor things.

Therefore, in order to avoid writing always the same pipeline, we can actually our own **PyTorch template**, that can be reused with minimal modifications. 

In this blog post, we will discuss the **benefits** of having your own PyTorch template for deep learning pipelines and how it can increase your productivity.

## Benefits

Below I will report three major benefits of having your own template:

- **Faster experimentation**: this allows you to quickly experiment different dataset, architectures and loss functions with a relatively low effort.
- **Reproducibility**: with a robust and consistent template, you can ensure that your experiments will be reproducible. 
- **Customization**: if you build your own template, tailored for your needs, you are free to add all the methods, plots and callbacks that you need on your day-to-day projects, avoiding to copy and paste the same snippets of code between older repositories. 

Of course there will be more than three benefits, and I am looking forward to hearing from you! 

## My personal setup

I shared my personal template in [this repository](https://github.com/francescodisalvo05/deep-learning-pytorch-template). As you can see, the overall structure is as follows:

```
└─── assets/ : store results, images or major checkpoints
|
└─── dataset/ : store the dataset involved on your tests
|
└─── src/ 
└───────── data/ 
└───────────────── dataset.py : dataset class
└───────────────── transformations.py : data augmentation pipeline
└───────── losses/
└───────── metrics/ 
└───────── models/
└───────────────── model.py : custom model
└───────────────── trainer.py : trainer class
└───────── utils/ 
└───────────────── plots.py : plots sample of images, metrics and so on
└───────────────── utils.py : set seed, early stopping and so on
|
└─── .gitignore
└─── README.md
└─── settings.yaml : settings for the current experiment 
└─── train.py : training pipeline with logging, cuda and tensorboard support
```

I was inspired by the wonderful work of [Francesco Saverio Zuppichini](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Template) and I will keep my template up-to-date with further features along the way! 

## Conclusions
This blogpost reported the main advantages of building your PyTorch or Tensorflow template instead of copying and pasting many times from your older projects. My personl setup is still a work-in-progress and I would be happy to receive comments and feedbacks based on your professional experience! 


### References
- [Repository](https://github.com/francescodisalvo05/deep-learning-pytorch-template)
- Inspired by [this repository](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Template)
- [Cover image](https://www.pexels.com/it-it/foto/mani-laptop-macbook-connessione-4068322/)