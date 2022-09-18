# Content manager
---

## 📩 How to submit
1. Fork this repository
2. Upload your blog post and relative images through the github markdown format /articles/
3. If you're a new author, upload also your image and description under /team/
4. Pull request of the desired changes
5. One of our team members will review the article and he/she may accept your post or eventually propose you some little adjustments. 

## 🛠 Proposed repository structure
- articles/[Article-Folder] (without spaces, used as slug)
  - .md file containing the article
  - any other assets references by the article must be referenced relatively to [Article-Folder]

- team/[Name-Surname] (without spaces, used as slug)
  - propic.jpg (or any other image format)
  - meta.md

### ⌨️ Article meta
```YAML
---
meta: brief description of the blogpost 
date: current date 
slug: id of the website (staituned/articles/[slug] )
target: Newbie | Midway | Expert 
topics: e.g. Deep Learning 
cover: image path of the cover image 
title:  
author:  
---
```
> This piece of code has to be placed at the beginning of the page.
> See an example [here](https://github.com/stAItuned/articles/tree/main/articles/machine-learning-intro).

### 🙋🏼‍♀️ 🙆🏽‍♂️ Team member meta
```YAML
---
name: name and surname
team: [...] # this is an array -> the options are [Content writer, Tech, Marketing]
title: job title  
linkedin: 
email: 
description: brief description about yourself
---
```

> This piece of code has to be placed at the beginning of the page.
> See an example [here](https://github.com/stAItuned/articles/blob/main/team/Francesco-Di-Salvo/meta.md).

### 🤔 Any question? 
Raise an [issue](https://github.com/stAItuned/articles/issues) or contact us through our platforms! 


## To do:
- [ ] Specify an ideal dimension for the cover, in order to fit the cards and the home preview 
- [ ] Update repo name? (and issue's link)
- [ ] Add template commit? 
- [ ] Add document structure? 
- [ ] Briefly explain the "goal" of this repo and add the link of our website and social account (when they'll be ready)
