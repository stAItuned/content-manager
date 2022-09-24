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
  - any other assets referenced by the article must be located directly into the [Article-Folder], next to the markdown file

- team/[Name-Surname] (without spaces, used as slug)
  - propic.jpg
  - meta.md

> The profile picture needs to have squared dimensions (e.g. 1024x1024).

### ⌨️ Article meta
```YAML
---
meta: brief description of the blogpost 
date: current date 
target: Newbie | Midway | Expert 
topics: [Deep Learning, List separated by commas] 
cover: filename of the cover image, located next to the markdown file 
title:  
language: English | Italian
author: Name must exist
---
```
> This piece of code has to be placed at the beginning of the page.
> See an example [here](https://github.com/stAItuned/articles/tree/main/articles/machine-learning-intro).

### 🙋🏼‍♀️ 🙆🏽‍♂️ Team member meta
```YAML
---
name: name and surname
team: [...] # this is an array -> the options are [Writers, Tech, Marketing]
title: job title  
linkedin: 
email: 
description: brief description about yourself
---
```

> This piece of code has to be placed at the beginning of the page.
> See an example [here](https://github.com/stAItuned/articles/blob/main/team/Francesco-Di-Salvo/meta.md).

> You can be part of the `Writers` team only if you already published an article. `Tech` team members actively work on the IT infrastructure and maintainance of this platform whereas `Marketing` members actively work on the graphics and marketing strategies of our social media accounts.

### 🤔 Any question? 
Raise an [issue](https://github.com/stAItuned/content-manager/issues) or contact us through our platforms! 


## To do:
- [ ] Specify an ideal dimension for the cover, in order to fit the cards and the home preview 
- [x] Update repo name? (and issue's link)
- [ ] Add template commit? 
- [ ] Add document structure? 
- [ ] Briefly explain the "goal" of this repo and add the link of our website and social account (when they'll be ready)
- [x] Specify who belong to "Tech", "Marketing" and "Writers" team

