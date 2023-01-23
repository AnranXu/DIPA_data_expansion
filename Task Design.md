# Task Design

## Task Preview Link

 https://anranxu.github.io/DIPA_data_expansion/?lg=en&test=true

## Overview

**task per worker**: 20

**task requirement**: Only annotating labels were annotated as privacy-threatening in DIPA 1.0 (1,495 images in total, non-privacy-threatening content will be removed in advance). Workers need to provide three answers to each privacy-threatening content they have identified, including "reason, informativeness, and maximum sharing scope". Also, we allow workers to add manual labels.

## Current Metrics

- **Reason for identifying content as privacy-threatening**
  - personal identity 
  - location of shooting
  - personal habits
  - social circle
  - others
- **Informativeness of the privacy-threatening content**
  - From 1 to 5, presenting in Star.  (last version: from 1 to 7, in descriptions)
- **Maximum sharing scope if you are the photo owner** 
  - I won't share it
  - Family or friend
  - Public
  - Broadcast program
  - Others

## Discussion

**Existing metrics**:

- **Reason for identifying content as privacy-threatening**

  - Other papers concluded that reasons like "**photo quality**" and "**illegal**" as the reasons to identify a photo to be sensitive. However, I do not want to add these options as they are subjective. "Personal habits" can include those things. Or we can say "**personal habits/life**". I wonder if you have a better idea to rename this option. I think others are fine.

- **Informativeness of the privacy-threatening content**

  - Using "uninformative" to describe how much information a privacy-threatening content might be not suitable. I change the description to only mention this metric as "informative" and how informative it is. As adverbs (e.g. slightly, little, moderate)  can be implied differently by different people, I choose to use rating scores in the next data collection. Also, changing 7-Likert to 5-Likert can avoid ambiguity in near scores (e.g. find it difficult to decide to choose 3 or 4).

- **Maximum sharing scope if you are the photo owner** 

  - Many other papers use different recipient groups (e.g. Colleagues & Classmates, supervisors, normal friends, and relatives). It is not practical to include all possible groups in our data collection because it will be too complex. Also, different people may hold different opinions when we tell them about a specific group of recipients. For example, for family or colleagues, somebody might think it can be a close relationship while others just want to avoid showing information to them. Sometimes, people do not want to share photos of their close relationships but choose to share with strangers. So, actually, it is not objective when we use a specific group to refer to different scales of sharing; or assuming that if people want to share a photo to the public, they must want to share it to closer relationships. I do not have a concrete idea to solve this problem. But I think we need to clearly show that the scale of provided choices is a recursive increase.

  - Design 1: **How many people are you willing to share this privacy-threatening content at most?**
  
    - 0
    - 1-10
    - 11-100
    - 101-1000
    - 1000+
  
  - Design 2: (Multiple choices available) **Which group of people do you want to share if you are the photo owner** 
  
    - I won't share it. 
    - close relationship 
    - normal relationship 
    - Unfamiliar people
    - Public 
    - Broadcast program
    - Others
  
  - Design 3: **Willingness to share** 
  
    - From 1 to 5
  
    ​	

**New possible metrics**:

- ​	**If the privacy-threatening content is related to you and someone else wants to share this image, to what extent would you share this content at most?**
  - I won't share it
  - Family or friend
  - Public
  - Broadcast program
  - Others

This can be useful when we analyze "bystander sharing" (i.e. sharing photos without getting approvals from stakeholders) behavior. 



**Manual labels:**

Although it is meaningful to allow workers to add manual labels, analyzing or using them can be difficult. 

- Half of the annotations will be written in Japanese.
- Even for the same privacy-threatening content, the bounding boxes created by different people will be different.

If time allows, we can manually unify these manual labels as there will not be too many.