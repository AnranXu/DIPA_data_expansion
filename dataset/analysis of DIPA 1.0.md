### Different Input/Output Groups

**All basic inputs**: ['age', 'gender', 'platform', 'category', 'extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']

**Basic info**: ['age', 'gender', 'platform'] 

**Bigfive**: ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']

**Category**: ['category']

**Three metrics**: ['reason', 'informativeness', 'sharing']

**Objective metrics**: ['reason', 'informativeness']

**Subjective metrics**: ['sharing']

### ANOVA test: Objective metrics -> Subjective metrics: 

I run two-way anova test to verify if there exist possitive correlationship bewteen objective metrics (reason, informativeness) to subjective metrics (sharing).

I found a significant effect of **informativeness** on **maximum sharing scope** (F(6, 3020)=91.84, p<0.01, effect size = .154).

I also found a small effect of **reason** on **maximum sharing scope**.

Although we can confirm **informativeness** can be a key factor to predict **maximum sharing scope**, we need to know two other things:

1. How exact the **informativeness** can be related to **maximum sharing scope**? How much we can use **informativeness** to predict **maximum sharing scope**?
2. If other factors influence the prediction of all (a part of) **three metrics**.



For analysis of multiple factors, I built prediction models for this because Analysis like 'ANOVA', 'Chi-square', and etc. only support limited number of input and output.

### Test 1: All basic inputs -> Three metrics

I prepared three models to test their performance by recognizing all three metrics we collect for each image.

Simple neural network model (4-layer full connection network):

|                 | accuracy | precision | recall | F1 score |
| --------------- | -------- | --------- | ------ | -------- |
| reason          | 0.729    | 0.745     | 0.729  | 0.728    |
| informativeness | 0.450    | 0.460     | 0.450  | 0.442    |
| sharing         | 0.660    | 0.675     | 0.660  | 0.659    |

<div style="display:incline-block; justify-content:center;">   <img src="./analysis results/All inputs -> Three metrics/confusion matrix for reason.png" alt="image1" style="max-width: 500px; max-height: 500px; flex:1; margin: 10px;">   <img src="./analysis results/All inputs -> Three metrics/confusion matrix for informativeness.png" alt="image2" style="flex:1; margin: 10px;max-width: 500px; max-height: 500px;">   <img src="./analysis results/All inputs -> Three metrics/confusion matrix for sharing.png" alt="image3" style="flex:1; margin: 10px;max-width: 500px; max-height: 500px;"> </div>



SVM:

kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.8

|                 | accuracy | precision | recall | F1 score |
| --------------- | -------- | --------- | ------ | -------- |
| reason          | 0.621    | 0.507     | 0.621  | 0.538    |
| informativeness | 0.326    | 0.303     | 0.326  | 0.269    |
| sharing         | 0.584    | 0.619     | 0.584  | 0.533    |



KNN:

n_neighbors=5

|                 | accuracy | precision | recall | F1 score |
| --------------- | -------- | --------- | ------ | -------- |
| reason          | 0.653    | 0.607     | 0.653  | 0.613    |
| informativeness | 0.415    | 0.408     | 0.415  | 0.410    |
| sharing         | 0.632    | 0.624     | 0.632  | 0.623    |



As the simple neural network performed the best, I will only use it to run the following test.

### Test 2: Basic info + Category -> Three metrics

"**Sharing**" is significantly influenced by "**Bigfive**".  

|                 | accuracy | precision | recall | F1 score |
| --------------- | -------- | --------- | ------ | -------- |
| reason          | 0.722    | 0.706     | 0.722  | 0.708    |
| informativeness | 0.351    | 0.318     | 0.351  | 0.316    |
| **sharing**     | 0.553    | 0.563     | 0.553  | 0.539    |

<div style="display:incline-block; justify-content:center;">   <img src="./analysis results/Basic info + Category -> Three metrics/confusion matrix for reason.png" alt="image1" style="max-width: 500px; max-height: 500px; flex:1; margin: 10px;">   <img src="./analysis results/Basic info + Category -> Three metrics/confusion matrix for informativeness.png" alt="image2" style="flex:1; margin: 10px;max-width: 500px; max-height: 500px;">   <img src="./analysis results/Basic info + Category -> Three metrics/confusion matrix for sharing.png" alt="image3" style="flex:1; margin: 10px;max-width: 500px; max-height: 500px;"> </div>



### Test 3: Category -> Three metrics

### Test 4: Category -> Three metrics