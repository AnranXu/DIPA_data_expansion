**All inputs**: ['age', 'gender', 'platform', 'category', 'extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']

**Three metrics**: ['reason', 'precision', 'sharing']

Simple neural network model (**All inputs -> Three metrics**):

|                 | accuracy | precision | recall | F1 score |
| --------------- | -------- | --------- | ------ | -------- |
| reason          | 0.729    | 0.745     | 0.729  | 0.728    |
| informativeness | 0.450    | 0.460     | 0.450  | 0.442    |
| sharing         | 0.660    | 0.675     | 0.660  | 0.659    |

SVM (**All inputs -> Three metrics**):

kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.8

|                 | accuracy | precision | recall | F1 score |
| --------------- | -------- | --------- | ------ | -------- |
| reason          | 0.621    | 0.507     | 0.621  | 0.538    |
| informativeness | 0.326    | 0.303     | 0.326  | 0.269    |
| sharing         | 0.584    | 0.619     | 0.584  | 0.533    |



KNN (**All inputs -> Three metrics**):

n_neighbors=5

|                 | accuracy | precision | recall | F1 score |
| --------------- | -------- | --------- | ------ | -------- |
| reason          | 0.653    | 0.607     | 0.653  | 0.613    |
| informativeness | 0.415    | 0.408     | 0.415  | 0.410    |
| sharing         | 0.632    | 0.624     | 0.632  | 0.623    |





