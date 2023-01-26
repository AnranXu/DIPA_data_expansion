import os
import csv
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.formula.api import ols
import torch
import torch.nn as nn

class analyzer:
    def __init__(self) -> None:
        self.annotation_path = './annotations/'
        self.platforms = ['CrowdWorks', 'Prolific']
        self.img_annotation_map_path = './img_annotation_map.json'
        self.img_annotation_map = {}
        self.code_openimage_map = {}
        self.openimages_mycat_map = {}
        self.lvis_mycat_map = {}

        with open(self.img_annotation_map_path) as f:
            self.img_annotation_map = json.load(f)
        with open('./mycat_lvis_map.csv') as f:
                res = csv.reader(f)
                flag = 0
                for row in res:
                    if not flag:
                        flag = 1
                        continue
                    lvis_cats = row[1].split('|')
                    if 'None' in row[1]:
                        continue
                    for cat in lvis_cats:
                        self.lvis_mycat_map[cat] = row[0]
        with open('./oidv6-class-descriptions.csv') as f:
            res = csv.reader(f)
            for row in res:
                self.code_openimage_map[row[0]] = row[1]
        with open('./mycat_openimages_map.csv') as f:
            res = csv.reader(f)
            flag = 0
            for row in res:
                if not flag:
                    flag = 1
                    continue
                openimages_cats = row[1].split('|')
                if 'None' in row[1]:
                    continue
                for cat in openimages_cats:
                    category_name = self.code_openimage_map[cat]
                    self.openimages_mycat_map[category_name] = row[0]

    def prepare_mega_table(self, mycat_mode = True, save_csv = False)->None:
        #mycat_mode: only aggregate annotations that can be summarized in mycat (also store them in mycat in mega_table).
        #the mega table includes all privacy annotations with all corresponding info (three metrics, big five, age, gender, platform)

        # make sure this sequence is correct.
        self.mega_table = pd.DataFrame(columns=["category", "reason", "informativeness", "sharing", 'age', 'gender', 
        'platform', 'extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness'])
        for key in self.img_annotation_map.keys():
            for platform, value in self.img_annotation_map[key].items():
                # now, value[0] is the only availiable index
                prefix_len = len(value[0].split('_')[0]) + 1
                worker_file = value[0][prefix_len:]
                worker_file = worker_file[:-11]
                worker_file = worker_file + '.json'
                with open(os.path.join(self.annotation_path, platform, 'workerinfo', worker_file)) as f_worker, \
                open(os.path.join(self.annotation_path, platform, 'labels', value[0])) as f_label:
                    worker = json.load(f_worker)
                    label = json.load(f_label)
                    # we only analyze default annotations
                    age = worker['age']
                    gender = worker['gender']
                    extraversion = worker['bigfives']['Extraversion']
                    agreeableness = worker['bigfives']['Agreeableness']
                    conscientiousness = worker['bigfives']['Conscientiousness']
                    neuroticism = worker['bigfives']['Neuroticism']
                    openness = worker['bigfives']['Openness to Experience']
                    dataset_name = label['source']             
                    for key, value in label['defaultAnnotation'].items():
                        if value['ifNoPrivacy']:
                            continue
                        category = ''
                        if mycat_mode:
                            if dataset_name == 'OpenImages':
                                if key in self.openimages_mycat_map.keys():
                                    category = self.openimages_mycat_map[key]
                            elif dataset_name == 'LVIS':
                                if key in self.lvis_mycat_map.keys():
                                    category = self.lvis_mycat_map[key]
                            if category == '':
                                continue
                        else:
                            category = value['category']
                        reason = value['reason']
                        informativeness = value['informativeness']
                        sharing = value['sharing']
                        entry = pd.DataFrame.from_dict({
                            "category": [category],
                            "reason":  [reason],
                            "informativeness": [informativeness],
                            "sharing": [sharing],
                            "age": [age],
                            "gender": [gender],
                            "platform": [platform],
                            "extraversion": [extraversion],
                            "agreeableness": [agreeableness],
                            "conscientiousness": [conscientiousness],
                            "neuroticism": [neuroticism],
                            "openness": [openness]
                        })

                        self.mega_table = pd.concat([self.mega_table, entry], ignore_index=True)
        if save_csv:
            self.mega_table.to_csv('./mega_table.csv', index =False)

    def svm(self, input_channel, output_channel, read_csv = False) -> None:
        if read_csv:
            self.mega_table = pd.read_csv('./mega_table.csv')
        else:
            self.prepare_mega_table()
        self.mega_table['category'] = encoder.fit_transform(self.mega_table['category'])
        self.mega_table['gender'] = encoder.fit_transform(self.mega_table['gender'])
        self.mega_table['platform'] = encoder.fit_transform(self.mega_table['platform'])
        scaler = StandardScaler()
        encoder = LabelEncoder()
        X = self.mega_table[input_channel].values
        y = self.mega_table[output_channel].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=0)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        classifier=svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.8)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)

        # Print evaluation metrics
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Precision:",metrics.precision_score(y_test, y_pred,average='weighted'))
        print("Recall:",metrics.recall_score(y_test, y_pred,average='weighted'))

    def anova(self,read_csv = False) -> None:
        ## the degree of freedom of "informativeness" is wrong, it should be 6 rather than 1
        ## I am using R to perform this
        if read_csv:
            self.mega_table = pd.read_csv('./mega_table.csv')
        else:
            self.prepare_mega_table()
        print(self.mega_table['informativeness'].unique())
        model = ols('sharing ~ category*informativeness', data=self.mega_table).fit()
        aov_table = sm.stats.anova_lm(model, typ=1)
        print(aov_table)

if __name__ == '__main__':
    analyze = analyzer()
    input_channel = ['category', 'reason', 'informativeness']
    output_channel = ['sharing']
    #analyze.svm(input_channel, output_channel, read_csv=True)
    #analyze.anova(True)
    