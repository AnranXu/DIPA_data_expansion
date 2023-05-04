import os
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from scipy.stats import binom

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class analyzer:
    def __init__(self) -> None:
        self.annotation_path = './annotations/'
        self.platforms = ['CrowdWorks', 'Prolific']
        self.img_annotation_map_path = './img_annotation_map.json'
        self.img_annotation_map = {}
        self.code_openimage_map = {}
        self.openimages_mycat_map = {}
        self.lvis_mycat_map = {}
        self.test_size = 0.2
        self.custom_informationType = []
        self.custom_recipient_owner = []
        self.custom_recipient_others = []
        self.description = {'informationType': ['personal information', 'location of shooting',
        'individual preferences/pastimes', 'social circle', 'others\' private/confidential information', 'Other things'],
        'informativeness':['Strongly disagree','Disagree','Slightly disagree','Neither',
        'Slightly agree','Agree','Strongly agree'],
        'sharingOwner': ['I won\'t share it', 'Close relationship',
        'Regular relationship', 'Acquaintances', 'Public', 'Broadcast program', 'Other recipients'], 
        'sharingOthers':['I won\'t allow others to share it', 'Close relationship',
        'Regular relationship', 'Acquaintances', 'Public', 'Broadcast program', 'Other recipients'],
        'frequency': ['Never', 'Less than once a month', 'Once or more per month', 
        'Once or more per week', 'Once or more per day']}
        self.mega_table_path = './mega_table (strict).csv'
        if not os.path.exists(self.img_annotation_map_path):
            self.generate_img_annotation_map()
        with open(self.img_annotation_map_path) as f:
            self.img_annotation_map = json.load(f)
        with open('./DIPA_lvis_map.csv') as f:
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
        with open('./DIPA_openimages_map.csv') as f:
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
    def basic_info(self, platform='CrowdWorks')->None:
        age = {'18-24': {'Male': 0, 'Female': 0, 'Other': 0}, 
        '25-34': {'Male': 0, 'Female': 0, 'Other': 0}, 
        '35-44': {'Male': 0, 'Female': 0, 'Other': 0}, 
        '45-54': {'Male': 0, 'Female': 0, 'Other': 0}, 
        '55': {'Male': 0, 'Female': 0, 'Other': 0}}
        valid_workers = []
        with open(os.path.join(self.annotation_path, platform, 'valid_workers.json')) as f:
            valid_workers = json.load(f)
        info_paths = os.listdir(os.path.join(self.annotation_path, platform, 'workerinfo'))
        for info_path in info_paths:
            # check if nan, nan!=nan
            with open(os.path.join(self.annotation_path, platform, 'workerinfo', info_path)) as f:
                text = f.read()
                info = json.loads(text)
                if info['workerId'] not in valid_workers:
                    print('unvalid worker: ', info['workerId'])
                    continue
                if int(info['age']) != int(info['age']):
                    print('wrong age found', info)
                    continue
                year = int(info['age'])
                if year < 18:
                    print('wrong age found', info)
                if 18 <= year <= 24:
                    age['18-24'][info['gender']] += 1
                elif 25 <= year <= 34:
                    age['25-34'][info['gender']] += 1
                elif 35 <= year <= 44:
                    age['35-44'][info['gender']] += 1
                elif 45 <= year <= 54:
                    age['45-54'][info['gender']] += 1
                elif 55 <= year:
                    age['55'][info['gender']] += 1
                valid_workers.remove(info['workerId'])
        print('valid worker:', len(valid_workers))
        print(valid_workers)
        print(age)

    def generate_img_annotation_map(self)->None:
        #label: the original label from OpenImages or LVIS 
        #annotation: the privacy-oriented annotations from our study
        img_annotation_map = {}
        valid_workers = []
        crowdworks_labels = os.listdir(os.path.join(self.annotation_path, 'CrowdWorks', 'labels'))
        with open(os.path.join(self.annotation_path, 'CrowdWorks', 'valid_workers.json')) as f:
            valid_workers = json.load(f)
        print('val len:', len(valid_workers))
        
        for label_path in crowdworks_labels:
            img_name = label_path.split('_')[0]
            prefix_len = len(img_name) + 1
            worker_name = label_path[prefix_len:]
            worker_name = worker_name[:-11]
            if worker_name not in valid_workers:
                continue
            if img_name != '':
                if img_name not in img_annotation_map.keys():
                    img_annotation_map[img_name] = {}
                if 'CrowdWorks' not in img_annotation_map[img_name].keys():
                    img_annotation_map[img_name]['CrowdWorks'] = [label_path]
                else:
                    img_annotation_map[img_name]['CrowdWorks'].append(label_path)

        prolific_labels = os.listdir(os.path.join(self.annotation_path, 'Prolific', 'labels'))
        with open(os.path.join(self.annotation_path, 'Prolific', 'valid_workers.json')) as f:
            valid_workers = json.load(f)
        print('val len:', len(valid_workers))
        for label_path in prolific_labels:
            img_name = label_path.split('_')[0]
            prefix_len = len(img_name) + 1
            worker_name = label_path[prefix_len:]
            worker_name = worker_name[:-11]
            if worker_name not in valid_workers:
                continue
            if img_name != '':
                if img_name not in img_annotation_map.keys():
                    img_annotation_map[img_name] = {}
                if 'Prolific' not in img_annotation_map[img_name].keys():
                    img_annotation_map[img_name]['Prolific'] = [label_path]
                else:
                    img_annotation_map[img_name]['Prolific'].append(label_path)

        with open('img_annotation_map.json', 'w') as w:
            json.dump(img_annotation_map, w)
            
    def distribution(self, read_csv = False, strict_mode = False, strict_num = 2)->None:
        #distribution of privacy and not privacy annotations in each category
        #if it is privacy, then calculate how many times for each object is annotated as privacy

        ## calculate amount of all category 
        category_and_id = {}
        category_number = {}
        for image_name in self.img_annotation_map.keys():
            for platform, annotations in self.img_annotation_map[image_name].items():
                for i, annotation in enumerate(annotations):
                    if strict_mode and i >= strict_num:
                        break
                    with open(os.path.join(self.annotation_path, platform, 'labels', annotation), encoding="utf-8") as f:
                        label = json.load(f)
                        dataset_name = label['source']
                        for key, value in label['defaultAnnotation'].items():
                            if dataset_name == 'OpenImages':
                                if key in self.openimages_mycat_map.keys():
                                    category = self.openimages_mycat_map[key]
                                else:
                                    category = 'others'
                            elif dataset_name == 'LVIS':
                                if key in self.lvis_mycat_map.keys():
                                    category = self.lvis_mycat_map[key]
                                else:
                                    category = 'others'
                            object_id = image_name + '_' + key + '_' + category
                            if category not in category_and_id.keys():
                                category_and_id[category] = {}
                            category_and_id[category][object_id] = 0

        for key, value in category_and_id.items():
            category_number[key] = len(value)
        
        # add up category number
        tot_num = 0
        for key, value in category_number.items():
            tot_num += value

        print(category_number)
        print('total:', tot_num)

        # calculate amount of privacy category
        privacy_category = {'CrowdWorks':{}, 'Prolific':{}, 'All': {}}
        privacy_num = {'CrowdWorks':{}, 'Prolific':{}, 'All': {}}
        if read_csv:
            self.mega_table = pd.read_csv(self.mega_table_path)
        else:
            self.prepare_mega_table()

        exclude_manual = self.mega_table[self.mega_table.category != 'Manual Label']
        # access each row of exclude_manual
        for index, row in exclude_manual.iterrows():
            dataset_name = row['datasetName']
            platform = row['platform']
            key = row['category']
            image_name = row['imagePath'][:-4]
            if dataset_name == 'OpenImages':
                if key in self.openimages_mycat_map.keys():
                    category = self.openimages_mycat_map[key]
                else:
                    category = 'others'
            elif dataset_name == 'LVIS':
                if key in self.lvis_mycat_map.keys():
                    category = self.lvis_mycat_map[key]
                else:
                    category = 'others'
            object_id = image_name + '_' + key + '_' + category
            if category not in privacy_category[platform].keys():
                privacy_category[platform][category] = {}
            if category not in privacy_category['All'].keys():
                privacy_category['All'][category] = {}

            if object_id not in privacy_category[platform][category].keys():
                privacy_category[platform][category][object_id] = 1
            else:
                privacy_category[platform][category][object_id] += 1

            if object_id not in privacy_category['All'][category].keys():
                privacy_category['All'][category][object_id] = 1
            else:
                privacy_category['All'][category][object_id] += 1
        
        for platform, category in privacy_category.items():
            for key, value in category.items():
                privacy_num[platform][key] = {1: 0, 2: 0, 3: 0, 4: 0}
                for object_id, num in value.items():
                    privacy_num[platform][key][num] += 1
        
        #Add up privacy number of All
        tot = {'CrowdWorks': {1: 0, 2: 0, 3: 0, 4: 0},
                'Prolific': {1: 0, 2: 0, 3: 0, 4: 0},
                'All': {1: 0, 2: 0, 3: 0, 4: 0}}
        for platform in privacy_num.keys():
            for key, value in privacy_num[platform].items():
                if key == 'others':
                    continue
                tot[platform][1] += value[1]
                tot[platform][2] += value[2]
                tot[platform][3] += value[3]
                tot[platform][4] += value[4]

        print('CrowdWorks:')
        print(privacy_num['CrowdWorks'])
        print('Prolific:')
        print(privacy_num['Prolific'])
        print('All:')
        print(privacy_num['All'])

        print('total:', tot)

        # #binomial test
        # p = (tot['CrowdWorks'][1] + tot['CrowdWorks'][2]) / tot_num
        # num_successes = tot['Prolific'][1] + tot['Prolific'][2]
        # # Number of trials
        # num_trials = tot_num
        # p_value = binom(num_trials, p).sf(num_successes - 1) + binom(num_trials, p).cdf(num_successes)

        # # Calculate 95% confidence interval
        # lower, upper = binom(num_trials, p).interval(0.95)

        # print("p-value:", p_value)
        # print("95% CI:", lower, upper)
        

    def prepare_mega_table(self, mycat_mode = True, save_csv = False, 
                           strict_mode = False, ignore_prev_manual_anns=True, strict_num = 2,
                           include_not_private = False)->None:
        #mycat_mode: only aggregate annotations that can be summarized in mycat (also score them in mycat in mega_table).
        #the mega table includes all privacy annotations with all corresponding info (three metrics, big five, age, gender, platform)

        # make sure this sequence is correct.
        with open('worker_privacy_num.json', encoding="utf-8") as f:
            worker_privacy_num = json.load(f)
        
        self.mega_table = pd.DataFrame(columns=["category", "informationType", "informativeness", "sharingOwner", "sharingOthers", 'age', 'gender', 'frequency', 
        'platform', 'extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness', 'imagePath', 'originalDataset', 'bbox'])
        for image_name in self.img_annotation_map.keys():
            for platform, annotations in self.img_annotation_map[image_name].items():
                # now, value[0] is the only availiable index
                for i, annotation in enumerate(annotations):
                    # stop to record exceed annotations
                    if strict_mode and i >= strict_num:
                        break
                    image_id = annotation.split('_')[0]
                    prefix_len = len(image_id) + 1
                    worker_file = annotation[prefix_len:]
                    worker_id = worker_file[:-11]
                    worker_file = worker_id + '.json'
                    privacy_num = worker_privacy_num[worker_id]
                    with open(os.path.join(self.annotation_path, platform, 'workerinfo', worker_file), encoding="utf-8") as f_worker, \
                    open(os.path.join(self.annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label, \
                    open(os.path.join('./new annotations', 'annotations', image_id + '_label.json'), encoding="utf-8") as f_oriLabel:
                        worker = json.load(f_worker)
                        label = json.load(f_label)
                        oriLabel = json.load(f_oriLabel)
                        oriLabel = oriLabel['annotations']
                        # we only analyze default annotations
                        age = worker['age']
                        gender = worker['gender']
                        extraversion = worker['bigfives']['Extraversion']
                        agreeableness = worker['bigfives']['Agreeableness']
                        conscientiousness = worker['bigfives']['Conscientiousness']
                        neuroticism = worker['bigfives']['Neuroticism']
                        openness = worker['bigfives']['Openness to Experience']
                        dataset_name = label['source']     
                        frequency = worker['frequency']
                        nationality = worker['nationality'] if worker['nationality'] == 'Japan' else 'UK'
                        for key, value in label['defaultAnnotation'].items():
                            bbox = []
                            if value['ifNoPrivacy'] and not include_not_private:
                                continue
                            category = ''
                            for ori in oriLabel.values():
                                if ori['category'] == key:
                                    ori['bbox'] = [int(x) for x in ori['bbox']]
                                    bbox.append(ori['bbox'])
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
                            if ignore_prev_manual_anns and category.startswith('Object'):
                                continue
                            id = annotation[:-11] + '_' + key
                            informationType = [0 for i in range(6)] if value['ifNoPrivacy'] else value['informationType']
                            informativeness = -1 if value['ifNoPrivacy'] else int(value['informativeness']) - 1
                            sharingOwner = [0 for i in range(7)] if value['ifNoPrivacy'] else value['sharingOwner']
                            sharingOthers = [0 for i in range(7)] if value['ifNoPrivacy'] else value['sharingOthers']
                            if informationType[5] == 1:
                                self.custom_informationType.append(value['informationTypeInput'])
                            if sharingOwner[6] == 1:
                                self.custom_recipient_owner.append(value['sharingOwnerInput'])
                            if sharingOthers[6] == 1:
                                self.custom_recipient_others.append(value['sharingOthersInput'])
                            
                            entry = pd.DataFrame.from_dict({
                                #'id': [id],
                                "category": [category],
                                "informationType":  [informationType],
                                "informativeness": [informativeness],
                                "sharingOwner": [sharingOwner],
                                "sharingOthers": [sharingOthers],
                                "age": [age],
                                "gender": [gender],
                                "platform": [platform],
                                "nationality": [nationality],
                                "extraversion": [extraversion],
                                "agreeableness": [agreeableness],
                                "conscientiousness": [conscientiousness],
                                "neuroticism": [neuroticism],
                                "openness": [openness],
                                'frequency': [frequency],
                                'imagePath': [image_name + '.jpg'],
                                #'originCategory': value['category'],
                                'originalDataset': [dataset_name],
                                #'privacyNum': [privacy_num],
                                'bbox': [bbox]
                            })

                            self.mega_table = pd.concat([self.mega_table, entry], ignore_index=True)
        if save_csv:
            self.mega_table.to_csv(self.mega_table_path, index =False)

    def prepare_manual_label(self, save_csv = False, strict_mode = True, ignore_prev_manual_anns=True, strict_num = 2) -> None:
        self.manual_table = pd.DataFrame(columns=["category", "informationType", "informativeness", "sharingOwner", "sharingOthers", 'age', 'gender', 'frequency',
        'platform', 'extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness', 'imagePath', 'originalDataset', 'bbox'])
        with open('worker_privacy_num.json', encoding="utf-8") as f:
            worker_privacy_num = json.load(f)
        for image_name in self.img_annotation_map.keys():
            for platform, annotations in self.img_annotation_map[image_name].items():
                # now, value[0] is the only availiable index
                for i, annotation in enumerate(annotations):
                    # stop to record exceed annotations
                    if strict_mode and i >= strict_num:
                        break
                    image_id = annotation.split('_')[0]
                    prefix_len = len(image_id) + 1
                    worker_file = annotation[prefix_len:]
                    worker_id = worker_file[:-11]
                    worker_file = worker_id + '.json'
                    privacy_num = worker_privacy_num[worker_id]
                with open(os.path.join(self.annotation_path, platform, 'workerinfo', worker_file), encoding="utf-8") as f_worker, \
                open(os.path.join(self.annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label:
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
                    frequency = worker['frequency']     
                    dataset_name = label['source']  
                    nationality = worker['nationality'] if worker['nationality'] == 'Japan' else 'UK'
                    for key, value in label['manualAnnotation'].items():
                        category = value['category']
                        id = annotation[:-11] + '_' + key
                        informationType = value['informationType']
                        informativeness = int(value['informativeness']) - 1
                        sharingOwner = value['sharingOwner']
                        sharingOthers = value['sharingOthers']
                        bboxes = [] 
                        bbox = value['bbox']
                        try:
                            bbox = [int(x) for x in bbox]
                            bboxes.append(bbox)
                            if informationType[5] == 1:
                                self.custom_informationType.append(value['informationTypeInput'])
                            if sharingOwner[6] == 1:
                                self.custom_recipient_owner.append(value['sharingOwnerInput'])
                            if sharingOthers[6] == 1:
                                self.custom_recipient_others.append(value['sharingOthersInput'])
                            entry = pd.DataFrame.from_dict({
                                #'id': [id],
                                "category": ['Manual Label'],
                                "informationType":  [informationType],
                                "informativeness": [informativeness],
                                "sharingOwner": [sharingOwner],
                                "sharingOthers": [sharingOthers],
                                "age": [age],
                                "gender": [gender],
                                "platform": [platform],
                                "nationality": [nationality],
                                "extraversion": [extraversion],
                                "agreeableness": [agreeableness],
                                "conscientiousness": [conscientiousness],
                                "neuroticism": [neuroticism],
                                "openness": [openness],
                                'frequency': [frequency],
                                'imagePath': [image_name + '.jpg'],
                                #'originCategory': value['category'],
                                'originalDataset': [dataset_name],
                                #'privacyNum': [privacy_num],
                                'bbox': [bboxes]
                            })

                            self.manual_table = pd.concat([self.manual_table, entry], ignore_index=True)
                        except:
                            print('wrong bbox', bbox)
        if save_csv:
            self.manual_table.to_csv('./manual_table.csv', index =False)
    def basic_count(self, read_csv = False, split_count = False, count_scale = 'CrowdWorks',
                    strict_mode = True, ignore_prev_manual_anns=False, strict_num = 2) -> None:

        def calculate_array(input_array, option_num):
            res = np.zeros(option_num, dtype='int')
            for i in range(input_array.shape[0]):
                res += np.array(json.loads(input_array[i]))
            return res
        if read_csv:
            self.mega_table = pd.read_csv(self.mega_table_path)
        else:
            self.prepare_mega_table()
        if split_count:
            print(self.mega_table)
            self.mega_table = self.mega_table[self.mega_table['platform'] == count_scale]
            print(self.mega_table)
        valid_workers = []
        with open(os.path.join(self.annotation_path, count_scale, 'valid_workers.json')) as f:
            valid_workers = json.load(f)
        print('val len:', len(valid_workers))
        frequency = self.mega_table['frequency'].value_counts()
        frequency = frequency.sort_index().values
        frequency = pd.DataFrame([frequency], columns=self.description['frequency'])
        informationType = calculate_array(self.mega_table['informationType'].values, 6)
        informationType = pd.DataFrame([informationType], columns=self.description['informationType'])
        informativeness = self.mega_table['informativeness'].value_counts()
        print(informativeness)
        informativeness = informativeness.sort_index().values
        informativeness = pd.DataFrame([informativeness], columns=self.description['informativeness'])

        #informativeness = pd.DataFrame([informativeness], columns=self.description['informativeness'])
        sharingOwner = calculate_array(self.mega_table['sharingOwner'].values, 7)
        sharingOwner = pd.DataFrame([sharingOwner], columns=self.description['sharingOwner'])
        sharingOthers = calculate_array(self.mega_table['sharingOthers'].values, 7)
        sharingOthers = pd.DataFrame([sharingOthers], columns=self.description['sharingOthers'])

        print('----------{}----------'.format('frequency'))
        print(frequency)
        print('----------{}----------'.format('informationType'))
        print(informationType)
        print('----------{}----------'.format('informativeness'))
        print(informativeness)
        print('----------{}----------'.format('sharingOwner'))
        print(sharingOwner)
        print('----------{}----------'.format('sharingOthers'))
        print(sharingOthers)

        ## privacy time in image wise
        image_privacy_time = {'Prolific': {}, 'CrowdWorks': {}, 'All': {}}
        privacy_time = {'Prolific': {0: 0, 1: 0, 2: 0}, 
                        'CrowdWorks': {0: 0, 1: 0, 2: 0}, 
                        'All': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}}
        for image_name in self.img_annotation_map.keys():
            all = 0
            for platform, annotations in self.img_annotation_map[image_name].items():
                this_platform = 0
                for i, annotation in enumerate(annotations):
                    if strict_mode and i >= strict_num:
                        break
                    if annotation not in image_privacy_time[platform].keys():
                        image_privacy_time[platform][annotation] = 0
                    if annotation not in image_privacy_time['All'].keys():
                        image_privacy_time['All'][annotation] = 0
                    with open(os.path.join(self.annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label:
                        ifPrivacy = False
                        label = json.load(f_label)
                        if len(label['manualAnnotation']) > 0:
                            ifPrivacy = True
                        for key, value in label['defaultAnnotation'].items():
                            category = value['category']
                            if ignore_prev_manual_anns and category.startswith('Object'):
                                continue
                            if not value['ifNoPrivacy']:
                                ifPrivacy = True
                        if ifPrivacy:
                            all += 1
                            this_platform += 1
                image_privacy_time[platform][annotation] = this_platform
                privacy_time[platform][this_platform] += 1

            image_privacy_time['All'][annotation] = all
            privacy_time['All'][all] += 1
        
        print('privacy', privacy_time)
        
        ## overlap with DIPA 1.0
        dipa1_table = pd.read_csv('mega_table (DIPA 1.0).csv')
        dipa1_table['id_content'] = dipa1_table.apply(lambda row: row['category'] + '_' + row['imagePath'], axis=1)
        privacy_content_dipa1 = dipa1_table['id_content'].tolist()
        self.mega_table['id_content'] = self.mega_table.apply(lambda row: row['category'] + '_' + row['imagePath'], axis=1)
        privacy_content_dipa2 =self.mega_table['id_content'].tolist()
        overlap = list(set(privacy_content_dipa2).intersection(privacy_content_dipa1))
        print('overlap with DIPA 1.0:', len(overlap))
        print('unique content in DIPA 2.0:', len(set(privacy_content_dipa2)))

        ## how many times annotated in DIPA 2.0
        print('content appear')
        exclude_manual = self.mega_table[self.mega_table.category != 'Manual Label']
        #print(exclude_manual)
        print(exclude_manual['id_content'].value_counts().value_counts())
        ## annotation per content

        self.mega_table['information choice time'] = self.mega_table.apply(lambda row: sum(json.loads(row['informationType'])), axis = 1)
        print(self.mega_table['information choice time'].mean())
        self.mega_table['owner choice time'] = self.mega_table.apply(lambda row: sum(json.loads(row['sharingOwner'])), axis = 1)
        print(self.mega_table['owner choice time'].mean())
        self.mega_table['others choice time'] = self.mega_table.apply(lambda row: sum(json.loads(row['sharingOthers'])), axis = 1)
        print(self.mega_table['others choice time'].mean())
    
        ### contigency table for information type, sharing owner, sharing others
        # information type
        # table 6*6
        information_tab = np.zeros((6, 6))
        for i, row in self.mega_table.iterrows():
            information = json.loads(row['informationType'])
            for j in range(6):
                if information[j] == 1:
                    information_tab[j] += information
        print(information_tab)

        # sharing owner
        # table 7*7
        sharingOwner_tab = np.zeros((7, 7))
        for i, row in self.mega_table.iterrows():
            sharingOwner = json.loads(row['sharingOwner'])
            for j in range(7):
                if sharingOwner[j] == 1:
                    sharingOwner_tab[j] += sharingOwner
        print(sharingOwner_tab)

        # sharing others
        # table 7*7
        sharingOthers_tab = np.zeros((7, 7))
        for i, row in self.mega_table.iterrows():
            sharingOthers = json.loads(row['sharingOthers'])
            for j in range(7):
                if sharingOthers[j] == 1:
                    sharingOthers_tab[j] += sharingOthers
        print(sharingOthers_tab)
    def count_worker_privacy_num(self) -> None:
        # as every image in image pool is somewhat privacy-threatening, we count how many privacy-threatening image have each worker choose to measure if they care about privacy.
        # input: read img_annotation_map.json
        # output: worker_privacy_num.json

        worker_privacy_num = {}
        for image_name in self.img_annotation_map.keys():
            for platform, annotations in self.img_annotation_map[image_name].items():
                # now, value[0] is the only availiable index
                for annotation in annotations:
                    image_id = annotation.split('_')[0]
                    prefix_len = len(image_id) + 1
                    worker_file = annotation[prefix_len:]
                    worker_id = worker_file[:-11]
                    if worker_id not in worker_privacy_num.keys():
                        worker_privacy_num[worker_id] = 0
                    with open(os.path.join(self.annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label:
                        ifPrivacy = False
                        label = json.load(f_label)
                        if len(label['manualAnnotation']) > 0:
                            ifPrivacy = True
                        for key, value in label['defaultAnnotation'].items():
                            if not value['ifNoPrivacy']:
                                ifPrivacy = True
                        if ifPrivacy:
                            worker_privacy_num[worker_id] += 1

        with open('worker_privacy_num.json', 'w', encoding="utf-8") as w:
            json.dump(worker_privacy_num, w)

    def prepare_regression_model_table(self, read_csv = False, strict_mode = True, strict_num = 2)->None:
        #we change running regression model to R
        #Two table: image_wise_regression_table.csv
        #           annotation_wise_regression_table.csv
        if read_csv:
            self.mega_table = pd.read_csv(self.mega_table_path)
        else:
            self.prepare_mega_table()
        with open('worker_privacy_num.json', encoding="utf-8") as f:
            worker_privacy_num = json.load(f)
        # image_wise_regression_table.csv
        # image_wise_regression_table = pd.DataFrame(columns=['age', "gender", "platform", 'privacyNum', 'frequency', 'ifPrivacy'])
        # for image_name in self.img_annotation_map.keys():
        #     for platform, annotations in self.img_annotation_map[image_name].items():
        #         for annotation in annotations:
        #             image_id = annotation.split('_')[0]
        #             prefix_len = len(image_id) + 1
        #             worker_file = annotation[prefix_len:]
        #             worker_id = worker_file[:-11]
        #             worker_file = worker_id + '.json'
        #             privacy_num = worker_privacy_num[worker_id]
                    
        #             with open(os.path.join(self.annotation_path, platform, 'workerinfo', worker_file), encoding="utf-8") as f_worker, \
        #             open(os.path.join(self.annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label:
        #                 worker = json.load(f_worker)
        #                 label = json.load(f_label)
        #                 # we only analyze default annotations
        #                 ifPrivacy = False
        #                 year = int(worker['age'])
        #                 # if 18 <= year <= 24:
        #                 #     age = 1
        #                 # elif 25 <= year <= 34:
        #                 #     age = 2
        #                 # elif 35 <= year <= 44:
        #                 #     age = 3
        #                 # elif 45 <= year <= 54:
        #                 #     age = 4
        #                 # elif 55 <= year:
        #                 #     age = 5
        #                 gender = worker['gender']
        #                 frequency = worker['frequency']
        #                 nationality = 'Japan' if platform == 'CrowdWorks' else 'British'
        #                 if len(label['manualAnnotation']) > 0:
        #                     ifPrivacy = True
        #                 for key, value in label['defaultAnnotation'].items():
        #                     if not value['ifNoPrivacy']:
        #                         ifPrivacy = True
        #                 entry = pd.DataFrame.from_dict({
        #                         "age": [year],
        #                         "gender": [gender],
        #                         "nationality": [int(nationality)],
        #                         'privacyNum': [privacy_num],
        #                         'ifPrivacy': [1 if ifPrivacy else 0],
        #                         'frequency': [frequency]
        #                     })

        #                 image_wise_regression_table = pd.concat([image_wise_regression_table, entry], ignore_index=True)

        # image_wise_regression_table.to_csv('./image_wise_regression_table.csv', index =False)        
        
        # annotation_wise_regression_table.csv
        annotation_wise_regression_table = pd.DataFrame(columns=['age', "gender", "nationality",
                                                                 'extraversion', 'agreeableness', 'conscientiousness', 
                                                                  'neuroticism', 'openness',
                                                                  'ifPrivacy', 'frequency'])
        for image_name in self.img_annotation_map.keys():
            for platform, annotations in self.img_annotation_map[image_name].items():
                for i, annotation in enumerate(annotations):
                    if strict_mode and i >= strict_num:
                        break
                    image_id = annotation.split('_')[0]
                    prefix_len = len(image_id) + 1
                    worker_file = annotation[prefix_len:]
                    worker_id = worker_file[:-11]
                    worker_file = worker_id + '.json'
                    privacy_num = worker_privacy_num[worker_id]
                    
                    with open(os.path.join(self.annotation_path, platform, 'workerinfo', worker_file), encoding="utf-8") as f_worker, \
                    open(os.path.join(self.annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label:
                        worker = json.load(f_worker)
                        label = json.load(f_label)
                        # we only analyze default annotations
                        ifPrivacy = False
                        year = int(worker['age'])
                        gender = worker['gender']
                        frequency = worker['frequency']
                        nationality = 'Japan' if platform == 'CrowdWorks' else 'British'
                        extraversion = worker['bigfives']['Extraversion']
                        agreeableness = worker['bigfives']['Agreeableness']
                        conscientiousness = worker['bigfives']['Conscientiousness']
                        neuroticism = worker['bigfives']['Neuroticism']
                        openness = worker['bigfives']['Openness to Experience']
                        for key, value in label['defaultAnnotation'].items():                       
                            entry = pd.DataFrame.from_dict({
                                    "age": [year],
                                    "gender": [gender],
                                    "nationality": [nationality],
                                    "extraversion": [extraversion],
                                    "agreeableness": [agreeableness],
                                    "conscientiousness": [conscientiousness],
                                    "neuroticism": [neuroticism],
                                    "openness": [openness],
                                    "frequency": [frequency],
                                    'ifPrivacy': [0 if value['ifNoPrivacy'] else 1],
                                })

                            annotation_wise_regression_table = pd.concat([annotation_wise_regression_table, entry], ignore_index=True)
                        for key, value in label['manualAnnotation'].items():                
                            entry = pd.DataFrame.from_dict({
                                    "age": [year],
                                    "gender": [gender],
                                    "nationality": [nationality],
                                    "extraversion": [extraversion],
                                    "agreeableness": [agreeableness],
                                    "conscientiousness": [conscientiousness],
                                    "neuroticism": [neuroticism],
                                    "openness": [openness],
                                    "frequency": [frequency],
                                    'ifPrivacy': [1],
                                })

                            annotation_wise_regression_table = pd.concat([annotation_wise_regression_table, entry], ignore_index=True)

        annotation_wise_regression_table.to_csv('./annotation_wise_regression_table.csv', index =False)        
    def regression_model(self, input_channel, output_channel, read_csv = False)->None:
        if read_csv:
            self.mega_table = pd.read_csv(self.mega_table_path)
        else:
            self.prepare_mega_table()
        output_dims = []
        # the output needs to be one-hot
        

        for output in output_channel:
            output_dims.append(len(self.mega_table[output].unique()))
        scaler = StandardScaler()
        encoder = LabelEncoder()
        self.mega_table['category'] = encoder.fit_transform(self.mega_table['category'])
        self.mega_table['gender'] = encoder.fit_transform(self.mega_table['gender'])
        self.mega_table['platform'] = encoder.fit_transform(self.mega_table['platform'])
        self.mega_table['id'] = encoder.fit_transform(self.mega_table['id'])
        self.mega_table['datasetName'] = encoder.fit_transform(self.mega_table['datasetName'])

        X = self.mega_table[input_channel].values
        y = []
        for idx in range(len(self.mega_table)):
            information = self.mega_table['informationType'].iloc[idx]
            information = np.array(json.loads(information))

            informativeness_num = self.mega_table['informativeness'].iloc[idx]
            informativeness = np.zeros(7)
            informativeness[informativeness_num] = 1.

            sharingOwner = self.mega_table['sharingOwner'].iloc[idx]
            sharingOwner = np.array(json.loads(sharingOwner))

            sharingOthers = self.mega_table['sharingOthers'].iloc[idx]
            sharingOthers = np.array(json.loads(sharingOthers))

            label = np.concatenate((information, informativeness, sharingOwner, sharingOthers))

            y.append(label)

        model = sm.OLS(y, X)
        results = model.fit()

        # print the intercept, coefficient, and p-value of each variable
        print('Intercept:', results.params[0])
        print('Coefficients:', results.params[1:])
        print('P-values:', results.rsquared)


    def svm(self, input_channel, output_channel, read_csv = False) -> None:
        if read_csv:
            self.mega_table = pd.read_csv(self.mega_table_path)
        else:
            self.prepare_mega_table()
        output_dims = []
        # the output needs to be one-hot
        for output in output_channel:
            output_dims.append(len(self.mega_table[output].unique()))
        scaler = StandardScaler()
        encoder = LabelEncoder()
        self.mega_table['category'] = encoder.fit_transform(self.mega_table['category'])
        self.mega_table['gender'] = encoder.fit_transform(self.mega_table['gender'])
        self.mega_table['platform'] = encoder.fit_transform(self.mega_table['platform'])
        print(self.mega_table[input_channel])
        X = self.mega_table[input_channel].values
        y = self.mega_table[output_channel].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=0)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svc = svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.8)
        classifier = MultiOutputClassifier(svc, n_jobs=-1)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)

        # Print evaluation metrics
        acc = np.zeros(len(output_channel))
        pre = np.zeros(len(output_channel))
        rec = np.zeros(len(output_channel))
        f1 = np.zeros(len(output_channel))
        conf = []
        for i, output_dim in enumerate(output_dims):
            conf.append(np.zeros((output_dim,output_dim)))
        for j, output in enumerate(output_channel):
            acc[j] = metrics.accuracy_score(y_test[:, j], y_pred[:, j])
            pre[j] = metrics.precision_score(y_test[:, j], y_pred[:, j], average='weighted')
            rec[j] = metrics.recall_score(y_test[:, j], y_pred[:, j], average='weighted')
            f1[j] = metrics.f1_score(y_test[:, j], y_pred[:, j], average='weighted')
            conf[j] += metrics.confusion_matrix(y_test[:, j], y_pred[:, j], labels = self.mega_table[output].unique())

        pandas_data = {'Accuracy' : acc, 'Precision' : pre, 'Recall': rec, 'f1': f1}
        
        for i, output in enumerate(output_channel):
            print('confusion matrix for {}'.format(output))
            print(np.round(conf[i], 3))
        df = pd.DataFrame(pandas_data, index=output_channel)
        print(df.round(3))

    def knn(self,input_channel, output_channel, read_csv = False) -> None:
        if read_csv:
            self.mega_table = pd.read_csv(self.mega_table_path)
        else:
            self.prepare_mega_table()
        
        output_dims = []
        # the output needs to be one-hot
        for output in output_channel:
            output_dims.append(len(self.mega_table[output].unique()))
        scaler = StandardScaler()
        encoder = LabelEncoder()
        self.mega_table['category'] = encoder.fit_transform(self.mega_table['category'])
        self.mega_table['gender'] = encoder.fit_transform(self.mega_table['gender'])
        self.mega_table['platform'] = encoder.fit_transform(self.mega_table['platform'])
        X = self.mega_table[input_channel].values
        y = self.mega_table[output_channel].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=0)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=5)
        #classifier.fit(X_train, np.ravel(y_train,order="c"))
        classifier = MultiOutputClassifier(knn, n_jobs=-1)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)

        # Print evaluation metrics
        acc = np.zeros(len(output_channel))
        pre = np.zeros(len(output_channel))
        rec = np.zeros(len(output_channel))
        f1 = np.zeros(len(output_channel))
        conf = []

        for i, output_dim in enumerate(output_dims):
            conf.append(np.zeros((output_dim,output_dim)))
        for j, output in enumerate(output_channel):
            acc[j] = metrics.accuracy_score(y_test[:, j], y_pred[:, j])
            pre[j] = metrics.precision_score(y_test[:, j], y_pred[:, j], average='weighted')
            rec[j] = metrics.recall_score(y_test[:, j], y_pred[:, j], average='weighted')
            f1[j] = metrics.f1_score(y_test[:, j], y_pred[:, j], average='weighted')
            conf[j] += metrics.confusion_matrix(y_test[:, j], y_pred[:, j], labels = self.mega_table[output].unique())
            
        pandas_data = {'Accuracy' : acc, 'Precision' : pre, 'Recall': rec, 'f1': f1}
        for i, output in enumerate(output_channel):
            print('confusion matrix for {}'.format(output))
            print(np.round(conf[i], 3))
        df = pd.DataFrame(pandas_data, index=output_channel)
        print(df.round(3))

    def anova(self,read_csv = False) -> None:
        ## the degree of freedom of "informativeness" is wrong, it should be 6 rather than 1
        ## I am using R to perform this
        if read_csv:
            self.mega_table = pd.read_csv(self.mega_table_path)
        else:
            self.prepare_mega_table()
        #agg_data = self.mega_table.groupby(['informationType', 'informativeness'])['sharing'].mean()
        encoder = LabelEncoder()
        self.mega_table['category'] = encoder.fit_transform(self.mega_table['category'])
        self.mega_table['gender'] = encoder.fit_transform(self.mega_table['gender'])
        self.mega_table['platform'] = encoder.fit_transform(self.mega_table['platform'])
        self.mega_table['informationType'] = encoder.fit_transform(self.mega_table['informationType'])
        self.mega_table['informativeness'] = encoder.fit_transform(self.mega_table['informativeness'])
        self.mega_table['id'] = encoder.fit_transform(self.mega_table['id'])
        # get dataset
        #aov = AnovaRM(self.mega_table, depvar='sharing', subject= 'id', within=['informationType', 'informativeness'], aggregate_func='mean')
        #res = aov.fit()
        #print(res)
        # Print the results
        model = ols('sharing ~ informationType*informativeness', data=self.mega_table).fit()
        aov_table = sm.stats.anova_lm(model, typ=1)
        print(aov_table)

    def neural_network(self, input_channel, output_channel, read_csv = False) -> None:
        def l1_distance_loss(prediction, target):
            loss = np.abs(prediction - target)
            return np.mean(loss)
        def l2_distance_loss(prediction, target):
            target = target.float()
            loss = (prediction - target) ** 2
            return loss.mean()
        def train_one_epoch():
            #running_loss = 0.
            last_loss = 0.
            for i, data in enumerate(training_loader):
                # Every data instance is an input + label pair
                inputs, labels = data
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                
                # Make predictions for this batch
                outputs = model(inputs)
                #labels = labels.squeeze()
                # Compute the loss and its gradients
                losses = []
                for j, output in enumerate(output_channel):
                    losses.append(loss_fns[j](outputs[j], labels[:, j]))
                tot_loss = 0
                for loss in losses:
                    tot_loss += loss
                tot_loss.backward()

                # Adjust learning weights
                optimizer.step()
                last_loss += tot_loss.item()
                # Gather data and report
                '''running_loss += loss.item()
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.'''
            last_loss = last_loss / (i + 1)
            return last_loss

        
        learning_rate = 0.01
        if read_csv:
            self.mega_table = pd.read_csv(self.mega_table_path)
        else:
            self.prepare_mega_table()


        input_dim  = len(input_channel)
        output_dims = []
        # the output needs to be one-hot
        for output in output_channel:
            output_dims.append(len(self.mega_table[output].unique()))

        scaler = StandardScaler()
        encoder = LabelEncoder()
        self.mega_table['category'] = encoder.fit_transform(self.mega_table['category'])
        self.mega_table['gender'] = encoder.fit_transform(self.mega_table['gender'])
        self.mega_table['platform'] = encoder.fit_transform(self.mega_table['platform'])
        self.mega_table['id'] = encoder.fit_transform(self.mega_table['id'])
        # get dataset
        X = self.mega_table[input_channel].values
        y = self.mega_table[output_channel].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=0)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) 
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        model = nn_model(input_dim,output_dims)
        loss_fns = []
        for output in output_channel:
            if output == 'informativeness':
                loss_fns.append(nn.CrossEntropyLoss())
                #loss_fns.append(l1_distance_loss)
            else:
                loss_fns.append(nn.CrossEntropyLoss())
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        training_dataset = nn_dataset(X_train, y_train)
        testing_dataset = nn_dataset(X_test, y_test)
        training_loader = DataLoader(training_dataset, batch_size=64, shuffle=True)
        testing_loader = DataLoader(testing_dataset, batch_size=64, shuffle=True)

        #start training
        writer = SummaryWriter()
        epoch_number = 0
        EPOCHS = 200

        #best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = train_one_epoch()
            
            # We don't need gradients on to do reporting
            model.train(False)
            acc = np.zeros(len(output_channel))
            pre = np.zeros(len(output_channel))
            rec = np.zeros(len(output_channel))
            f1 = np.zeros(len(output_channel))
            distance = 0.0
            conf = []
            for i, output_dim in enumerate(output_dims):
                conf.append(np.zeros((output_dim,output_dim)))
            running_vloss = 0.0
            for i, vdata in enumerate(testing_loader):
                vloss = 0.0
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                #vlabels = vlabels.squeeze()
                losses = []
                for j, output in enumerate(output_channel):
                    losses.append(loss_fns[j](voutputs[j], vlabels[:, j]))
                    y_pred, max_indices = torch.max(voutputs[j], dim = 1)
                    acc[j] += metrics.accuracy_score(vlabels[:, j].detach().numpy(), max_indices.detach().numpy())
                    pre[j] += metrics.precision_score(vlabels[:, j].detach().numpy(), max_indices.detach().numpy(),average='weighted')
                    rec[j] += metrics.recall_score(vlabels[:, j].detach().numpy(), max_indices.detach().numpy(),average='weighted')
                    f1[j] += metrics.f1_score(vlabels[:, j].detach().numpy(), max_indices.detach().numpy(),average='weighted')
                    conf[j] += metrics.confusion_matrix(vlabels[:, j].detach().numpy(), max_indices.detach().numpy(), labels = self.mega_table[output].unique())
                    if output == 'informativeness':
                        distance += l1_distance_loss(vlabels[:, j].detach().numpy(), max_indices.detach().numpy())
                tot_vloss = 0
                for loss in losses:
                    tot_vloss += loss
                
                running_vloss += tot_vloss
            acc = acc / (i + 1)
            pre = pre / (i + 1)
            rec = rec / (i + 1)
            f1 = f1 / (i + 1)
            distance = distance / (i + 1)
            #print("Accuracy:",acc)
            #print("Precision:",pre)
            #print("Recall:",rec)
            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            if epoch == EPOCHS - 1:
                for i, output in enumerate(output_channel):
                    conf[i] = conf[i].astype('float') / conf[i].sum(axis=1)[:, np.newaxis]
                    plt.imshow(conf[i], cmap=plt.cm.Blues)
                    plt.xticks(np.arange(0, len(self.description[output])), self.description[output], rotation = 45, ha='right')
                    plt.yticks(np.arange(0, len(self.description[output])), self.description[output])
                    plt.xlabel("Predicted Label")
                    plt.ylabel("True Label")
                    plt.title('confusion matrix for {}'.format(output))
                    plt.colorbar()
                    plt.tight_layout()

                    plt.savefig('confusion matrix for {}.png'.format(output), dpi=1200)
                    plt.clf()
                    print('confusion matrix for {}'.format(output))
                    print(np.round(conf[i], 3))
            
            pandas_data = {'Accuracy' : acc, 'Precision' : pre, 'Recall': rec, 'f1': f1}
            df = pd.DataFrame(pandas_data, index=output_channel)
            print(df.round(3))
            if 'informativeness' in output_channel:
                print('informativenss distance: ', distance)
            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            for i, output in enumerate(output_channel):
                writer.add_scalars('{} Metrics, Accuracy Precision Recall'.format(output),
                                {'Accuracy' : acc[i], 'Precision' : pre[i], 'Recall': rec[i] },
                                epoch_number + 1)
            #writer.flush()

            # Track best performance, and save the model's state
            '''if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)'''

            epoch_number += 1
        writer.close()
if __name__ == '__main__':
    analyze = analyzer()
    bigfives = ["extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness"]
    basic_info = [ "age", "gender", "platform", 'frequency', 'privacyNum']
    category = ['category']
    privacy_metrics = ['informationType', 'informativeness', 'sharingOwner', 'sharingOthers']

    input_channel = []
    output_channel = []

    input_channel.extend(bigfives)
    input_channel.extend(basic_info)
    input_channel.extend(category)
    output_channel = privacy_metrics

    #analyze.distribution(read_csv=True)
    #analyze.basic_info()
    #analyze.generate_img_annotation_map()
    #analyze.count_worker_privacy_num()
    analyze.prepare_mega_table(mycat_mode = False, save_csv=True, strict_mode=True, ignore_prev_manual_anns=False, include_not_private=False)
    analyze.prepare_manual_label(save_csv=True, strict_mode=True)
    #analyze.basic_count(read_csv = True, ignore_prev_manual_anns=False,split_count=False,count_scale='Prolific')
    #analyze.prepare_regression_model_table(read_csv=True)
    #analyze.regression_model(input_channel=input_channel, output_channel=output_channel, read_csv=True)
    