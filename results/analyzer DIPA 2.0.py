import os
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 

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
from scipy.stats import zscore

import random
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
        # some categories in others can still be mapped to mycat due to the mistakes when summarizing before
        self.others_map = {}
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
        self.manual_table_path = './manual_table.csv'
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
        with open('./others_convertor.csv') as f:
            res = csv.reader(f)
            flag = 0
            for row in res:
                if not flag:
                    flag = 1
                    continue
                self.others_map[row[0]] = row[1]

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
            dataset_name = row['originalDataset']
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

        #record all images that hit four times as privacy from privacy_category
        four_time_list = []
        for platform, category in privacy_category.items():
            for key, value in category.items():
                for object_id, num in value.items():
                    if num == 4:
                        img_id = object_id.split('_')[0] + '.jpg'
                        four_time_list.append(img_id)
        
        #unique and print len
        four_time_list = list(set(four_time_list))
        print('four time list len:', len(four_time_list))
        #generate task record for DMBIS pilot study
        task_record = {}
        task_per_worker = 20
        task_num = -1
        for i, imageId in enumerate(four_time_list):
            if i % task_per_worker == 0:
                task_num += 1
                task_record[task_num] = []
            task_record[task_num].append(imageId)
        with open('task_record_for_dmbis_pilot.json', 'w') as f:
            json.dump(task_record, f)

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
                    # read image and get image size
                    image = Image.open(os.path.join('images', image_name + '.jpg'))
                    width, height = image.size
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
                            mycat = ''
                            for ori in oriLabel.values():
                                if ori['category'] == key:
                                    ori['bbox'] = [int(x) for x in ori['bbox']]
                                    bbox.append(ori['bbox'])

                            if dataset_name == 'OpenImages':
                                if key in self.openimages_mycat_map.keys():
                                    mycat = self.openimages_mycat_map[key]
                            elif dataset_name == 'LVIS':
                                if key in self.lvis_mycat_map.keys():
                                    mycat = self.lvis_mycat_map[key]
                            if mycat == '':
                                mycat = 'Others'

                            if mycat_mode:
                                category = mycat
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
                                'DIPACategory': [mycat],
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
                                'bbox': [bbox],
                                'width': [int(width)],
                                'height': [int(height)]
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
                    # read image and get image size
                    image = Image.open(os.path.join('images', image_name + '.jpg'))
                    width, height = image.size
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
                                "DIPACategory": ['Others'],
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
                                'bbox': [bboxes],
                                'width': [int(width)],
                                'height': [int(height)]
                            })

                            self.manual_table = pd.concat([self.manual_table, entry], ignore_index=True)
                        except:
                            print('wrong bbox', bbox)
        if save_csv:
            self.manual_table.to_csv('./manual_table.csv', index =False)
    def count_frequency(self)->None:
        ## count frenquency of sharing for each annotator
        frequency = {'CrowdWorks': {0: 0, 1:0, 2:0, 3:0, 4:0}, 'Prolific': {0: 0, 1:0, 2:0, 3:0, 4:0}, 'All': {0: 0, 1:0, 2:0, 3:0, 4:0}}
        # CrowdWorks
        worker_file = os.listdir('./annotations/CrowdWorks/workerinfo')
        with open(os.path.join(self.annotation_path, 'CrowdWorks', 'valid_workers.json')) as f:
            valid_workers = json.load(f)
        for file in worker_file:
            if file[:-5] not in valid_workers:
                continue
            with open('./annotations/CrowdWorks/workerinfo/' + file) as f:
                worker = json.load(f)
                frequency['CrowdWorks'][int(worker['frequency'])] += 1
                frequency['All'][int(worker['frequency'])] += 1
        # Prolific
        worker_file = os.listdir('./annotations/Prolific/workerinfo')
        with open(os.path.join(self.annotation_path, 'Prolific', 'valid_workers.json')) as f:
            valid_workers = json.load(f)
        for file in worker_file:
            if file[:-5] not in valid_workers:
                continue
            with open('./annotations/Prolific/workerinfo/' + file) as f:
                worker = json.load(f)
                frequency['Prolific'][int(worker['frequency'])] += 1
                frequency['All'][int(worker['frequency'])] += 1
        
        print(frequency)

    def non_max_suppression(self, boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding boxes by their area
        area = boxes[:, 2] * boxes[:, 3]
        idxs = np.argsort(area)[::-1]

        # loop over the indexes of the bounding boxes
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box intersection
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the bounding box and the rest of the bounding boxes
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have an overlap greater than the overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked
        return boxes[pick]

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

        
        # #count the not overlap content of DIPA1 according to their category
        # dipa1_table_unoverlap = dipa1_table[~dipa1_table['id_content'].isin(overlap)]
        # dipa1_table_unoverlap = dipa1_table_unoverlap[dipa1_table_unoverlap['DIPACategory'] != 'Manual Label']
        
        # dipa1_table_overlap = dipa1_table[dipa1_table['id_content'].isin(overlap)]
        # dipa1_table_overlap = dipa1_table_overlap[dipa1_table_overlap['DIPACategory'] != 'Manual Label']
        # # dipa1_table_unoverlap['category'] = dipa1_table_unoverlap['DIPACategory'].replace({
        # #     'Paper&Document&Label': 'Printed materials',
        # #     'Vehicle Plate': 'Vehicle plate'
        # # })

        # # dipa1_table_overlap['category'] = dipa1_table_overlap['DIPACategory'].replace({
        # #     'Paper&Document&Label': 'Printed materials',
        # #     'Vehicle Plate': 'Vehicle plate'
        # # })
        # dipa1_table_unoverlap['DIPACategory'] = dipa1_table_unoverlap['DIPACategory'].str.lower()
        # dipa1_table_overlap['DIPACategory'] = dipa1_table_overlap['DIPACategory'].str.lower()
        # # we generate a histogram of the categories of the not overlap content and overlap content
        # # two sets of data
        # # 1. not overlap content
        # # 2. overlap content
        # category_count_unoverlap = dipa1_table_unoverlap['DIPACategory'].value_counts()

        # # Get the categories distribution for overlap data
        # category_count_overlap = dipa1_table_overlap['DIPACategory'].value_counts()
        # # order the two sets by number of content, the category name should also be ordered according to it
        

        
        # print(category_count_unoverlap)
        # print(category_count_overlap)
        # # Create a DataFrame for plotting
        # plot_df = pd.DataFrame({
        #     'Unoverlap': category_count_unoverlap,
        #     'Overlap': category_count_overlap
        # })
        # plot_df['total'] = plot_df['Unoverlap'] + plot_df['Overlap']
        # plot_df = plot_df.sort_values('total', ascending=False).drop(columns='total')
        # # Plotting the DataFrame as a bar plot
        # plot_df.plot(kind='bar')
        # # change the name of Paper&Document&Label to Printed materials
        # # Vehicle Plate to Vehicle plate
        

        # plt.title('Category Distribution for Overlap vs Unoverlap Data')
        # plt.ylabel('Count')
        # plt.xlabel('Category')
        # plt.yscale('log')
        # plt.xticks(rotation=90, fontsize=8)
        # plt.tight_layout()
        # plt.show()




        # count annotation per annotators, and the visualization of the distribution for DIPA 2.0
        
        annotator = {}
        for image_name in self.img_annotation_map.keys():
            for platform, annotations in self.img_annotation_map[image_name].items():
                for annotation in annotations:
                    image_id = annotation.split('_')[0]
                    prefix_len = len(image_id) + 1
                    worker_file = annotation[prefix_len:]
                    worker_id = worker_file[:-11]
                    # we + 1 for each valid annotation by the worker, we need to read annotation file to check if it is valid for each annotation
                    if worker_id not in annotator.keys():
                        annotator[worker_id] = 0
                    with open(os.path.join(self.annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label:
                        
                        label = json.load(f_label)
                        # for each valid label, we + 1
                        for key, value in label['defaultAnnotation'].items():
                            if not value['ifNoPrivacy']:
                                annotator[worker_id] += 1
                        for key, value in label['manualAnnotation'].items():
                            annotator[worker_id] += 1
        # calulate the distribution of annotation per annotator, according to 0--5 & 6--10 & 10-15 & 15-20 & 20--30 & 30--

        annotator_count = {'0--5': 0, '6--10': 0, '10--15': 0, '15--20': 0, '20--30': 0, '30--': 0}
        for key, value in annotator.items():
            if value <= 5:
                annotator_count['0--5'] += 1
            elif value <= 10:
                annotator_count['6--10'] += 1
            elif value <= 15:
                annotator_count['10--15'] += 1
            elif value <= 20:
                annotator_count['15--20'] += 1
            elif value <= 30:
                annotator_count['20--30'] += 1
            else:
                annotator_count['30--'] += 1
        print('annotator count')
        print(annotator_count)
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
        # if a larger index == 1 while smaller index == 0, then it is unusual, except the case of index 0 and 6
        unusual = 0
        sharingOwner_tab = np.zeros((7, 7))
        for i, row in self.mega_table.iterrows():
            sharingOwner = json.loads(row['sharingOwner'])
            for j in range(7):
                if sharingOwner[j] == 1:
                    sharingOwner_tab[j] += sharingOwner
            # find unusual
            ifunusual = False
            for j in range(1, 5):
                for k in range(j + 1, 6):
                    if sharingOwner[j] == 0 and sharingOwner[k] == 1:
                        unusual += 1
                        ifunusual = True
                        break
                if ifunusual:
                    break
        print('unusual:', unusual)
        print(sharingOwner_tab)

        # sharing others
        # table 7*7
        unusual = 0
        sharingOthers_tab = np.zeros((7, 7))
        for i, row in self.mega_table.iterrows():
            sharingOthers = json.loads(row['sharingOthers'])
            for j in range(7):
                if sharingOthers[j] == 1:
                    sharingOthers_tab[j] += sharingOthers
            ifunusual = False
            for j in range(1, 5):
                for k in range(j + 1, 6):
                    if sharingOthers[j] == 0 and sharingOthers[k] == 1:
                        unusual += 1
                        ifunusual = True
                        break
                if ifunusual:
                    break
        print('unusual:', unusual)
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

    def bbox_iou(self,boxA, boxB):
    # boxA and boxB are expected to be lists or tuples of four numbers representing the (x, y, w, h) coordinates of the boxes

    # calculate the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        # calculate the area of intersection rectangle
        intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # calculate the area of both bounding boxes
        boxA_area = boxA[2] * boxA[3]
        boxB_area = boxB[2] * boxB[3]

        # calculate the union area
        union_area = boxA_area + boxB_area - intersection_area

        # calculate IoU
        iou = intersection_area / union_area

        return iou
    
    def count_overlap_in_manual_table(self, split_count = False, count_scale = 'CrowdWorks')->None:
        # read manual table
        self.manual_table = pd.read_csv(self.manual_table_path)
        #print len
        print('manual table len:', len(self.manual_table))
        # divide it by CrowdWorks and Prolific and out put length
        print('CrowdWorks len:', len(self.manual_table[self.manual_table['platform'] == 'CrowdWorks']))
        print('Prolific len:', len(self.manual_table[self.manual_table['platform'] == 'Prolific']))
        #print unique image path len
        if split_count:
            self.manual_table = self.manual_table[self.manual_table['platform'] == count_scale]
        print('unique image path len:', len(self.manual_table['imagePath'].unique()))
        #imagePath bounding box map
        imagePath = {}
        for i, row in self.manual_table.iterrows():
            if row['imagePath'] not in imagePath.keys():
                imagePath[row['imagePath']] = []
            bbox = json.loads(row['bbox'])
            bbox = bbox[0]
            imagePath[row['imagePath']].append(bbox)
        
        # count overlap in bounding box over a threshold
        threshold = 0.7
        overlap = 0
       
        #Non-Maximum Suppression if overlap > threshold
        filtered_imagePath = {}
        for key, value in imagePath.items():
            value = np.array(value)
            value = self.non_max_suppression(value, threshold)
            filtered_imagePath[key] = value

        #count number of filtered bounding box
        for key, value in filtered_imagePath.items():
            overlap += len(value)
        print('overlap:', overlap)

        #count specific overlap by filtered bounding box
        overlap = {}
        for key, value in filtered_imagePath.items():
            for bbox in value:
                overlap_num = 0
                for ori_bbox in imagePath[key]:
                    if self.bbox_iou(bbox, ori_bbox) > threshold:
                        overlap_num += 1
                if overlap_num not in overlap.keys():
                    overlap[overlap_num] = 1
                else:
                    overlap[overlap_num] += 1
        print('overlap:', overlap)  
                        
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
    def visualDistribution(self, read_csv=False, filter_data = True, filter_corridor = (25,75), visualization=False, outputSelection=False, only_manual = False, compare = False):
        #target: get the distribution of all visual content by different metrics, like size, foreground or background, and relative location to the center.
        #Each distribution has three or two categories, like small, medium, large, foreground, background, close, netural, far, etc.
        # read mega table 
        if read_csv:
            self.mega_table = pd.read_csv(self.mega_table_path)
        else:
            self.prepare_mega_table()
        
        # for each item, get the image path, and get the image
        # get the bounding box from mega table, and get the size of the bounding box and size of the image through PIL.image
        # get the relative location of the bounding box to the center of the image
        # read every row of the meage table
        relative_sizes = []
        relative_position = []
        width_height_ratio = []
        bbox_name=[]
        bboxes = []
        informativeness = []
        information_type = []
        category = []
        others = {}
        # check if mega_table has category named manual label 
        print(self.mega_table)
        if only_manual:
            self.mega_table = self.mega_table[self.mega_table['category'] == 'Manual Label']
        else:
            # remove manual label and category with prefix Object
            self.mega_table = self.mega_table[self.mega_table['category'] != 'Manual Label']
            self.mega_table = self.mega_table[self.mega_table['category'].str.startswith('Object') == False]
        for index, row in self.mega_table.iterrows():
            #bboxes = json.loads(row['bbox'])
            image_path = os.path.join('images', row['imagePath'])
            image = Image.open(image_path)
            image_width, image_height = image.size
            if read_csv:
                row['bbox'] = json.loads(row['bbox'])
            for bbox in row['bbox']:
                #size
                if bbox[3] == 0:
                    continue
                # if category is Manual Label, we skip it
                if row['category'] == 'Manual Label':
                    continue
                size = bbox[2] * bbox[3]
                relative_size = size / (image_width * image_height)
                
                #relative location
                center_x = image_width / 2
                center_y = image_height / 2
                relative_x = (bbox[0] + bbox[2] / 2 - center_x) / image_width
                relative_y = (bbox[1] + bbox[3] / 2 - center_y) / image_height
                #print(relative_x, relative_y)
                # the relative_x and relative_y are in the range of [-0.5, 0.5]
                if relative_x > 0.5 or relative_x < -0.5 or relative_y > 0.5 or relative_y < -0.5:
                    print('relative_x or relative_y out of range')
                    continue
                #width heigh ratio

                relative_position.append([relative_x, relative_y])
                relative_sizes.append(relative_size)
                width_height_ratio.append(bbox[2] / bbox[3])
                bbox_name.append(row['imagePath'] + '_' + row['category'])
                informativeness.append(row['informativeness'])
                information_type.append(json.loads(row['informationType'])[:-1])
                bboxes.append(bbox)
                category_name = row['DIPACategory']
                if category_name == 'Others':
                    if row['category'] in self.others_map.keys():
                        category_name = self.others_map[row['category']]
                category.append(category_name)
                # if the category is Others, recording it in others
                
        
        #visualize the distribution in coordinate system
        relative_sizes = np.array(relative_sizes)
        relative_position = np.array(relative_position)
        width_height_ratio = np.array(width_height_ratio)
        informativeness = np.array(informativeness)
        information_type = np.array(information_type)
        bboxes = np.array(bboxes)
        bbox_name = np.array(bbox_name)

        print('len of relative_sizes:', len(relative_sizes))
        print('len of relative_position:', len(relative_position))
        print('len of width_height_ratio:', len(width_height_ratio))
        if filter_data:
            print('filtering data')
            # we first filter all data the size less than 0.01
            valid_data_indices = np.where(relative_sizes > 0.005)[0]
            # Filter data
            relative_sizes = relative_sizes[valid_data_indices]
            relative_position = relative_position[valid_data_indices]
            width_height_ratio = width_height_ratio[valid_data_indices]
            bbox_name = [bbox_name[i] for i in valid_data_indices]
            informativeness = [informativeness[i] for i in valid_data_indices]
            information_type = [information_type[i] for i in valid_data_indices]
            bboxes = [bboxes[i] for i in valid_data_indices]
            category = [category[i] for i in valid_data_indices]
            print('Number of valid data points:', len(valid_data_indices))

            Q1 = np.percentile(relative_sizes, filter_corridor[0])
            Q3 = np.percentile(relative_sizes, filter_corridor[1])
            IQR = Q3 - Q1

            # Identify indices where relative_sizes fall within Q1 - 1.5 * IQR and Q3 + 1.5 * IQR
            valid_data_indices = np.where((Q1<= relative_sizes) & (relative_sizes <= Q3))[0]
            #show how many data are removed
            print('Number of valid data points:', len(valid_data_indices))
            print('Number of invalid data points:', len(relative_sizes) - len(valid_data_indices))
            # Filter data
            relative_sizes = relative_sizes[valid_data_indices]
            relative_position = relative_position[valid_data_indices]
            width_height_ratio = width_height_ratio[valid_data_indices]
            bbox_name = [bbox_name[i] for i in valid_data_indices]
            informativeness = [informativeness[i] for i in valid_data_indices]
            information_type = [information_type[i] for i in valid_data_indices]
            bboxes = [bboxes[i] for i in valid_data_indices]
            category = [category[i] for i in valid_data_indices]
            print('Number of valid data points:', len(valid_data_indices))

        # print the distribution of category, according to each category
        category_count = {}
        for cat in category:
            # we first convert the category to DIPA's category
            
            if cat in category_count:
                category_count[cat] += 1
            else:
                category_count[cat] = 1
           
        for cat, count in category_count.items():
            print(f"{cat}: {count}")
        
        for i, b_name in enumerate(bbox_name):
            cat = category[i]
            if cat == 'Others':
                b_cat = b_name.split('_')[1]
                if b_cat not in others.keys():
                    others[b_cat] = 1
                else:
                    others[b_cat] += 1
        print('others:', others)
        print('others key', others.keys())

        
        # divide data into 30, 40 ,30
        low_number = 30
        high_number = 70

        # Divide data into 30, 40 ,30
        low_size, high_size = np.percentile(relative_sizes, [low_number, high_number])
        low_ratio, high_ratio = np.percentile(width_height_ratio, [low_number, high_number])

        lowest_30_size = relative_sizes[relative_sizes < low_size]
        middle_40_size = relative_sizes[(relative_sizes >= low_size) & (relative_sizes <= high_size)]
        highest_30_size = relative_sizes[relative_sizes > high_size]

        lowest_30_ratio = width_height_ratio[width_height_ratio < low_ratio]
        middle_40_ratio = width_height_ratio[(width_height_ratio >= low_ratio) & (width_height_ratio <= high_ratio)]
        highest_30_ratio = width_height_ratio[width_height_ratio > high_ratio]
        # Compute the Euclidean distance of each point from the origin
        # Compute the Euclidean distance of each point from the origin
        distances = np.sqrt(np.sum(np.square(relative_position), axis=1))

        # Divide the distances into quantiles
        low_distance, high_distance = np.percentile(distances, [low_number, high_number])

        # Divide the positions based on these distances
        lowest_30_position = relative_position[distances < low_distance]
        middle_40_position = relative_position[(distances >= low_distance) & (distances <= high_distance)]
        highest_30_position = relative_position[distances > high_distance]

        # Compute the median distance for each group
        lowest_30_distance = np.median(distances[distances < low_distance])
        middle_40_distance = np.median(distances[(distances >= low_distance) & (distances <= high_distance)])
        highest_30_distance = np.median(distances[distances > high_distance])

        # Print the median distance for each group
        print('Median distance for the closest '+str(low_number)+'% of points:', lowest_30_distance)
        print('Median distance for the middle '+str(high_number-low_number)+'% of points:', middle_40_distance)
        print('Median distance for the farthest '+str(100-high_number)+'% of points:', highest_30_distance)
        print('max distance and min distance:', np.max(distances), np.min(distances))

        # Now you have your divided data, and you can use them for any further analysis. For example, you could print the average values:
        print('Median size of the smallest '+str(low_number)+'% of boxes:', np.median(lowest_30_size))
        print('Median size of the middle '+str(high_number-low_number)+'% of boxes:', np.median(middle_40_size))
        print('Median size of the largest '+str(100-high_number)+'% of boxes:', np.median(highest_30_size))
        print('max size and min size:', np.max(relative_sizes), np.min(relative_sizes))

        print('Median ratio of the smallest '+str(low_number)+'% of boxes:', np.median(lowest_30_ratio))
        print('Median ratio of the middle '+str(high_number-low_number)+'% of boxes:', np.median(middle_40_ratio))
        print('Median ratio of the largest '+str(100-high_number)+'% of boxes:', np.median(highest_30_ratio))
        print('max ratio and min ratio:', np.max(width_height_ratio), np.min(width_height_ratio))

        sample_size_per_group =  10 # floor 270/23 
        if outputSelection:
            df = pd.DataFrame(columns=['image_path', 'category', 'bbox', 'informationType', 'informativeness', 'relative_size', 'relative_position', 'width_height_ratio'])
            for i in range(len(relative_sizes)):
                df.loc[i] = [bbox_name[i].split('_')[0], category[i], bboxes[i], information_type[i], informativeness[i], relative_sizes[i], relative_position[i], width_height_ratio[i]]
            # Add a new column to identify the group of each row
            df['size_group'] = pd.cut(df['relative_size'], bins=[-np.inf, low_size, high_size, np.inf], labels=['low', 'middle', 'high'])
            df['position_group'] = pd.cut(df['relative_position'].apply(lambda x: np.sqrt(x[0]**2 + x[1]**2)), bins=[-np.inf, low_distance, high_distance, np.inf], labels=['low', 'middle', 'high'])
            df['ratio_group'] = pd.cut(df['width_height_ratio'], bins=[-np.inf, low_ratio, high_ratio, np.inf], labels=['low', 'middle', 'high'])
            
            # Group by image_path, category, and the new group columns, then aggregate the bboxes and informativeness
            grouped = df.groupby(['image_path', 'category', 'size_group', 'position_group', 'ratio_group']).agg({
                'bbox': lambda x: [list(b) for b in x],  # convert each bbox into a list, resulting in a list of lists
                'informativeness': lambda x: np.round(np.mean(x), 2), # calculate the mean informativeness, to 2 decimal places
                'informationType': lambda x: np.any(np.array(list(x)), axis=0).astype(int).tolist() # calculate the logical OR of the informationType, to get a list of 0s and 1s
            }).reset_index()
            #print unique bbox
            #remove all column if the informativeness or bbox is nan
            grouped = grouped.dropna(subset=['informativeness', 'bbox'])
            # save the dataframe
            # randomly choose sample_size rows
            # Stratified sampling
            grouped['informativeness_group'] = pd.cut(grouped['informativeness'], bins=[-np.inf, 2, 4.0001, np.inf], labels=['low', 'middle', 'high'])
            grouped['group (size, position, ratio)'] = grouped['size_group'].astype(str) + "_" + grouped['position_group'].astype(str) + "_" + grouped['ratio_group'].astype(str)
            

            stratified = grouped.groupby('group (size, position, ratio)').apply(lambda x: x.sample(min(len(x), 1000), random_state=0))
            # add a new col ''bbox_name''
            stratified['bbox_name'] = stratified['image_path'] + '_' + stratified['category']
            # we need to ensure that each group (size, position, ratio) has exact 10 samples
            # then we want to make the samples evenly distributed in category as much as possible.
            # we have 23 categories, so each category will try to have 270/23 = 12 samples (some should be 11 to make sure the sum is 270)
            
            all_categories = stratified['category'].unique()
            sample_size_per_category = 270 // 23  # ideal number of samples per category across all groups
            category_num = {category: 0 for category in all_categories}
            selected_data = []
            selected_files = set()

            final_samples_list = []  # This will be used to store all our final samples
            def categories_sorted_by_count(all_categories, category_num):
                return sorted(all_categories, key=lambda x: category_num[x])
            for group in stratified['group (size, position, ratio)'].unique():
                group_df = stratified[stratified['group (size, position, ratio)'] == group]
                
                samples_for_group = []  # samples collected for the current group
                
                # Try to get samples from each category for this group
                for category in categories_sorted_by_count(all_categories, category_num):
                    if len(samples_for_group) >= 10:
                        break

                    if category_num[category] >= sample_size_per_category:
                        continue
                    
                    available_samples = group_df[(group_df['category'] == category) 
                                                & (~group_df['bbox_name'].isin(selected_data))
                                                & (~group_df['image_path'].isin(selected_files))]
                    
                    if not available_samples.empty:
                        selected_sample = available_samples.sample(n=1, random_state=0)
                        samples_for_group.append(selected_sample)
                        selected_data.extend(selected_sample['bbox_name'].tolist())
                        selected_files.add(selected_sample['image_path'].values[0])
                        category_num[category] += 1
                
                # If not enough samples selected for this group, take extra from any available category
                while len(samples_for_group) < 10:
                    for category in categories_sorted_by_count(all_categories, category_num):
                        if len(samples_for_group) >= 10:
                            break
                        
                        available_samples = group_df[(group_df['category'] == category) 
                                                    & (~group_df['bbox_name'].isin(selected_data))
                                                    & (~group_df['image_path'].isin(selected_files))]
                        if not available_samples.empty:
                            selected_sample = available_samples.sample(n=1, random_state=0)
                            samples_for_group.append(selected_sample)
                            selected_data.extend(selected_sample['bbox_name'].tolist())
                            selected_files.add(selected_sample['image_path'].values[0])
                            category_num[category] += 1

                assert len(samples_for_group) == 10, f"Group {group} has {len(samples_for_group)} samples instead of 10."
                final_samples_list.extend(samples_for_group)


            # Convert the list of selected dataframes into a single dataframe
            # print len
            print('len of final samples:', len(final_samples_list))

            final_samples_df = pd.concat(final_samples_list).reset_index(drop=True)
            # print the len of each group
            print('len of each group:', final_samples_df.groupby('group (size, position, ratio)').size())

            final_samples_df.to_csv('final_samples.csv')

            # Print the sample count for each category
            print("Sample Counts by Category:")
            for category, count in category_num.items():
                print(f"{category}: {count} samples")

            # add up count in category_num
            total_count = 0
            for category, count in category_num.items():
                total_count += count
            print('total count:', total_count)


        if visualization:
            # compare with vizwiz-Priv dataset
            if compare:
                # read npy file to get vizwiz's results
                vizwiz_relative_position = np.load('vizwiz_relative_position.npy')
                vizwiz_relative_sizes = np.load('vizwiz_relative_sizes.npy')
                vizwiz_width_height_ratio = np.load('vizwiz_width_height_ratio.npy')
                # only show vizwiz data in one plt
                plt.scatter(vizwiz_relative_position[:, 0], vizwiz_relative_position[:, 1], s=vizwiz_relative_sizes * 1000, c=vizwiz_width_height_ratio, cmap='viridis')
                #add title for colorbar
                plt.clim(0, 2)
                plt.colorbar().ax.set_title('aspect ratio')
                #add x,y axis title
                plt.xlabel('relative position to center (width)')
                plt.ylabel('relative position to center (height)')
                #add title
                plt.title('relative position and size of bounding box')
                plt.show()

                
                
            else:
                plt.scatter(relative_position[:, 0], relative_position[:, 1], s=relative_sizes * 1000, c=width_height_ratio, cmap='viridis')
                
                #add title for colorbar
                plt.clim(0, 2)
                plt.colorbar().ax.set_title('aspect ratio')
                #add x,y axis title
                plt.xlabel('relative position to center (width)')
                plt.ylabel('relative position to center (height)')
                #add title
                plt.title('relative position and size of bounding box')
                plt.show()
                #visualize the distribution in histogram
                plt.hist(relative_sizes, bins=20)
                plt.title('relative size of bounding box')
                plt.show()
                plt.hist(relative_position[:, 0], bins=20)
                plt.title('relative x position of bounding box')
                plt.show()
                plt.hist(relative_position[:, 1], bins=20)
                plt.title('relative y position of bounding box')
                plt.show()
                plt.hist(width_height_ratio, bins=20)
                plt.title('width height ratio of bounding box')
                plt.show()
    
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
    #analyze.prepare_mega_table(mycat_mode = False, save_csv=True, strict_mode=True, ignore_prev_manual_anns=False, include_not_private=False)
    #analyze.prepare_manual_label(save_csv=True, strict_mode=True)
    #analyze.basic_count(read_csv = True, ignore_prev_manual_anns=False,split_count=False,count_scale='CrowdWorks')
    #analyze.prepare_regression_model_table(read_csv=True)
    #analyze.regression_model(input_channel=input_channel, output_channel=output_channel, read_csv=True)
    #analyze.count_frequency()
    #analyze.count_overlap_in_manual_table(split_count=False, count_scale='Prolific')
    analyze.visualDistribution(read_csv=True, visualization=True, filter_data=True, filter_corridor=(10,90), outputSelection=True, only_manual=False, compare=True)



    