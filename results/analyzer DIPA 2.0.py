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
        'informativeness':['negligible source','minor source','slight source','moderate source',
        'effective source','substantial','maximum source'],
        'sharingOwner': ['I won\'t share it', 'Close relationship',
        'Regular relationship', 'Acquaintances', 'Public', 'Broadcast program', 'Other recipients'], 
        'sharingOthers':['I won\'t allow others to share it', 'Close relationship',
        'Regular relationship', 'Acquaintances', 'Public', 'Broadcast program', 'Other recipients'],
        'frequency': ['Never', 'Less than once a month', 'Once or more per month', 
        'Once or more per week', 'Once or more per day']}
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

    def generate_img_annotation_map(self)->None:
        #label: the original label from OpenImages or LVIS 
        #annotation: the privacy-oriented annotations from our study
        img_annotation_map = {}
        valid_workers = []
        crowdworks_labels = os.listdir(os.path.join(self.annotation_path, 'CrowdWorks', 'labels'))
        with open(os.path.join(self.annotation_path, 'CrowdWorks', 'valid_workers.json')) as f:
            valid_workers = json.load(f)
        print('val len:', len(valid_workers))
        #prolific_labels = os.listdir(os.path.join(self.annotation_path, 'Prolific', 'crowdscouringlabel'))
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

        '''for label_path in prolific_labels:
            img_name = label_path.split('_')[0]
            if img_name != '':
                if img_name not in img_annotation_map.keys():
                    img_annotation_map[img_name] = {}
                if 'Prolific' not in img_annotation_map[img_name].keys():
                    img_annotation_map[img_name]['Prolific'] = [label_path]
                else:
                    img_annotation_map[img_name]['Prolific'].append(label_path)'''

        with open('img_annotation_map.json', 'w') as w:
            json.dump(img_annotation_map, w)

    def prepare_mega_table(self, mycat_mode = True, save_csv = False)->None:
        #mycat_mode: only aggregate annotations that can be summarized in mycat (also score them in mycat in mega_table).
        #the mega table includes all privacy annotations with all corresponding info (three metrics, big five, age, gender, platform)

        # make sure this sequence is correct.
        self.mega_table = pd.DataFrame(columns=["category", "informationType", "informativeness", "sharingOwner", "sharingOthers", 'age', 'gender', 
        'platform', 'extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness', 'frequency', 'imagePath', 'originCategory', 'datasetName'])
        for image_name in self.img_annotation_map.keys():
            for platform, value in self.img_annotation_map[image_name].items():
                # now, value[0] is the only availiable index
                image_id = value[0].split('_')[0]
                prefix_len = len(image_id) + 1
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
                    frequency = worker['frequency']
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
                        id = image_id + '_' + key
                        informationType = value['informationType']
                        informativeness = int(value['informativeness']) - 1
                        sharingOwner = value['sharingOwner']
                        sharingOthers = value['sharingOthers']
                        if informationType[5] == 1:
                            self.custom_informationType.append(value['informationTypeInput'])
                        if sharingOwner[6] == 1:
                            self.custom_recipient_owner.append(value['sharingOwenerInput'])
                        if sharingOthers[6] == 1:
                            self.custom_recipient_others.append(value['sharingOthersInput'])
                        entry = pd.DataFrame.from_dict({
                            'id': [id],
                            "category": [category],
                            "informationType":  [informationType],
                            "informativeness": [informativeness],
                            "sharingOwner": [sharingOwner],
                            "sharingOthers": [sharingOthers],
                            "age": [age],
                            "gender": [gender],
                            "platform": [platform],
                            "extraversion": [extraversion],
                            "agreeableness": [agreeableness],
                            "conscientiousness": [conscientiousness],
                            "neuroticism": [neuroticism],
                            "openness": [openness],
                            'frequency': [frequency],
                            'imagePath': [image_name + '.jpg'],
                            'originCategory': value['category'],
                            'datasetName': [dataset_name]
                        })

                        self.mega_table = pd.concat([self.mega_table, entry], ignore_index=True)
        if save_csv:
            self.mega_table.to_csv('./mega_table.csv', index =False)

    def prepare_manual_label(self, save_csv = False) -> None:
        self.manual_table = pd.DataFrame(columns=["category", "informationType", "informativeness", "sharingOwner", "sharingOthers", 'age', 'gender', 
        'platform', 'frequency', 'extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness'])
        for key in self.img_annotation_map.keys():
            for platform, value in self.img_annotation_map[key].items():
                # now, value[0] is the only availiable index
                image_id = value[0].split('_')[0]
                prefix_len = len(image_id) + 1
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
                    frequency = worker['frequency']      
                    for key, value in label['manualAnnotation'].items():
                        category = value['category']
                        id = image_id + '_' + key
                        informationType = value['informationType']
                        informativeness = int(value['informativeness']) - 1
                        sharingOwner = value['sharingOwner']
                        sharingOthers = value['sharingOthers']
                        if informationType[5] == 1:
                            self.custom_informationType.append(value['informationTypeInput'])
                        if sharingOwner[6] == 1:
                            self.custom_recipient_owner.append(value['sharingOwenerInput'])
                        if sharingOthers[6] == 1:
                            self.custom_recipient_others.append(value['sharingOthersInput'])
                        entry = pd.DataFrame.from_dict({
                            'id': [id],
                            "category": [category],
                            "informationType":  [informationType],
                            "informativeness": [informativeness],
                            "sharingOwner": [sharingOwner],
                            "sharingOthers": [sharingOthers],
                            "age": [age],
                            "gender": [gender],
                            "platform": [platform],
                            "extraversion": [extraversion],
                            "agreeableness": [agreeableness],
                            "conscientiousness": [conscientiousness],
                            "neuroticism": [neuroticism],
                            "openness": [openness],
                            'frequency': [frequency]
                        })

                        self.manual_table = pd.concat([self.manual_table, entry], ignore_index=True)
        if save_csv:
            self.manual_table.to_csv('./manual_table.csv', index =False)
    def basic_count(self, read_csv = False) -> None:

        def calculate_array(input_array, option_num):
            res = np.zeros(option_num, dtype='int')
            for i in range(input_array.shape[0]):
                res += np.array(input_array[i])
            return res
        if read_csv:
            self.mega_table = pd.read_csv('./mega_table.csv')
        else:
            self.prepare_mega_table()

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

    def regression_model(self, input_channel, output_channel, read_csv = False)->None:
        if read_csv:
            self.mega_table = pd.read_csv('./mega_table.csv')
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
            self.mega_table = pd.read_csv('./mega_table.csv')
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
            self.mega_table = pd.read_csv('./mega_table.csv')
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
            self.mega_table = pd.read_csv('./mega_table.csv')
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
            self.mega_table = pd.read_csv('./mega_table.csv')
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
    basic_info = [ "age", "gender", "platform", 'frequency']
    category = ['category']
    privacy_metrics = ['informationType', 'informativeness', 'sharingOwner', 'sharingOthers']

    input_channel = []
    output_channel = []

    input_channel.extend(bigfives)
    input_channel.extend(basic_info)
    input_channel.extend(category)
    output_channel = privacy_metrics

    #analyze.generate_img_annotation_map()
    analyze.prepare_mega_table(mycat_mode = False, save_csv=True)
    analyze.basic_count()
    #analyze.regression_model(input_channel=input_channel, output_channel=output_channel, read_csv=True)
    