import os
import csv
import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.metrics import classification_report
import statsmodels.api as sm
from statsmodels.formula.api import ols

from neural_network import  nn_model
from neural_network import nn_dataset
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
                        reason = int(value['reason']) - 1
                        informativeness = int(value['informativeness']) - 1
                        sharing = int(value['sharing']) - 1
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

        classifier=svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.8)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)

        # Print evaluation metrics
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Precision:",metrics.precision_score(y_test, y_pred,average='weighted'))
        print("Recall:",metrics.recall_score(y_test, y_pred,average='weighted'))

    def knn(self,input_channel, output_channel, read_csv = False) -> None:
        if read_csv:
            self.mega_table = pd.read_csv('./mega_table.csv')
        else:
            self.prepare_mega_table()
        
        
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

        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(X_train, np.ravel(y_train,order="c"))
        y_pred = classifier.predict(X_test)

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

    def neural_network(self, input_channel, output_channel, read_csv = False) -> None:
        
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
        
        loss_fns = [nn.CrossEntropyLoss() for output in output_channel]
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        training_dataset = nn_dataset(X_train, y_train)
        testing_dataset = nn_dataset(X_test, y_test)
        training_loader = DataLoader(training_dataset, batch_size=64, shuffle=True)
        testing_loader = DataLoader(testing_dataset, batch_size=64, shuffle=True)

        #start training
        writer = SummaryWriter()
        epoch_number = 0
        EPOCHS = 50

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
                tot_vloss = 0
                for loss in losses:
                    tot_vloss += loss
                
                running_vloss += tot_vloss
            acc = acc / (i + 1)
            pre = pre / (i + 1)
            rec = rec / (i + 1)
            print("Accuracy:",acc)
            print("Precision:",pre)
            print("Recall:",rec)
            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            
            

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
    basic_info = [ "age", "gender", "platform"]
    category = ['category']
    privacy_metrics = ['reason', 'informativeness', 'sharing']
    input_channel = []
    input_channel.extend(basic_info)
    input_channel.extend(category)
    input_channel.extend(bigfives)
    print(input_channel)
    output_channel = privacy_metrics
    #analyze.prepare_mega_table(save_csv=True)
    #analyze.svm(input_channel, output_channel, read_csv=True)
    #analyze.anova(True)
    analyze.neural_network(input_channel, output_channel, read_csv=True)
    #analyze.knn(input_channel, output_channel, read_csv=True)
    