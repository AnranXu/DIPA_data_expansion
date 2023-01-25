import os
import json
import argparse
import numpy as np
import csv
class analyzer:
    def __init__(self, platform) -> None:
        self.platform = platform
        self.worker_info = {'age': [], 'gender': {'Male': 0, 'Female': 0, 'Other': 0}, 'bigfive': []}
        self.task_record_path = platform + '/' + 'task_record.json'
        self.label_folder = platform + '/' + 'crowdscouringlabel/'
        self.mycat = {'OpenImages': {}, 'LVIS': {}, 'all': {}}
        self.valid_workers = []
        self.valid_images = []
        self.default_category = {'OpenImages': {}, 'LVIS': {}}
        self.manual_category = {'OpenImages': {}, 'LVIS': {}}
        self.privacy_count_by_image = {'OpenImages': 0, 'LVIS': 0}
        self.nonprivacy_count_by_image = {'OpenImages': 0, 'LVIS': 0}
        self.privacy_count_by_label = {'OpenImages': 0, 'LVIS': 0}
        self.nonprivacy_count_by_label = {'OpenImages': 0, 'LVIS': 0}
        self.code_openimage_map = {}
        self.openimages_mycat_map = {}
        self.lvis_mycat_map = {}
        self.reason = np.zeros(5)
        self.importance = np.zeros(7)
        self.sharing = np.zeros(5)
        self.unique_content = {}
        with open(self.task_record_path, encoding='utf-8') as f:
            text = f.read()
            self.task_record = json.loads(text)
        with open('img_list') as f:
            self.valid_images = f.read().splitlines() 
    
    def basic_info(self, select_bar)->None:
        record_path = os.path.join(self.platform, 'task_record (original).json')
        age = {'18-24': {'Male': 0, 'Female': 0, 'Other': 0}, 
        '25-34': {'Male': 0, 'Female': 0, 'Other': 0}, 
        '35-44': {'Male': 0, 'Female': 0, 'Other': 0}, 
        '45-54': {'Male': 0, 'Female': 0, 'Other': 0}, 
        '55': {'Male': 0, 'Female': 0, 'Other': 0}}
        with open(record_path, encoding='utf-8') as f:
            text = f.read()
            record = json.loads(text)
            list_len = record['list_len']
            for i in range(list_len):
                worker_record = record[str(i)]
                if worker_record['workerprogress'] > select_bar:
                    self.valid_workers.append(worker_record['workerid'])
        info_paths = os.listdir(os.path.join(self.platform, 'workerInfo'))
        
        for info_path in info_paths:
            # check if nan, nan!=nan
            with open(os.path.join(self.platform, 'workerInfo', info_path)) as f:
                text = f.read()
                info = json.loads(text)
                if info['workerId'] not in self.valid_workers:
                    print('unvalid worker: ', info['workerId'])
                    continue
                if int(info['age']) != int(info['age']):
                    print('wrong age found', info)
                    continue
                year = int(info['age'])
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
        print('valid worker:', len(self.valid_workers))
        print(self.worker_info)
        print(age)

    def basic_count(self) -> None:
        ## count each original labels' annotations 
        ## generate a list the map image_id to its annotations 
        labels = os.listdir(self.label_folder)
        manual_num = 0
        used_image = []
        for label in labels:
            img_id = label.split('_')[0]
            if img_id not in self.valid_images:
                continue
            if img_id not in used_image:
                used_image.append(img_id)
            else:
                continue
            with open(os.path.join(self.label_folder, label), encoding='utf-8') as f:
                text = f.read()
                record = json.loads(text)
                ifPrivacy = False
                source  = record['source']
                if source not in ['OpenImages', 'LVIS']:
                    continue
                for key, value in record['defaultAnnotation'].items():
                    if key not in self.default_category[source].keys():
                        self.default_category[source][key] = {'reason': np.zeros(5), 'importance': np.zeros(7), 'sharing': np.zeros(5), 
                        'reasonInput': [], 'sharingInput': [], 'notPrivacy': 0, 'privacy': 0, 'num': 0}
                    self.default_category[source][key]['num'] += 1
                    if record['defaultAnnotation'][key]['ifNoPrivacy']:
                        self.default_category[source][key]['notPrivacy'] += 1
                        self.nonprivacy_count_by_label[source] += 1
                        continue
                    if img_id not in self.unique_content.keys():
                        self.unique_content[img_id] = {'keys': [], 'count': 0}
                    if key not in self.unique_content[img_id]['keys']:
                        self.unique_content[img_id]['keys'].append(key) 
                        self.unique_content[img_id]['count'] += 1
                    self.privacy_count_by_label[source] += 1
                    self.default_category[source][key]['privacy'] += 1
                    ifPrivacy = True
                    #reason
                    reason_value = int(record['defaultAnnotation'][key]['reason']) - 1
                    self.default_category[source][key]['reason'][reason_value] += 1
                    self.reason[reason_value] += 1
                    # if other reasons
                    if reason_value == 4:
                        self.default_category[source][key]['reasonInput'].append(record['defaultAnnotation'][key]['reasonInput'])
                    # importance
                    importance_value = int(record['defaultAnnotation'][key]['importance']) - 1
                    self.default_category[source][key]['importance'][importance_value] += 1
                    self.importance[importance_value] += 1
                    # sharing
                    sharing_value = int(record['defaultAnnotation'][key]['sharing']) - 1
                    self.default_category[source][key]['reason'][sharing_value] += 1
                    self.sharing[sharing_value] += 1
                    # if other sharing
                    if sharing_value == 4:
                        self.default_category[source][key]['sharingInput'].append(record['defaultAnnotation'][key]['sharingInput'])

                for key, value in record['manualAnnotation'].items():
                    category = record['manualAnnotation'][key]['category']
                    if category not in self.manual_category[source].keys():
                        self.manual_category[source][category] = {'num': 0, 'reason': np.zeros(5), 'importance': np.zeros(7), 'sharing': np.zeros(5), 
                        'reasonInput': [], 'sharingInput': []}
                    #num += 1
                    if img_id not in self.unique_content.keys():
                        self.unique_content[img_id] = {'keys': [], 'count': 0}
                    self.unique_content[img_id]['count'] += 1
                    self.privacy_count_by_label[source] += 1
                    manual_num += 1
                    ifPrivacy = True
                    self.manual_category[source][category]['num'] += 1
                    #reason
                    reason_value = int(record['manualAnnotation'][key]['reason']) - 1
                    self.manual_category[source][category]['reason'][reason_value] += 1
                    self.reason[reason_value] += 1
                    # if other reasons
                    if reason_value == 4:
                        self.manual_category[source][category]['reasonInput'].append(record['manualAnnotation'][key]['reasonInput'])
                    # importance
                    importance_value = int(record['manualAnnotation'][key]['importance']) - 1
                    self.manual_category[source][category]['importance'][importance_value] += 1
                    self.importance[importance_value] += 1
                    # sharing
                    sharing_value = int(record['manualAnnotation'][key]['sharing']) - 1
                    self.manual_category[source][category]['reason'][sharing_value] += 1
                    self.sharing[sharing_value] += 1
                    # if other sharing
                    if sharing_value == 4:
                        self.manual_category[source][category]['sharingInput'].append(record['manualAnnotation'][key]['sharingInput'])
                if ifPrivacy:
                    self.privacy_count_by_image[source] += 1
                else:
                    self.nonprivacy_count_by_image[source] += 1
        '''
        #only used for uniqueness for getting double annotations
        self.label_folder = 'Prolific' + '/' + 'crowdscouringlabel/'
        labels = os.listdir(self.label_folder)
        manual_num = 0
        used_image = []
        for label in labels:
            img_id = label.split('_')[0]
            if img_id not in self.valid_images:
                continue
            if img_id not in used_image:
                used_image.append(img_id)
            else:
                continue
            with open(os.path.join(self.label_folder, label), encoding='utf-8') as f:
                text = f.read()
                record = json.loads(text)
                ifPrivacy = False
                source  = record['source']
                if source not in ['OpenImages', 'LVIS']:
                    continue
                for key, value in record['defaultAnnotation'].items():
                    if key not in self.default_category[source].keys():
                        self.default_category[source][key] = {'reason': np.zeros(5), 'importance': np.zeros(7), 'sharing': np.zeros(5), 
                        'reasonInput': [], 'sharingInput': [], 'notPrivacy': 0, 'privacy': 0, 'num': 0}
                    self.default_category[source][key]['num'] += 1
                    if record['defaultAnnotation'][key]['ifNoPrivacy']:
                        self.default_category[source][key]['notPrivacy'] += 1
                        self.nonprivacy_count_by_label[source] += 1
                        continue
                    if img_id not in self.unique_content.keys():
                        self.unique_content[img_id] = {'keys': [], 'count': 0}
                    if key not in self.unique_content[img_id]['keys']:
                        self.unique_content[img_id]['keys'].append(key) 
                        self.unique_content[img_id]['count'] += 1
                    self.privacy_count_by_label[source] += 1
                    self.default_category[source][key]['privacy'] += 1
                    ifPrivacy = True
                    #reason
                    reason_value = int(record['defaultAnnotation'][key]['reason']) - 1
                    self.default_category[source][key]['reason'][reason_value] += 1
                    self.reason[reason_value] += 1
                    # if other reasons
                    if reason_value == 4:
                        self.default_category[source][key]['reasonInput'].append(record['defaultAnnotation'][key]['reasonInput'])
                    # importance
                    importance_value = int(record['defaultAnnotation'][key]['importance']) - 1
                    self.default_category[source][key]['importance'][importance_value] += 1
                    self.importance[importance_value] += 1
                    # sharing
                    sharing_value = int(record['defaultAnnotation'][key]['sharing']) - 1
                    self.default_category[source][key]['reason'][sharing_value] += 1
                    self.sharing[sharing_value] += 1
                    # if other sharing
                    if sharing_value == 4:
                        self.default_category[source][key]['sharingInput'].append(record['defaultAnnotation'][key]['sharingInput'])

                for key, value in record['manualAnnotation'].items():
                    category = record['manualAnnotation'][key]['category']
                    if category not in self.manual_category[source].keys():
                        self.manual_category[source][category] = {'num': 0, 'reason': np.zeros(5), 'importance': np.zeros(7), 'sharing': np.zeros(5), 
                        'reasonInput': [], 'sharingInput': []}
                    #num += 1
                    if img_id not in self.unique_content.keys():
                        self.unique_content[img_id] = {'keys': [], 'count': 0}
                    self.unique_content[img_id]['count'] += 1
                    self.privacy_count_by_label[source] += 1
                    manual_num += 1
                    ifPrivacy = True
                    self.manual_category[source][category]['num'] += 1
                    #reason
                    reason_value = int(record['manualAnnotation'][key]['reason']) - 1
                    self.manual_category[source][category]['reason'][reason_value] += 1
                    self.reason[reason_value] += 1
                    # if other reasons
                    if reason_value == 4:
                        self.manual_category[source][category]['reasonInput'].append(record['manualAnnotation'][key]['reasonInput'])
                    # importance
                    importance_value = int(record['manualAnnotation'][key]['importance']) - 1
                    self.manual_category[source][category]['importance'][importance_value] += 1
                    self.importance[importance_value] += 1
                    # sharing
                    sharing_value = int(record['manualAnnotation'][key]['sharing']) - 1
                    self.manual_category[source][category]['reason'][sharing_value] += 1
                    self.sharing[sharing_value] += 1
                    # if other sharing
                    if sharing_value == 4:
                        self.manual_category[source][category]['sharingInput'].append(record['manualAnnotation'][key]['sharingInput'])
                if ifPrivacy:
                    self.privacy_count_by_image[source] += 1
                else:
                    self.nonprivacy_count_by_image[source] += 1
        tot = 0
        for value in self.unique_content.values():
            tot += value['count']
        print('unique number: ', tot)'''
        print('manual num: ', manual_num)
        
    #check unfinished task and generate a new task_record.json for only unfinished tasks
    def integrity_check(self, select_bar = 0, generate_new_json = False)->None:
        record_path = os.path.join(self.platform, 'task_record.json')
        new_record = {'cur_progress': '0', 'worker_record': {}}
        cur_step = '0'
        with open(record_path, encoding='utf-8') as f:
            text = f.read()
            record = json.loads(text)
            list_len = record['list_len']
            for i in range(list_len):
                worker_record = record[str(i)]
                if worker_record['workerprogress'] <= select_bar:
                    print(worker_record['workerprogress'])
                    worker_record['workerid'] = ''
                    worker_record['workerprogress'] = 0
                    new_record[cur_step] = worker_record
                    cur_step = str(int(cur_step) + 1)                    

            if cur_step != '0' and generate_new_json:
                new_record['list_len'] = int(cur_step)
                with open('task_record.json', 'w') as w:
                    w.write(str(new_record))


    def check_labels_by_mycat(self)->None:

        # for OpenImages 
        with open('../dataset/OpenImages/csv/oidv6-class-descriptions.csv') as f:
            res = csv.reader(f)
            for row in res:
                self.code_openimage_map[row[0]] = row[1]
            
        with open('../dataset/OpenImages/mycat_openimages_map.csv') as f:
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
        self.openimages_mycat_map['Human eye'] = 'Person'
        self.openimages_mycat_map['Human beard'] = 'Person'
        self.openimages_mycat_map['Human mouth'] = 'Person'
        self.openimages_mycat_map['Human foot'] = 'Person'
        self.openimages_mycat_map['Human leg'] = 'Person'
        self.openimages_mycat_map['Human ear'] = 'Person'
        self.openimages_mycat_map['Human hair'] = 'Person'
        self.openimages_mycat_map['Human head'] = 'Person'
        self.openimages_mycat_map['Human face'] = 'Person'
        self.openimages_mycat_map['Human arm'] = 'Person'
        self.openimages_mycat_map['Human nose'] = 'Person'
        self.openimages_mycat_map['Human hand'] = 'Person'
        self.openimages_mycat_map['Human body'] = 'Person'
        self.openimages_mycat_map['Man'] = 'Person'
        self.openimages_mycat_map['Woman'] = 'Person'
        self.openimages_mycat_map['Boy'] = 'Person'
        self.openimages_mycat_map['Girl'] = 'Person'
        with open('../dataset/LVIS/mycat_lvis_map.csv') as f:
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
        not_privacy = {'OpenImages': 0, 'LVIS': 0, 'all': 0}
        privacy = {'OpenImages': 0, 'LVIS': 0, 'all': 0}
        for dataset_name, dataset_value in self.default_category.items():
            for key, value in self.default_category[dataset_name].items():
                mycat = ''
                if dataset_name == 'OpenImages':
                    if key in self.openimages_mycat_map.keys():
                        mycat = self.openimages_mycat_map[key]
                elif dataset_name == 'LVIS':
                    if key in self.lvis_mycat_map.keys():
                        mycat = self.lvis_mycat_map[key]
                
                if mycat != '':
                    if mycat not in self.mycat[dataset_name].keys():
                        self.mycat[dataset_name][mycat] = {'reason': np.zeros(5), 'importance': np.zeros(7), 'sharing': np.zeros(5), 
                        'reasonInput': [], 'sharingInput': [], 'notPrivacy': 0, 'privacy': 0, 'num': 0}
                    if mycat not in self.mycat['all'].keys():
                        self.mycat['all'][mycat] = {'reason': np.zeros(5), 'importance': np.zeros(7), 'sharing': np.zeros(5), 
                        'reasonInput': [], 'sharingInput': [], 'notPrivacy': 0, 'privacy': 0, 'num': 0}
                    self.mycat[dataset_name][mycat]['reason'] += value['reason']
                    self.mycat[dataset_name][mycat]['importance'] += value['importance']
                    self.mycat[dataset_name][mycat]['sharing'] += value['sharing']
                    self.mycat[dataset_name][mycat]['reasonInput'].extend(value['reasonInput'])
                    self.mycat[dataset_name][mycat]['sharingInput'].extend(value['sharingInput'])
                    self.mycat[dataset_name][mycat]['notPrivacy'] += value['notPrivacy']
                    self.mycat[dataset_name][mycat]['privacy'] += value['privacy']
                    self.mycat[dataset_name][mycat]['num'] += value['num']

                    self.mycat['all'][mycat]['reason'] += value['reason']
                    self.mycat['all'][mycat]['importance'] += value['importance']
                    self.mycat['all'][mycat]['sharing'] += value['sharing']
                    self.mycat['all'][mycat]['reasonInput'].extend(value['reasonInput'])
                    self.mycat['all'][mycat]['sharingInput'].extend(value['sharingInput'])
                    self.mycat['all'][mycat]['notPrivacy'] += value['notPrivacy']
                    self.mycat['all'][mycat]['privacy'] += value['privacy']
                    self.mycat['all'][mycat]['num'] += value['num']
                    privacy[dataset_name] += value['privacy']
                    privacy['all'] += value['privacy']
                    not_privacy[dataset_name] += value['notPrivacy']
                    not_privacy['all'] += value['notPrivacy']
        sorted_category = dict(sorted(self.mycat['all'].items(),\
        key=lambda item: float(item[1]['num']), reverse=True))
        print([ [key, value['privacy'], value['notPrivacy']] for key, value in sorted_category.items()])
        print('my cat privacy: ', privacy)
        print('my cat no privacy: ', not_privacy)

    def merge_task_json(self)->None:
        old_record_path = os.path.join(self.platform, 'task_record (original).json')
        new_record_path = os.path.join(self.platform, 'task_record (new).json')
        with open(old_record_path, encoding='utf-8') as old_file, open(new_record_path, encoding='utf-8') as new_file,\
             open('task_record (merged).json', 'w', encoding='utf-8') as w:
            old_text = old_file.read()
            new_text = new_file.read()
            record = json.loads(old_text)
            new_record = json.loads(new_text)
            # maybe have bugs because the cur_progress can be reset
            record['cur_progress'] = str(record['list_len'] + int(new_record['cur_progress']))
            record['worker_record'] = {**record['worker_record'], **new_record['worker_record']}  
            for i in range(new_record['list_len']):
                index  = str(i + record['list_len'])
                record[index] = new_record[str(i)]
            record['list_len'] += new_record['list_len']
            w.write(str(record))


    def generate_img_annotation_map(self)->None:
        #label: the original label from OpenImages or LVIS 
        #annotation: the privacy-oriented annotations from our study
        img_annotation_map = {}
        crowdworks_labels = os.listdir(os.path.join('CrowdWorks', 'crowdscouringlabel'))
        prolific_labels = os.listdir(os.path.join('Prolific', 'crowdscouringlabel'))
        for label_path in crowdworks_labels:
            img_name = label_path.split('_')[0]
            if img_name != '':
                if img_name not in img_annotation_map.keys():
                    img_annotation_map[img_name] = {}
                if 'CrowdWorks' not in img_annotation_map[img_name].keys():
                    img_annotation_map[img_name]['CrowdWorks'] = [label_path]
                else:
                    img_annotation_map[img_name]['CrowdWorks'].append(label_path)

        for label_path in prolific_labels:
            img_name = label_path.split('_')[0]
            if img_name != '':
                if img_name not in img_annotation_map.keys():
                    img_annotation_map[img_name] = {}
                if 'Prolific' not in img_annotation_map[img_name].keys():
                    img_annotation_map[img_name]['Prolific'] = [label_path]
                else:
                    img_annotation_map[img_name]['Prolific'].append(label_path)
        key_to_delete = []
        for img_name, annotations in img_annotation_map.items():
            ifPrivacy = False
            for platform, label_names in annotations.items():
                for label_name in label_names:
                    dirname = os.path.join(platform,'crowdscouringlabel')
                    with open(os.path.join(dirname,label_name)) as f:
                        text = f.read()
                        record = json.loads(text)
                        for key, value in record['defaultAnnotation'].items():
                            if not record['defaultAnnotation'][key]['ifNoPrivacy']:
                                ifPrivacy = True
                        if len(record['manualAnnotation']):
                            ifPrivacy = True
            if not ifPrivacy:
                key_to_delete.append(img_name)
        for key in key_to_delete:
            del img_annotation_map[key]

        with open('img_annotation_map.json', 'w') as f:
            f.write(str(img_annotation_map))
        with open('img_list', 'w', encoding='utf-8') as f:
            for i, key in enumerate(img_annotation_map.keys()):
                if i != len(img_annotation_map.keys()):
                    f.write(key + '\n')
                else:
                    f.write(key)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', type=str, default='CrowdWorks')
    opt = parser.parse_args()
    platform_name = opt.platform
    analyze = analyzer(platform_name)
    analyze.merge_task_json()
    #analyze.integrity_check(select_bar = 1, generate_new_json=True)
    analyze.basic_info(select_bar = 0)
    analyze.basic_count()
    source = 'LVIS'
    #print(analyze.default_category[source].keys())
    #print(analyze.manual_category[source].keys())
    sorted_category = dict(sorted(analyze.default_category[source].items(),\
        key=lambda item: float(item[1]['privacy'])/float(item[1]['num']), reverse=True))
    privacy_list = [[key, value['privacy']]for key, value in sorted_category.items() if value['privacy'] >= 1]
    print('len of privacy_list: ', len(privacy_list))
    
    analyze.check_labels_by_mycat()
    #analyze.generate_img_annotation_map()
    print('privacy count by image: ', analyze.privacy_count_by_image)
    print('nonprivacy count by image: ', analyze.nonprivacy_count_by_image)
    print('privacy count by label: ', analyze.privacy_count_by_label)
    print('nonprivacy count by label: ', analyze.nonprivacy_count_by_label)
    print('reason:', analyze.reason, np.sum(analyze.reason))
    print('importance:', analyze.importance/np.sum(analyze.importance))
    print('sharing:', analyze.sharing/np.sum(analyze.sharing))
    
    