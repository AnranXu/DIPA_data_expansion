import os
import csv
import json
import pandas as pd

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

    def prepare_mega_table(self, mycat_mode = True)->None:
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

if __name__ == '__main__':
    analyze = analyzer()
    analyze.prepare_mega_table()
    print(analyze.mega_table)