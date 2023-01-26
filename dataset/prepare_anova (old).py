import os
import csv
import json
import shutil

class anova:
    def __init__(self) -> None:
        self.img_annotation_map_path = './img_annotation_map.json'
        self.code_openimage_map = {}
        self.openimages_mycat_map = {}
        self.lvis_mycat_map = {}
        self.folder = './annotations/'
        self.original = './original labels/'
        self.new_folder = './annotations (filtered)/'
        self.new_original = './original labels (filtered)/'
        self.platforms = ['CrowdWorks', 'Prolific']
        self.label_mycat = {}
        self.valid_label_list = []
        self.img_annotation_map = {}
        #self.new_img_annotation_map = {}
        with open(self.img_annotation_map_path) as f:
            self.img_annotation_map = json.load(f)
        self.data_preparation_mycat()

    def data_preparation_mycat(self)->None:
        # for OpenImages 
            
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

            cnt = 0
            not_mycat = 0
            photo_cnt = 0
            for platform in self.platforms:
                used_image = []
                folder = os.path.join(self.folder,platform,'labels')
                labels = os.listdir(folder)
                for label_path in labels:
                    img_id = label_path.split('_')[0]
                    with open(os.path.join(folder,label_path)) as f:
                        text = f.read()
                        label = json.loads(text)
                        dataset_name = label['source']
                        for key, value in label['defaultAnnotation'].items():
                            mycat = ''
                            if dataset_name == 'OpenImages':
                                if key in self.openimages_mycat_map.keys():
                                    mycat = self.openimages_mycat_map[key]
                            elif dataset_name == 'LVIS':
                                if key in self.lvis_mycat_map.keys():
                                    mycat = self.lvis_mycat_map[key]
                            if mycat == '':
                                not_mycat += 1
                                continue
                            if value['ifNoPrivacy']:
                                continue
                            self.label_mycat[str(cnt)] = value
                            self.label_mycat[str(cnt)]['mycat'] = mycat
                            #self.label_mycat[str(cnt)]['photoNum'] = photo_cnt
                            cnt += 1
                        #photo_cnt += 1

    def prepare_anova_csv(self)->None:
        print(self.label_mycat)
        with open('for_anova (mycat).csv', 'w') as w:
            w.write('ID,my_cat,reason,informativeness,sharing\n')
            for key, value in self.label_mycat.items():
                w.write(str(key)+ ',' + str(value['mycat']) + ',' + str(value['reason']) + ',' + str(value['informativeness']) + ',' + str(value['sharing']) + '\n')
    # remove all triple annotations
    # This code is already be used
    def filter_dataset(self)->None:
        with open('new_img_annotation_map.json', 'w') as w:
            json.dump(self.img_annotation_map, w)
        for key in self.img_annotation_map.keys():
            for platform, value in self.img_annotation_map[key].items():
                original_file = value[0].split('_')[0] + '_label'
                prefix_len = len(value[0].split('_')[0]) + 1
                worker_file = value[0][prefix_len:]
                worker_file = worker_file[:-11]
                worker_file = worker_file + '.json'
                shutil.copyfile(os.path.join(self.folder, platform, 'labels', value[0]), os.path.join(self.new_folder, platform, 'labels', value[0]))
                shutil.copyfile(os.path.join(self.folder, platform, 'workerinfo', worker_file), os.path.join(self.new_folder, platform, 'workerinfo', worker_file))
                shutil.copyfile(os.path.join(self.original, original_file), os.path.join(self.new_original, original_file))

if __name__ == '__main__':
    aov = anova()
    #print(aov.label_mycat)
    aov.prepare_anova_csv()