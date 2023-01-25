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

        with open(self.img_annotation_map_path) as f:
            self.img_annotation_map = json.load(f)

    def prepare_mega_table(self, mycat_mode = True)->None:
        #mycat_mode: only aggregate annotations that can be summarized in mycat (also store them in mycat in mega_table).
        #the mega table includes all privacy annotations with all corresponding info (three metrics, big five, age, gender, platform)

        # make sure this sequence is correct.
        self.mega_table = pd.DataFrame(columns=["category", "reason", "informativeness", "shaaring", 'age', 'gender', 'platform', 'bigfive'])
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

if __name__ == '__main__':
    pass