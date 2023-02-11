import os
import json
import shutil
import csv
import numpy as np

class unify_annotation:
    # unify the annotations of each image, so every image only has one annotation file that contains
    # all privacy-threatening content annotated by at least one annotator.
    # Currently, I only record names and bboxes of those privacy-threatening content except existing scores.
    def __init__(self) -> None:
        self.img_annotation_map_path = './img_annotation_map.json'
        self.img_folder = './images/'
        self.annotation_folder = './annotations/'
        self.output_folder = './new annotations/'
        self.original_label = './original labels/'
        self.code_category_map = {}
        self.category_code_map = {}
        #read img_annotation_map
        with open(self.img_annotation_map_path) as f:
            self.img_annotation_map = json.load(f)

    def unifying(self) -> None:
        for key, value in self.img_annotation_map.items():
            # copy image to the new folder
            # add suffix
            key = key + '.jpg'
            new_label = key[:-4] + '_label.json'
            shutil.copyfile(os.path.join(self.img_folder, key), os.path.join(self.output_folder, 'images', key))
            anns = {'annotations': {}, 'width': 0, 'height': 0, 'source': ""}
            for platform_name, annotation_names in value.items():
                # copy workerinfo
                for annotation_name in annotation_names:
                    label_name = annotation_name.split('_')[0] + '_label'
                    worker_id = annotation_name.split('_')[1] + '.json'
                    if os.path.exists(os.path.join(self.annotation_folder, platform_name, 'workerinfo', worker_id)):
                        shutil.copyfile(os.path.join(self.annotation_folder, platform_name, 'workerinfo', worker_id), 
                    os.path.join(self.output_folder, 'workerinfo', worker_id))
                    with open(os.path.join(self.annotation_folder, platform_name, 'labels', annotation_name)) as f, \
                    open(os.path.join(self.original_label, label_name)) as f1:
                        ori_ann = json.load(f)
                        ori_label = [json.loads(i.replace('\'', '\"')) for i in f1.readlines()]
                        # we only choose default annotations
                        source = ori_ann['source']
                        anns['source'] = source
                        cnt = 0
                        for object, ann in ori_ann['defaultAnnotation'].items():
                            if not ann['ifNoPrivacy'] and object not in anns.keys():
                                #find all corresponding bboxes
                                for label in ori_label:
                                    if label['category'] == object:
                                        anns['annotations'][str(cnt)] = {'category': object, 'bbox': []}
                                        anns['annotations'][str(cnt)]['bbox'] = label['bbox']
                                        anns['width'] = label['width']
                                        anns['height'] = label['height']
                                        cnt += 1
            #print(anns)
            with open(os.path.join(self.output_folder, 'annotations', new_label), 'w') as w:
                w.write(json.dumps(anns))
    
    def generate_task_json(self)->None:
        img_list = os.listdir(os.path.join(self.output_folder, 'images'))
        name_list = [img[:-4] for img in img_list]
        #name_list = np.array(name_list)
        np.random.seed(0)
        np.random.shuffle(name_list)
        task_per_worker = 10
        task_record = dict()
        for i in range(int(len(name_list) / task_per_worker) + 1):
            start = i*task_per_worker
            end = (i+1)*task_per_worker if (i+1)*task_per_worker < len(name_list) else len(name_list)
            task_record[str(i)] = {'taskId': i, 'taskList': name_list[start:end],
            'progress': 0, 'assigned': 0}
        print(task_record)
        with open('task_record.json', 'w') as w:
            w.write(json.dumps(task_record))
if __name__ == '__main__':
    unifier = unify_annotation()
    #unifier.unifying()
    unifier.generate_task_json()