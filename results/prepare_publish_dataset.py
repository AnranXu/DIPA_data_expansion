import os
import json

if __name__ == '__main__':
    annotation_path = './annotations/'
    img_path = './images/'
    img_annotation_map = {}
    img_annotation_map_path = './img_annotation_map.json'
    valid_workers = {}
    with open(img_annotation_map_path) as f:
        img_annotation_map = json.load(f)
    with open('./annotations/CrowdWorks/valid_workers.json') as f:
        valid_workers['CrowdWorks'] = json.load(f)
    with open('./annotations/Prolific/valid_workers.json') as f:
        valid_workers['Prolific'] = json.load(f)
    for image_name in img_annotation_map.keys():
            annotator_num = 1
            for platform, annotations in img_annotation_map[image_name].items():
                for i, annotation in enumerate(annotations):
                    if i >= 2:
                        break
                    image_id = annotation.split('_')[0]
                    prefix_len = len(image_id) + 1
                    worker_file = annotation[prefix_len:]
                    worker_id = worker_file[:-11]
                    worker_file = worker_id + '.json'
                    if worker_id not in valid_workers[platform]:
                        continue
                    with open(os.path.join(annotation_path, platform, 'workerinfo', worker_file), encoding="utf-8") as f_worker, \
                    open(os.path.join(annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label:
                        worker = json.load(f_worker)
                        label = json.load(f_label)
                        # remove key 'frequency' in worker
                        worker.pop('frequency')
                        if platform == 'CrowdWorks':
                             worker['nationality'] = 'Japan'
                        else:
                             worker['nationality'] = 'U.K.'
                        # copy labels, worker to folder publish
                        with open(os.path.join('publish', platform, 'labels', annotation), 'w') as f:
                            json.dump(label, f)
                        with open(os.path.join('publish', platform, 'workerinfo', worker_file), 'w') as f:
                            json.dump(worker, f)

                        # copy images to folder publish
                        image_file = os.path.join(img_path, image_name + '.jpg')
                        if os.path.exists(image_file):
                            os.system('cp {} {}'.format(image_file, os.path.join('publish', 'images', image_name + '.jpg')))
