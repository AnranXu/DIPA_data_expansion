import os
import json

#check if an image contain any privacy content
def check_privacy_content(image_name):
    ifPrivacy = False
    img_annotation_map_path = './img_annotation_map.json'
    img_annotation_map = {}
    valid_workers = []
    with open(img_annotation_map_path) as f:
        img_annotation_map = json.load(f)
    with open('./annotations/CrowdWorks/valid_workers.json') as f:
        valid_workers = json.load(f)
    with open('./annotations/Prolific/valid_workers.json') as f:
        valid_workers += json.load(f)
    for platform, annotations in img_annotation_map[image_name].items():
        for i, annotation in enumerate(annotations):
            if i >= 2:
                continue
            image_id = annotation.split('_')[0]
            prefix_len = len(image_id) + 1
            worker_file = annotation[prefix_len:]
            worker_id = worker_file[:-11]
            worker_file = worker_id + '.json'
            if worker_id not in valid_workers:
                continue
            with open(os.path.join(annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label:
                label = json.load(f_label)
                if len(label['manualAnnotation'].keys()):
                    ifPrivacy = True
                    return ifPrivacy
                for key, value in label['defaultAnnotation'].items():
                    if not value['ifNoPrivacy']:
                        ifPrivacy = True
                        return ifPrivacy
    return ifPrivacy    

def get_images_with_four_hit():
    annotation_path = './publish/annotations/'
    img_path = './publish/images/'
    img_annotation_map = {}
    new_img_annotation_map = {}
    img_annotation_map_path = './publish/img_annotation_map.json'
    with open(img_annotation_map_path) as f:
        img_annotation_map = json.load(f)
    for image_name in img_annotation_map.keys():
        privacy_num = 0
        for platform, annotations in img_annotation_map[image_name].items():
                for i, annotation in enumerate(annotations):
                    #read annotation into label
                    image_id = annotation.split('_')[0]
                    prefix_len = len(image_id) + 1
                    worker_file = annotation[prefix_len:]
                    worker_id = worker_file[:-11]
                    worker_file = worker_id + '.json'
                    with open(os.path.join(annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label:
                        label = json.load(f_label)
                    if len(label['manualAnnotation'].keys()):
                        privacy_num += 1
                        continue
                    for key, value in label['defaultAnnotation'].items():
                        if not value['ifNoPrivacy']:
                            privacy_num += 1
                            break
        if privacy_num == 4:
            # copy images to folder privacy-threatening images
            image_file = os.path.join(img_path, image_name + '.jpg')
            output_path = os.path.join('privacy-threateningImages', 'images')
            if os.path.exists(image_file):
                os.system('cp {} {}'.format(image_file, output_path))
            # copy labels, worker to folder privacy-threatening images
            # add to new_img_annotation_map
            if image_name not in new_img_annotation_map.keys():
                new_img_annotation_map[image_name] = {}
            for platform, annotations in img_annotation_map[image_name].items():
                if platform not in new_img_annotation_map[image_name].keys():
                    new_img_annotation_map[image_name][platform] = []
                for i, annotation in enumerate(annotations):
                    #read annotation into label
                    image_id = annotation.split('_')[0]
                    prefix_len = len(image_id) + 1
                    worker_file = annotation[prefix_len:]
                    worker_id = worker_file[:-11]
                    worker_file = worker_id + '.json'
                    new_img_annotation_map[image_name][platform].append(annotation)
                    with open(os.path.join(annotation_path, platform, 'workerinfo', worker_file), encoding="utf-8") as f_worker, \
                    open(os.path.join(annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label:
                        worker = json.load(f_worker)
                        label = json.load(f_label)
                        # copy labels, worker to folder publish
                        new_annotation = image_id + '_' + worker_id + '_label.json'
                        new_workerfile = worker_id + '.json'
                        with open(os.path.join('privacy-threateningImages', platform, 'labels', new_annotation), 'w') as f:
                            json.dump(label, f)
                        with open(os.path.join('privacy-threateningImages', platform, 'workerinfo', new_workerfile), 'w') as f:
                            json.dump(worker, f)

    # save new_img_annotation_map
    with open('./privacy-threateningImages/img_annotation_map.json', 'w') as f:
        json.dump(new_img_annotation_map, f)
    
if __name__ == '__main__':
    annotation_path = './annotations/'
    img_path = './images/'
    img_annotation_map = {}
    img_annotation_map_path = './img_annotation_map.json'
    new_img_annotation_map = {}
    valid_workers = []
    with open(img_annotation_map_path) as f:
        img_annotation_map = json.load(f)
        #new_img_annotation_map = img_annotation_map.copy()
    with open('./annotations/CrowdWorks/valid_workers.json') as f:
        valid_workers = json.load(f)
    with open('./annotations/Prolific/valid_workers.json') as f:
        valid_workers += json.load(f)
    # map all valid workers to a unique workerid (from 0 to n)
    workerid_map = {}
    workerid = 0
    for worker in valid_workers:
        workerid_map[worker] = 'annotator' + str(workerid)
        workerid += 1
    print(len(img_annotation_map.keys()))
    for image_name in img_annotation_map.keys():
            if not check_privacy_content(image_name):
                continue
            for platform, annotations in img_annotation_map[image_name].items():
                for i, annotation in enumerate(annotations):
                    if i >= 2:
                        #delete the annotation in new_img_annotation_map
                        #new_img_annotation_map[image_name][platform].remove(annotation)
                        continue
                    image_id = annotation.split('_')[0]
                    prefix_len = len(image_id) + 1
                    worker_file = annotation[prefix_len:]
                    worker_id = worker_file[:-11]
                    worker_file = worker_id + '.json'
                    if worker_id not in valid_workers:
                        continue
                    if image_id not in new_img_annotation_map.keys():
                        new_img_annotation_map[image_id] = {}
                    if platform not in new_img_annotation_map[image_id].keys():
                        new_img_annotation_map[image_id][platform] = []
                    new_workerid = workerid_map[worker_id]
                    new_annotation = image_id + '_' + workerid_map[worker_id] + '_label.json'
                    new_img_annotation_map[image_id][platform].append(new_annotation)
                    with open(os.path.join(annotation_path, platform, 'workerinfo', worker_file), encoding="utf-8") as f_worker, \
                    open(os.path.join(annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label:
                        worker = json.load(f_worker)
                        label = json.load(f_label)
                        # remove key 'frequency' in worker
                        if platform == 'CrowdWorks':
                             worker['nationality'] = 'Japan'
                        else:
                             worker['nationality'] = 'U.K.'
                        worker['workerId'] = new_workerid
                        label['workerId'] = new_workerid
                        # copy labels, worker to folder publish
                        new_annotation = image_id + '_' + new_workerid + '_label.json'
                        new_workerfile = new_workerid + '.json'
                        with open(os.path.join('publish', 'annotations', platform, 'labels', new_annotation), 'w') as f:
                            json.dump(label, f)
                        with open(os.path.join('publish', 'annotations', platform, 'workerinfo', new_workerfile), 'w') as f:
                            json.dump(worker, f)
                        
                        # copy images to folder publish
                        image_file = os.path.join(img_path, image_name + '.jpg')
                        if os.path.exists(image_file):
                            os.system('cp {} {}'.format(image_file, os.path.join('publish', 'images', image_name + '.jpg')))
    # save new_img_annotation_map
    print('length:', len(new_img_annotation_map.keys()))
    with open('./publish/img_annotation_map.json', 'w') as f:
        json.dump(new_img_annotation_map, f)

    # get images with four hit
    get_images_with_four_hit()