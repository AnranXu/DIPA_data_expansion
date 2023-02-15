import os

if __name__ == '__main__':
    wrong_folder = './new annotations(wrong one)/annotations/'
    true_folder = './new annotations/annotations/'
    annotations = os.listdir(true_folder)
    cnt = 0
    for annotation in annotations:
        with open (os.path.join(wrong_folder, annotation)) as wrong_one, \
            open(os.path.join(true_folder, annotation)) as true_one:
            true = true_one.read()
            wrong = wrong_one.read()
            if true != wrong:
                print(annotation)
                cnt += 1