import requests
import os
from PIL import Image

import pandas as pd
import numpy as np

# Query URL
BASE_URL = 'http://155.138.233.78:8000/api/'
# BASE_URL = 'http://localhost:8000/api/'
USER = 'pjohnson@example.com'
PASSWORD = 'MOlita01!'
LOGIN_ENDPOINT = 'users/login/token/'
TRAINING_ENDPOINT = 'trainings'

CROP_SHAPE = (256, 256)
SCALE = 20
DATASET_PATH = f"finetunning-dataset-{SCALE}-{CROP_SHAPE[0]}x{CROP_SHAPE[1]}"
WITH_ERRORS_PATH = os.path.join(DATASET_PATH, "with_errors")
WITHOUT_ERRORS_PATH = os.path.join(DATASET_PATH, "without_errors")


def authenticate():
    url = (f'{BASE_URL}{LOGIN_ENDPOINT}')
    res = requests.post(url,
                        headers={'accept': 'application/json',
                                 'Content-Type': 'application/x-www-form-urlencoded'},
                        data={'username': USER, 'password': PASSWORD})
    access_token = res.json()['access_token']
    headers = {'Authorization': f'Bearer {access_token}',
               'accept': 'application/json'}
    return headers


def get_trainings_info(scale):
    headers = authenticate()
    url = (f'{BASE_URL}{TRAINING_ENDPOINT}')
    res = requests.get(url, headers=headers, params={
                       "scale": scale, "show_discarded": False, "get_findings": True})
    trainings = res.json()
    return trainings


# def download_image(url, name):
#     page = requests.get(url)
#     f_ext = os.path.splitext(url)[-1]
#     with open(name, 'wb') as f:
#         f.write(page.content)


def preprocess_data(trainings, crop_size):
    findings = []
    num_trainings = len(trainings)
    train = 1
    for training_map in trainings:
        print(f"Downloading training {train} of {num_trainings}")
        train += 1
        for layer_map in training_map['layers']:
            if len(layer_map['findings']) > 0:
                image = {}
                image['without_errors_path'] = os.path.join(
                    WITHOUT_ERRORS_PATH, f"{layer_map['training_id']}-{layer_map['id']}.png")
                for finding_map in layer_map['findings']:
                    if finding_map['status'] != 'discarded':
                        finding = {}
                        finding['id'] = finding_map['id']
                        finding['class'] = finding_map['error_type_id'] - 1
                        finding['top']= finding_map['y']
                        finding['left']= finding_map['x']
                        finding['width']= finding_map['w']
                        finding['height']= finding_map['h']
                        # bounding_box = ((finding_map['x'], finding_map['y']), (
                        #     finding_map['x'] + finding_map['w'], finding_map['y'] + finding_map['h']))
                        # finding['bounding_box'] = bounding_box
                        try:
                            img = Image.open(image['without_errors_path'])
                            img_size = img.size
                            finding['image_height'] = img_size[0]
                            finding['image_width'] = img_size[1]
                            if img_size[0] > crop_size[0] and img_size[1] > crop_size[1]:
                                findings.append(finding)
                        except Exception as e:
                            print(e.errno)

    return findings


def go():
    trainings = get_trainings_info(SCALE)
    findings = preprocess_data(trainings, CROP_SHAPE)
    # arr = np.asarray([ [7,8,9], [5,8,9] ])
    pd.DataFrame(findings).to_csv(
        'sample.csv', index_label="Index", header=['id', 'class', 'top', 'left', 'width', 'height', 'image_width', 'image_height'])
    print(len(findings))


if __name__ == '__main__':
    go()
