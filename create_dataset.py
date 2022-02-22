import random
import pandas as pd
import yaml

from utils_datasets import get_dataset

with open('config.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

SELECTED_DATASET = config['DATASET']

print(SELECTED_DATASET, "is selected")

PATH_TO_ONTOLOGIES = config[SELECTED_DATASET]['PATH_TO_ONTOLOGIES']
ONTOLOGIES = config[SELECTED_DATASET]['ONTOLOGIES']
PATH_TO_ALIGNMENTS = config[SELECTED_DATASET]['PATH_TO_ALIGNMENTS']
TRAIN_ALIGNMENTS = config[SELECTED_DATASET]['TRAIN_ALIGNMENTS']
TEST_ALIGNMENTS = config[SELECTED_DATASET]['TEST_ALIGNMENTS']

datasets = []

ONTO_FILE_NAMES_MAPPING = {}
for fname in ONTOLOGIES:
    onto_name = fname.split('.')[0]
    ONTO_FILE_NAMES_MAPPING[onto_name] = fname

# Create dataset

print("Creating training dataset...")

for align_name in TRAIN_ALIGNMENTS:
    print('Read from', align_name)
    ont1, ont2 = align_name.split('.')[0].split('-')

    # All ontologies with 101 in name have RDF format, other have OWL format
    ont1_path = PATH_TO_ONTOLOGIES + ONTO_FILE_NAMES_MAPPING[ont1]
    ont2_path = PATH_TO_ONTOLOGIES + ONTO_FILE_NAMES_MAPPING[ont2]
    # if '101' in align_name:
    #     ont1_path = PATH_TO_ONTOLOGIES + ont1 + '.rdf'
    #     ont2_path = PATH_TO_ONTOLOGIES + ont2 + '.rdf'
    # else:
    #     ont1_path = PATH_TO_ONTOLOGIES + ont1 + '.owl'
    #     ont2_path = PATH_TO_ONTOLOGIES + ont2 + '.owl'
    alignment_path = PATH_TO_ALIGNMENTS + align_name

    if(SELECTED_DATASET == 'biodiv' or SELECTED_DATASET == 'phenotype'):
        datasets.append(get_dataset(ont1_path, ont2_path, alignment_path, SELECTED_DATASET))
    else:
        datasets.append(get_dataset(ont1_path, ont2_path, alignment_path))

random.shuffle(datasets)
train = pd.concat(datasets, ignore_index=True)

print("Training dataset is created")
print("Creating testing dataset...")

datasets = []

for align_name in TEST_ALIGNMENTS:
    print('Read from', align_name)
    ont1, ont2 = align_name.split('.')[0].split('-')
    ont1_path = PATH_TO_ONTOLOGIES + ONTO_FILE_NAMES_MAPPING[ont1]
    ont2_path = PATH_TO_ONTOLOGIES + ONTO_FILE_NAMES_MAPPING[ont2]
    # if '101' in align_name:
    #     ont1_path = PATH_TO_ONTOLOGIES + ont1 + '.rdf'
    #     ont2_path = PATH_TO_ONTOLOGIES + ont2 + '.rdf'
    # else:
    #     ont1_path = PATH_TO_ONTOLOGIES + ont1 + '.owl'
    #     ont2_path = PATH_TO_ONTOLOGIES + ont2 + '.owl'
    alignment_path = PATH_TO_ALIGNMENTS + align_name

    if(SELECTED_DATASET == 'biodiv' or SELECTED_DATASET == 'phenotype'):
        datasets.append(get_dataset(ont1_path, ont2_path, alignment_path, SELECTED_DATASET))
    else:
        datasets.append(get_dataset(ont1_path, ont2_path, alignment_path))

random.shuffle(datasets)
test = pd.concat(datasets, ignore_index=True)

print("Testing dataset is created")

train.to_csv(SELECTED_DATASET + '_train.csv', index=False)
test.to_csv(SELECTED_DATASET + '_test.csv', index=False)

print('Training dataset is saved into ' + SELECTED_DATASET + '_train.csv')
print('Testing dataset is saved into ' + SELECTED_DATASET + '_test.csv')
