import itertools

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from owlready2 import get_ontology
from sklearn.metrics import f1_score


def read_ontology(path):
    onto = get_ontology(path)
    onto.load()

    # Read classes
    classes = []

    for cl in onto.classes():
        classes.append(cl)

    classes = list(set(classes))

    # Read properties
    properties = []

    for prop in onto.properties():
        properties.append(prop)

    properties = list(set(properties))

    return classes, properties


def get_mappings(filename, dataset=''):
    mappings = []

    with open(filename) as f:
        soup = BeautifulSoup(f, 'xml')

    cells = soup.find_all('Cell')

    for cell in cells:
        if(dataset=='biodiv' or dataset == 'phenotype'):
            entity1 = cell.find('entity1').attrs['rdf:resource'].split('/')[-1]
            entity2 = cell.find('entity2').attrs['rdf:resource'].split('/')[-1]
        else:
            entity1 = cell.find('entity1').attrs['rdf:resource'].split('#')[1]
            entity2 = cell.find('entity2').attrs['rdf:resource'].split('#')[1]
        mappings.append((entity1, entity2))

    return mappings


def get_dataset(ont1_path, ont2_path, alignment_path, dataset=''):
    data = []

    mappings = get_mappings(alignment_path, dataset)
    mappings = [tuple(x) for x in mappings]

    all_mappings = []

    # Parse ontologies
    classes1, properties1 = read_ontology(ont1_path)
    classes2, properties2 = read_ontology(ont2_path)

    # Generate pairs of classes
    # class_pairs = list(itertools.product(classes1, classes2))
    # print('Total class_pairs: ', len(class_pairs))
    # for class_pair in class_pairs:
    for class1 in classes1:
        for class2 in classes2:
            class_pair = (class1, class2)
            # pair = (class_pair[0].name, class_pair[1].name)
            pair = (class1.name, class2.name)
            # print(pair)
            if pair in mappings:
                match = 1
                # all_mappings.append(pair)
                mappings.remove(pair)
            else:
                match = 0

            data.append((ont1_path, ont2_path, pair[0], pair[1],
                        class_pair[0].is_a[0].name, class_pair[1].is_a[0].name,
                        get_path(class_pair[0]), get_path(class_pair[1]), match,
                        'Class'))

    # Generate pairs of properties
    # properties_pairs = list(itertools.product(properties1, properties2))
    # print('Total properties_pairs: ', len(properties_pairs))
    # for prop_pair in properties_pairs:
    for property1 in properties1:
        for property2 in properties2:
            # pair = (prop_pair[0].name, prop_pair[1].name)
            pair = (property1.name, property2.name)
            # print(pair)
            if pair in mappings:
                match = 1
                # all_mappings.append(pair)
                mappings.remove(pair)
            else:
                match = 0

            data.append((ont1_path, ont2_path, pair[0], pair[1],
                        class_pair[0].is_a[0].name, class_pair[1].is_a[0].name,
                        get_path(class_pair[0]), get_path(class_pair[1]), match,
                        'Property'))

    # print('Readed mappings', len(all_mappings), '\n')

    dataset = pd.DataFrame(data, columns=['Ontology1', 'Ontology2', 'Entity1',
                                          'Entity2', 'Parent1', 'Parent2',
                                          'Path1', 'Path2', 'Match', 'Type'])

    return dataset


def get_path(cl):
    path = cl.name
    while True:
        try:
            path = path + '/' + cl.is_a[0].name
        except IndexError:
            break
        cl = cl.is_a[0]
        if cl == 'owl.Thing':
            break

    return '/'.join(path.split('/')[::-1])


def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1 - f1_score(y_true, np.round(y_pred))
    return 'f1_err', err
