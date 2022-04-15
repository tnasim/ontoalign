import math
import pandas as pd
import yaml
import csv

with open('config.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

SELECTED_DATASET = config['DATASET']
SELECTED_MODEL = config['MODEL']

train_features = pd.read_csv(SELECTED_DATASET + '_train_features.csv')
test_features = pd.read_csv(SELECTED_DATASET + '_test_features.csv')

# Create feature "Type" for training dataset
train_types = []

for row in train_features['Type']:
    if row == 'Class':
        train_types.append(1)
    else:
        train_types.append(0)

train_features['Type_encode'] = train_types

# Create feature "Type" for testing dataset
test_types = []

for row in test_features['Type']:
    if row == 'Class':
        test_types.append(1)
    else:
        test_types.append(0)

test_features['Type_encode'] = test_types

X_train = train_features.loc[:, 'Ngram1_Entity':'Type_encode']
y_train = train_features['Match']

X_test = test_features.loc[:, 'Ngram1_Entity':'Type_encode']
y_test = test_features['Match']

df_train = train_features.loc[:, 'Ngram1_Entity':'Type_encode']
df_train['Match'] = train_features['Match']

df_test = test_features.loc[:, 'Ngram1_Entity':'Type_encode']
df_test['Match'] = test_features['Match']

# Fill nan values with zero
X_train = X_train.fillna(value=0)
X_test = X_test.fillna(value=0)

train = pd.read_csv(SELECTED_DATASET + '_train.csv')
test = pd.read_csv(SELECTED_DATASET + '_test.csv')

TEST_ALIGNMENTS = config[SELECTED_DATASET]['TEST_ALIGNMENTS']

# Train model
# 'GPC', 'SVM_RBF'
MODELS = ['LogisticRegression', 'RandomForest', 'DecisionTreeClassifier', 'SVM', 'LDA', 'SGDClassifier', 'KNN', 'MLPClassifier', 'GaussianNaiveBayes' ]
# MODELS = ['LogisticRegression' ]
# MODELS = ['DecisionTreeClassifier' ]
# MODELS = ['MLPClassifier' ]
# MODELS = ['LDA' ]
# MODELS = ['SGDClassifier' ]
MODELS = ['AdaBoostClassifier' ]
# MODELS = ['KNN' ]
# MODELS = [ 'RandomForest']

HEADER_NAME = {\
    'LogisticRegression': 'LR',\
    'RandomForest': 'RF',\
    'DecisionTreeClassifier': 'DT',\
    'SVM': 'SVM',\
    'LDA': 'LDA',\
    'SGDClassifier': 'SGD',\
    'KNN': 'KNN',\
    'KNN_5': 'KNN(K=5)',\
    'KNN_3': 'KNN(K=3)',\
    'AdaBoostClassifier': 'AdaBoost',\
    'MLPClassifier': 'MLP',\
    'GaussianNaiveBayes': 'NB'\
}
ROWS = []
COLS = [TEST_ALIGNMENTS,]
# x_values = [7.742637] # c values for Logistic Regression
# x_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0] # c values for Logistic Regression
# x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] # random states for MLP
# x_values = [(4, 2), (5, 3), (6, 2), (6, 4), (20, 16), (24, 18), (50, 32), (80, 72), (100, 64), (100, 96), (200, 160), (500, 420)] # Hidden Layer size for MLP
# x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 25, 30, 50] # Decision Tree Classifier for different depth values
# x_values = [0.025, 1.75, 5.125, 15.25, 50.125] # SVM C values
# x_values = [75, 150, 300, 600, 1200, 2400, 4800, 9600, 19200] # SGD Classifier Max Iterations
# x_values = [i for i in range(100, 20000, 100)] # SGD Classifier Max Iterations
# x_values = [0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# x_values = [i for i in range(2, 51)] # Value of K for KNN
# x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] # random states for Ada Boost
x_values = [i for i in range(1, 20)] # number of estimator for Ada Boost
# x_values = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] # number of estimator for Ada Boost

y_values = []
for est in x_values:
    print('est=', est)
    for SELECTED_MODEL in MODELS:
        
        print('Model Selected: ', SELECTED_MODEL)
        print('------------------------------------------------------------------')
        if SELECTED_MODEL != 'XGBoost':
            if SELECTED_MODEL == 'LogisticRegression':
                print("Training logistic regression...")
                from sklearn.linear_model import LogisticRegression

                if SELECTED_DATASET == 'dataset1' or SELECTED_DATASET == 'dataset1Extended':
                    model = LogisticRegression(penalty='l2', C=1.0, class_weight=None)
                elif SELECTED_DATASET == 'dataset2':
                    model = LogisticRegression(penalty='l2', C=1.5,
                                            class_weight=None)
            elif SELECTED_MODEL == 'RandomForest':
                print("Training random forest classifier...")
                from sklearn.ensemble import RandomForestClassifier

                if SELECTED_DATASET == 'dataset1' or SELECTED_DATASET == 'dataset1Extended':
                    model = RandomForestClassifier(n_estimators=500,
                                                max_features='sqrt', max_depth=3,
                                                random_state=42)
                elif SELECTED_DATASET == 'dataset2':
                    model = RandomForestClassifier(n_estimators=100, max_features=None,
                                                max_depth=3)
            
            elif SELECTED_MODEL == 'AdaBoostClassifier':
                print("Training AdaBoost Classifier...")
                from sklearn.ensemble import AdaBoostClassifier
                model = AdaBoostClassifier(n_estimators=est, learning_rate=1.0, random_state=1)

            elif SELECTED_MODEL == 'GPC':
                print("Training Gaussian Process Classifier...")
                from sklearn.gaussian_process.kernels import RBF
                from sklearn.gaussian_process import GaussianProcessClassifier
                model = GaussianProcessClassifier(2.0 * RBF(1.0))
            
            elif SELECTED_MODEL == 'GaussianNaiveBayes':
                print("Training Gaussian Naive Bayes Classifier...")
                from sklearn.naive_bayes import GaussianNB

                model = GaussianNB()

            elif SELECTED_MODEL == 'DecisionTreeClassifier':
                print("Training DecisionTreeClassifier...")
                from sklearn.tree import DecisionTreeClassifier

                model = DecisionTreeClassifier(max_depth=est)

            elif SELECTED_MODEL == 'MLPClassifier':
                print("Training multi-layer perceptron...")
                from sklearn.neural_network import MLPClassifier

                model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 2), random_state=6)
                
            elif SELECTED_MODEL == 'SVM':
                print("Training Support Vector Machine...")
                from sklearn import svm
                # model = svm.SVC(probability=True)
                model = svm.SVC(kernel="linear", C=0.025, probability=True)
            
            elif SELECTED_MODEL == 'SVM_POLY':
                print("Training Support Vector Machine (Polynomial Kernel)...")
                from sklearn import svm
                model = svm.SVC(kernel="poly", degree=3, gamma='auto', C=0.025, probability=True)
            
            elif SELECTED_MODEL == 'SVM_RBF':
                print("Training Support Vector Machine (RBF)...")
                from sklearn import svm
                model = svm.SVC(gamma='auto', C=0.002, probability=True)
            
            elif SELECTED_MODEL == 'SGDClassifier':
                print("Training Stochastic Gradient Descent...")
                from sklearn.linear_model import SGDClassifier

                model = SGDClassifier(loss="log", max_iter=est)
                # model = SGDClassifier(loss="hinge", max_iter=est, penalty='l2')
            
            elif SELECTED_MODEL == 'QDA':
                print("Training QuadraticDiscriminantAnalysis...")
                from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

                model = QuadraticDiscriminantAnalysis()
            
            elif SELECTED_MODEL == 'LDA':
                print("Training LinearDiscriminantAnalysis...")
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

                model = LinearDiscriminantAnalysis(solver='eigen', shrinkage=est)

            elif SELECTED_MODEL == 'KNN_3':
                print('Training K Nearest Neighbours Classifier ...')
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(3)

            elif SELECTED_MODEL == 'KNN_5':
                print('Training K Nearest Neighbours Classifier ...')
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(5)

            elif SELECTED_MODEL == 'KNN':
                print('Training K Nearest Neighbours Classifier ...')
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(est)

            model.fit(X_train, y_train)
            print("Predicting for testing dataset...")
            y_prob = model.predict_proba(X_test)


        elif SELECTED_MODEL == 'XGBoost':
            import xgboost as xgb

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            param = {'silent': 0, 'objective': 'binary:logistic',
                    'min_child_weight': 10, 'gamma': 2.0, 'subsample': 0.8,
                    'colsample_bytree': 0.8, 'max_depth': 5, 'nthread': 4,
                    'eval_metric': 'error'}

            evallist = [(dtest, 'eval'), (dtrain, 'train')]

            plst = param.items()

            num_round = 10
            bst = xgb.train(plst, dtrain, num_round, evallist,
                            verbose_eval=False)

            y_prob = bst.predict(dtest)

        # Choose best threshold
        COL = []
        for alignment in TEST_ALIGNMENTS:
            ont1 = alignment.split('-')[0]
            ont2 = alignment.split('-')[1].replace('.rdf', '')
            best_ts = 0
            best_fmeasure = 0

            for ts in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

                preds = []

                if SELECTED_MODEL != 'XGBoost':
                    for x in y_prob:
                        if x[1] >= ts:
                            preds.append(1)
                        else:
                            preds.append(0)
                else:
                    for x in y_prob:
                        if x >= ts:
                            preds.append(1)
                        else:
                            preds.append(0)

                test['Predict'] = preds

                if SELECTED_DATASET == 'dataset1' or SELECTED_DATASET == 'dataset1Extended':
                    onto_format = 'rdf'
                elif SELECTED_DATASET == 'dataset2':
                    onto_format = 'owl'

                pred_mappings = test[(test[
                                        'Ontology1'] == f"{SELECTED_DATASET}/ontologies/{ont1}.{onto_format}") &
                                    (test[
                                        'Ontology2'] == f"{SELECTED_DATASET}/ontologies/{ont2}.{onto_format}") &
                                    (test['Predict'] == 1)]

                true_mappings = test[(test[
                                        'Ontology1'] == f"{SELECTED_DATASET}/ontologies/{ont1}.{onto_format}") &
                                    (test[
                                        'Ontology2'] == f"{SELECTED_DATASET}/ontologies/{ont2}.{onto_format}") &
                                    (test['Match'] == 1)]

                correct_mappings = test[
                    (test[
                        'Ontology1'] == f"{SELECTED_DATASET}/ontologies/{ont1}.{onto_format}") &
                    (test[
                        'Ontology2'] == f"{SELECTED_DATASET}/ontologies/{ont2}.{onto_format}") &
                    (test['Match'] == 1) & (test['Predict'] == 1)]

                true_num = len(true_mappings)
                predict_num = len(pred_mappings)
                correct_num = len(correct_mappings)

                if predict_num != 0 and true_num != 0 and correct_num != 0:
                    precision = correct_num / predict_num
                    recall = correct_num / true_num
                    fmeasure = 2 * precision * recall / (precision + recall)
                else:
                    fmeasure = 0

                if fmeasure > best_fmeasure:
                    best_fmeasure = fmeasure
                    best_ts = ts
                    best_preds = preds
            
            COL.append(best_fmeasure)
            print(
                f"Best fmeasure for {alignment} is {best_fmeasure} with threshold {best_ts}")
        
        COLS.append(COL)
        print('------------------------------------------------------------------')

    c = len(COLS)
    r = len(COLS[0])
    ROW_AVG = [ sum(col) for col in COLS[1:]]
    for i in range(r):
        ROW = []
        for j in range(c):
            if(j>0):
                ROW.append(round(float(COLS[j][i]), 2)) # round upto 2 decimal points
            else:
                ROW.append(COLS[j][i].split('.')[0]) # the alignment file name
        ROWS.append(ROW)

    ROW_AVG = [round(a/r, 2) for a in ROW_AVG]
    FOOTER = ['Average']
    FOOTER.extend(ROW_AVG)
    print(ROW_AVG)
    y_values.append(ROW_AVG[0])

print(x_values)
print(ROW_AVG)
from create_plots import createLinePlot
from create_plots import createBarChartSingle
#ßcreateLinePlot(x_values, y_values, "Random Forest for Different Depths", "depth", "F-Measure")
# createLinePlot(x_values, ROW_AVG, "Random Forest for Different Number of Estimators", "No. of Estimator", "F-Measure")
# createLinePlot(x_values, ROW_AVG, "Logistic Regression for Different values of C", "C", "F-Measure")
# createLinePlot(x_values, ROW_AVG, "Decision Tree Classifier for different depth values", "depth", "F-Measure")
# createLinePlot(x_values, ROW_AVG, "MLP Classifier for different Random States (Dataset 1)", "No. of Random States", "F-Measure")
# createBarChartSingle(x_values, ROW_AVG, "No. of Hidden Layers", "F-Measure", "MLP Classifier for different hidden layers (Dataset 1)")
# createLinePlot(x_values, ROW_AVG, "SVM for different C-Values", "C", "F-Measure")
# createLinePlot(x_values, ROW_AVG, "SGD Classifier for Different No. of Iterations (loss='hinge', penalty='l2')", "No. of Iterations", "F-Measure")
# createLinePlot(x_values, ROW_AVG, "KNN Classifier for Different values of K", "K", "F-Measure")
# createLinePlot(x_values, ROW_AVG, "LDA Classifier for 'eigen' solver with different shrinkage values", "shrinkage", "F-Measure")
# createLinePlot(x_values, ROW_AVG, "Ada Boost Classifier for different Random States (Dataset 1)", "No. of Random States", "F-Measure")
# createLinePlot(x_values, ROW_AVG, "Ada Boost Classifier for different No. of Estimators (Dataset 1)", "No. of Estimators", "F-Measure")
createLinePlot(x_values, ROW_AVG, "Ada Boost Classifier for different No. of Estimators (Dataset 2)", "No. of Estimators", "F-Measure")

# HEADER = ['Alignments']
# for m in MODELS:
#     HEADER.append(HEADER_NAME[m])

# # name of csv file 
# filename = "ML-ALL-RESULTS-" + SELECTED_DATASET + ".csv"
# ROWS.append(FOOTER)

# writing to csv file 
# with open(filename, 'w') as csvfile: 
#     # creating a csv writer object 
#     csvwriter = csv.writer(csvfile) 
        
#     # writing the fields 
#     csvwriter.writerow(HEADER) 
        
#     # writing the data rows 
#     csvwriter.writerows(ROWS)

# for h in HEADER:
#     print(h, endl='')
# print('')
# for r in ROWS:
#     for c in r:
#         print(c, endl='')
#     print('')



