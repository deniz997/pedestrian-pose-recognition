import sklearn.metrics as metrics
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain

from model.data_parser import convert_jaad_dict_to_df, get_data

if __name__ == '__main__':

    data_dir = "C:/Users/max00/Documents/PoseRecognition/pedestrian-pose-recognition/data/JAAD_JSON_Labels/"

    X, Y = convert_jaad_dict_to_df(get_data(data_dir))

    # SVM Classifier

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    sp_X_train = sparse.lil_matrix(X_train.values)
    sp_Y_train = sparse.lil_matrix(Y_train.values)
    sp_X_test = sparse.csr_matrix(X_test.values)
    sp_Y_test = sparse.csr_matrix(Y_test.values)

    classifier = ClassifierChain(
        classifier=RandomForestClassifier(n_estimators=100),
        require_dense=[False, True]
    )

    # train
    classifier.fit(sp_X_train, sp_Y_train)
    # predict
    predictions = classifier.predict(sp_X_test)

    print(metrics.hamming_loss(sp_Y_test, predictions))
    print(metrics.accuracy_score(sp_Y_test, predictions))


    clf = BinaryRelevance(
        classifier=SVC(),
        require_dense=[False, True]
    )

    clf.fit(sp_X_train, sp_Y_train)
    prediction = clf.predict(sp_X_test)


    print(metrics.hamming_loss(sp_Y_test, prediction))
    print(metrics.accuracy_score(sp_Y_test, prediction))

