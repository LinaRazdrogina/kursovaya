import glob
from utils import make_set, standartize
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':

    # подготовка тестовой и обучающей выборок
    X_train, y_train = make_set(glob.glob('training_files/*.wav'))
    X_test, y_test = make_set(glob.glob('test_files/*.wav'))

    # стандартизация признаков X_train и X_test
    X_train = standartize(X_train)
    X_test = standartize(X_test)

    # построение классификатора kNN
    clf = KNeighborsClassifier(n_neighbors=3, p=1)
    clf.fit(X_train, y_train.ravel())

    # тестирование точности классификатора
    print('Classification accuracy:', clf.score(X_test, y_test))
