# Load libraries
from pandas import read_csv,to_numeric
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing


# Load dataset
url = "./data/2019-Nov.csv"
names = ['event_time', 'event_type', 'product_id', 'category_id',
         'category_code', 'brand', 'price', 'user_id','user_session']
dataset = read_csv(url, names=names, skiprows = 1, nrows = 1000)

array = dataset.values
X = array[:,2:9]
y = array[:,4]
x_n, x_v, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=2, shuffle=True)
Y_train = ['empty' if str(x) == 'nan' else x for x in Y_train]
Y_validation = ['empty' if str(x) == 'nan' else x for x in Y_validation]
# print(Y_train)
# print(Y_validation)
le = preprocessing.LabelEncoder()
dataset = dataset.apply(le.fit_transform)
dataset = dataset.fillna(0)


# shape
print(dataset.shape)
# # head
print(dataset.head(20))
# # descriptions
print(dataset.describe())
# # class distribution
print(dataset.groupby('category_code').size())


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(9,1), sharex=False, sharey=False)
pyplot.show()
# histograms
dataset.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


array = dataset.values
X = array[:,2:9]
y = array[:,5]
inverted = le.inverse_transform(y)
X_train, X_validation, yt, yv = train_test_split(X, y, test_size=0.20, random_state=2, shuffle=True)


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))




model = GaussianNB()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print(predictions)
