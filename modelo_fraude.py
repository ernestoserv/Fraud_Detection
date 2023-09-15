from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  f1_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
''' Como primer paso se van a a limpiar el dataset train.csv'''
df = pd.read_csv('/Users/claudiaeenriquezgracia/Documents/Ciencia de datos/DataKnow/train.csv')
df = df.drop(columns=['FECHA_VIN','COD_PAIS','Canal1','id','FECHA'])
df[['Dist_Mean_NAL','Dist_Sum_INTER','Dist_Mean_INTER','Dist_Max_INTER']] = df[['Dist_Mean_NAL','Dist_Sum_INTER','Dist_Mean_INTER','Dist_Max_INTER']].fillna(1,inplace=True)
df2 = pd.get_dummies(df)
df2 = df2.drop(columns='SEGMENTO_Empresarial')
df2 = df2.dropna(axis=0)
X = df2.drop(columns='FRAUDE')
y = df2['FRAUDE'].values
'''A conitnuacion se van a realizar funciones con las cuales vamos a definir el modelo a utilizar y buscar los
mejores hiperparametros'''
def model_performance(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.30,random_state=43,stratify=y)
    models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(),
              "Decision Tree Classifier": DecisionTreeClassifier()}
    results = []
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.fit_transform(X_test)
    for model in models.values():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        kf = KFold(n_splits=5, random_state=43, shuffle=True)

        cv_results = cross_val_score(model, X_train_s, y_train, cv=kf,n_jobs=-1,scoring='f1')
        results.append(cv_results)

    plt.boxplot(results, labels=models.keys())
    plt.show()
    return
def trees_score(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.30, random_state=43)
    tree = DecisionTreeClassifier(random_state=43)
    tree.fit(X_train,y_train)
    y_pred = tree.predict(X_test)
    bc = BaggingClassifier(base_estimator=tree, random_state=43,n_jobs=-1)
    bc.fit(X_train,y_train)
    y_pred2 = bc.predict(X_test)
    rf = RandomForestClassifier(random_state=43)
    rf.fit(X_train,y_train)
    y_pred3 = rf.predict(X_test)
    dt = DecisionTreeClassifier(max_depth=1, random_state=1)
    ada = AdaBoostClassifier(base_estimator=dt ,random_state=43)
    ada.fit(X_train, y_train)
    y_pred4 = ada.predict(X_test)
    print('Tree:',f1_score(y_test, y_pred))
    print('BC:',f1_score(y_test, y_pred2))
    print('RF:', f1_score(y_test, y_pred3))
    print('ADA:', f1_score(y_test, y_pred4))
    print(confusion_matrix(y_test, y_pred3))
    importances = pd.Series(data=rf.feature_importances_,
                            index=X.columns)
    importances_sorted = importances.sort_values()

    importances_sorted.plot(kind='barh', color='lightgreen')
    plt.title('Features Importances')
    plt.show()
    return
def tuning_tree(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.30, random_state=43, stratify=y)
    rf = RandomForestClassifier()
    params_dt = {'n_estimators': [200,300,400,500,600],'max_depth':[10,13,17,20,23],
                 }
    grid_dt = GridSearchCV(estimator=rf,
                           param_grid=params_dt,
                           scoring='f1',
                           cv=4,
                           n_jobs=-1)
    grid_dt.fit(X_train,y_train)
    best_model = grid_dt.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    f1 = classification_report(y_test,y_pred)
    print('RF final:',f1)
    print(grid_dt.best_params_)
    return
'''Aqui limpiaremos el dataset test para poder realizar las predicciones y crear un nuevo documento'''
test_set = pd.read_csv('/Users/claudiaeenriquezgracia/Documents/Ciencia de datos/DataKnow/test.csv')
test_set = test_set.drop(columns=['FECHA_VIN','COD_PAIS','Canal1','id','FECHA','Dist_mean_NAL','Dist_sum_INTER',
                            'Dist_mean_INTER','Dist_max_INTER','FECHA_FRAUDE','Dist_Sum_NAL'])
test_set[['Dist_Mean_NAL','Dist_Sum_INTER','Dist_Mean_INTER','Dist_Max_INTER']]=test_set[['Dist_Mean_NAL','Dist_Sum_INTER','Dist_Mean_INTER','Dist_Max_INTER']].fillna(1,inplace=True)
test_final = pd.get_dummies(test_set)
test_final =test_final.rename(columns={'Dist_max_COL':'Dist_max_NAL'})
X_final = test_final.drop(columns='FRAUDE')
def predict_test(X,y,X_final):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.30, random_state=43, stratify=y)
    rf = RandomForestClassifier(n_estimators=600,max_depth=20)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    print('F1 Score:',classification_report(y_test,y_pred),'\n')
    print('Confusion Matrix:', confusion_matrix(y_test,y_pred))
    y_pred_test = rf.predict(X_final)
    return y_pred_test

#model_performance(X,y)
#trees_score(X,y)
tuning_tree(X,y)
#test_set['FRAUDE'] = predict_test(X,y,X_final)
#test_set.to_csv('/Users/claudiaeenriquezgracia/Documents/Ciencia de datos/DataKnow/test_evaluado.csv',sep=',')