import time
import warnings
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

def get_new_feature_name_df(old_feature_name_df):
    '''
    Pandas 데이터프레임 내 중복이름 컬럼의 허용이 금지되어 이를 해결하기 위해 컬럼명을 변경하는 함수
    :param old_feature_name_df: 기존 데이터프레임
    :return: 컬럼명 수정 데이터프레임
    '''
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(),
                                  columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1])
    if x[1] >0 else x[0] ,  axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df

def get_human_dataset( ):
    feature_name_df = pd.read_csv('Dataset/human_activity/features.txt',sep='\s+',
                                  header=None,names=['column_index','column_name'])

    # 중복된 피처명을 수정
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()

    X_train = pd.read_csv('Dataset/human_activity/train/X_train.txt',sep='\s+', names=feature_name )
    X_test = pd.read_csv('Dataset/human_activity/test/X_test.txt',sep='\s+', names=feature_name)

    y_train = pd.read_csv('Dataset/human_activity/train/y_train.txt',sep='\s+',header=None,names=['action'])
    y_test = pd.read_csv('Dataset/human_activity/test/y_test.txt',sep='\s+',header=None,names=['action'])

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_human_dataset()

# GBM 수행시간 측정
start_time = time.time()

gbm_clf = GradientBoostingClassifier(random_state=0)
gbm_clf.fit(X_train, y_train)
y_pred = gbm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("GBM 정확도: {0:.4f}".format(accuracy))
print("GBM 수행시간: {0:.1f}초".format(time.time() - start_time))

# 하이퍼파라미터 최적화
params = {
    "n_estimators": [100, 500],
    "learning_rate": [0.05, 0.1]
}

grid_cv = GridSearchCV(estimator=gbm_clf, param_grid=params, cv=2, verbose=1)
grid_cv.fit(X_train, y_train)

print("최적 하이퍼파라미터: \n", grid_cv.best_params_)
print("최고 예측 정확도: {0: .4f}".format(grid_cv.best_score_))

# GridSearchCV 예측 결과 적용한 실제 예측 수행
y_pred_optim = grid_cv.best_estimator_.predict(X_test)
accuracy_optim = accuracy_score(y_test, y_pred_optim)

print("GBM 정확도: {0:.4f}".format(accuracy_optim))