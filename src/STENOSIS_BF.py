# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import accuracy_score
from merf.utils import MERFDataGenerator
from merf.merf import MERF
from merf.viz import plot_merf_training_stats
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
import matplotlib.pyplot as plt

######## Read data ########
fip_data = "H:/cloud/cloud_data/Projects/MixedEffectModels/data/forMERF.csv"
df_csv = pd.read_csv(fip_data)

# cols_select_mahmoud = ['patient_ID','dataset','study','intercept','slope','intercept1','slope1','study_no','gender',
#  'typical3 ','age','CT_pos','Agatston_score','typical3_imp','atypical2_imp','nongil_imp','iv_prot_imp',
#  'contrast_amount_imp','contrast_conc_imp','height_imp','hypertension_imp','diabetes_imp','hyperlipidemia_imp',
#  'smoker_imp','risk','pos_fam_hist_imp','prior_myo_inf_imp','Study','interaction1',' interaction2','interaction3']

cols_select_features = ['study','gender','age','Agatston_score','typical3_imp','contrast_amount_imp','contrast_conc_imp',
               'height_imp','hypertension_imp','diabetes_imp','hyperlipidemia_imp','smoker_imp','risk',
               'pos_fam_hist_imp','prior_myo_inf_imp']
cols_select_label = ['Cath_pos']
features = df_csv[cols_select_features]
target = df_csv[cols_select_label]

######## Split data ########
data_train, data_test, target_train, target_test = train_test_split(features,target, stratify=df_csv['study'], test_size = 0.20, random_state = 10,)
#data_train = data_train.loc[:, (data_train.columns != 'study')]
#data_test = data_test.loc[:, (data_test.columns != 'study')]
data_train['study'] = pd.factorize(data_train['study'])[0]
data_test['study'] = pd.factorize(data_test['study'])[0]

######## Train RF ########
X_train_rf = np.array(data_train)
X_train_rf = np.array(data_train)
Y_train_rf = np.array(target_train)[:,0]
X_test_rf = np.array(data_test)
X_test_rf = np.array(data_test)
Y_test_rf = np.array(target_test)[:,0]

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_rf, Y_train_rf)
P_pred_rf = rf.predict_proba(X_test_rf)[:, 1]
Y_pred_rf = rf.predict(X_test_rf)

C_RF = confusion_matrix(Y_test_rf, Y_pred_rf)
ACC_RF = accuracy_score(Y_test_rf, Y_pred_rf)


importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
column_names = list(data_train.columns[indices])

for f in range(X_train_rf.shape[1]):
    print("%d. %s (%f)" % (f + 1, column_names[f], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train_rf.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train_rf.shape[1]), indices)
plt.xlim([-1, X_train_rf.shape[1]])
plt.show()

######## Train MERF ########
X_train_merf = np.array(data_train.loc[:, data_train.columns != 'study'])
Y_train_merf = np.array(target_train)[:,0]
Z_train_merf = np.ones((len(X_train_merf),1))
clusters_train = pd.Series(ord_enc.fit_transform(data_train[['study']])[:,0])
X_test_merf = np.array(data_test.loc[:, data_test.columns != 'study'])
Y_test_merf = np.array(target_test)[:,0]
Z_test_merf = np.ones((len(X_test_merf),1))
clusters_test = pd.Series(ord_enc.fit_transform(data_test[['study']])[:,0])

fixed_effects_model=RandomForestRegressor(n_estimators=100, n_jobs=-1)
gll_early_stop_threshold = None
mrf = MERF(fixed_effects_model=fixed_effects_model, max_iterations=50, gll_early_stop_threshold=gll_early_stop_threshold)
mrf.fit(X_train_merf, Z_train_merf, clusters_train, Y_train_merf)
Y_pred_merf = mrf.predict(X_test_merf, Z_test_merf, clusters_test)

C_MERF = confusion_matrix(Y_test_merf, np.round(Y_pred_merf))
ACC_MERF = accuracy_score(Y_test_merf, np.round(Y_pred_merf))


plot_merf_training_stats(mrf, num_clusters_to_plot=10)






