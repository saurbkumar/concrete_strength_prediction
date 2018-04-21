import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
data = pandas.read_excel("data/Concrete_Data.xls")

# Input Variable
inpt_colmn = ['Cement (component 1)(kg in a m^3 mixture)',
         'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
         'Fly Ash (component 3)(kg in a m^3 mixture)',
         'Water  (component 4)(kg in a m^3 mixture)',
         'Superplasticizer (component 5)(kg in a m^3 mixture)',
         'Coarse Aggregate  (component 6)(kg in a m^3 mixture)'
         ]
input_var = data[inpt_colmn]
outpt_colm = ['Fine Aggregate (component 7)(kg in a m^3 mixture)',
              'Age (day)',
              'Concrete compressive strength(MPa, megapascals) ']
output_var = data[outpt_colm]
# Histogram plot for features
input_var.plot(kind='hist',subplots=True, layout=(4,3), figsize=(15,15),sharex=False,alpha=.5,grid = True,
                 title="Histogram Plot for Concrete features")
plt.show()

output_var.plot(kind='hist',subplots=True, layout=(4,3), figsize=(15,15),sharex=False,alpha=.5,grid = True,
                 title="Histogram Plot for Concrete features")
plt.show()

correlations = input_var.corr()

plt.figure(figsize=(14,14))
plt.title("Correlation heatmap for Patient features")
temp = sns.heatmap(correlations, cbar = True,  square = True, annot=True, 
            fmt= '.2f',annot_kws={'size': 12}, cmap= 'coolwarm') 
plt.show()
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
regr = linear_model.LinearRegression()

ml_regr = MLPRegressor()
X_train, X_test, y_train, y_test = train_test_split(input_var.get_values(), output_var['Concrete compressive strength(MPa, megapascals) '].get_values(), test_size=0.40, random_state=42)

ml_regr.fit(X_train,y_train)
y_predt = ml_regr.predict(X_test)
mean_squared_error()