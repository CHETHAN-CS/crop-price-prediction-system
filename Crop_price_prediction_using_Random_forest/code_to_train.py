import xlrd

import numpy as np
import seaborn
import numpy as np
import matplotlib.pyplot as matplotlib
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from matplotlib.lines import Line2D
from scipy.stats import pearsonr
import cv2

# set seed to make results reproducible
RF_SEED = 30
def load_input(excel_file):
    y_prediction = []
    data = []
    feature_names = []

    loc = (excel_file)
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)
    for index_row in range(0, 81):
        row = sheet.row_values(index_row)
        row = row[1:]
        if index_row == 0:
            feature_names = row
        else:
            row[0] = str(row[0]).split(".")[0]
            data.append([float(x) for x in row[:-1]])
            y_prediction.append(float(row[-1]))
    return y_prediction, data, feature_names[:-1]

def split_data_train_model(labels, data):# 20% examples in test data
    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=RF_SEED)
    # training data fit
    regressor = RandomForestRegressor(n_estimators=1000, random_state=RF_SEED)
    regressor.fit(x_data, y_data)
    filename = 'rf_model.sav'
    joblib.dump(regressor, filename)
    return test, test_labels, regressor

def simple_scatter_plot(x_data, y_data, output_filename, title_name, x_axis_label, y_axis_label):
    """Simple scatter plot.

    Args:
        x_data (list): List with x-axis data.
        y_data (list): List with y-axis data.
        output_filename (str): Path to output image in PNG format.
        title_name (int): Plot title.
        x_axis_label (str): X-axis Label.
        y_axis_label (str): Y-axis Label.

    """
    seaborn.set(color_codes=True)
    matplotlib.figure(1, figsize=(9, 6))

    matplotlib.title(title_name)

    ax = seaborn.scatterplot(x=x_data, y=y_data)

    ax.set(xlabel=x_axis_label, ylabel=y_axis_label)

    matplotlib.savefig(output_filename, bbox_inches='tight', dpi=300)
    matplotlib.close()

y_data, x_data, feature_names = load_input("regression_dataset.xlsx")
x_test, x_test_labels, regressor = split_data_train_model(y_data, x_data)


predictions = regressor.predict(x_test)


#code to test user data as input for testing
user_test_data = np.array([[2,2020.0,30.0], [6,2020.0, 25.0]])
print(user_test_data)
prediction2=regressor.predict(user_test_data)
print(prediction2)

# find the correlation between real answer and prediction
correlation = round(pearsonr(predictions, x_test_labels)[0], 5)

output_filename = "regression_graph.png"
title_name = "Random Forest Regression - Real crop Price vs Predicted crop Price - correlation ({})".format(correlation)
x_axis_label = "Real Ragi Price"
y_axis_label = "Predicted Ragi Price"

# plot data
simple_scatter_plot(x_test_labels, predictions, output_filename, title_name, x_axis_label, y_axis_label)
#matplotlib.style.use('seaborn-whitegrid')
#matplotlib.plot(x_axis_label, y_axis_label)
#
features_importance = regressor.feature_importances_

print("Feature ranking:")
for i, data_class in enumerate(feature_names):
    print("{}. {} ({})".format(i + 1, data_class, features_importance[i]))

img = cv2. imread("regression_graph.png")
img2=cv2.resize(img, (800, 600))
cv2.imshow("Result", img2)
cv2.waitKey(0)
# load the model from
filename = 'rf_model.sav'
loaded_model = joblib.load(filename)
results = loaded_model.score(x_data, y_data)
print(results)

#closing all open windows
cv2.destroyAllWindows()
