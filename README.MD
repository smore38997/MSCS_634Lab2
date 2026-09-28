
MSC634 Advanced Big Data and Data mining Lab 2

By Shreyas More

Install

pip install numpy pandas matplotlib scikit-learn jupyter

jupyter ShreyasMoreMSCS634Lab2.ipynb



Purpose

This lab compares two distance-based classifiers from scikit-learn on the Wine dataset (178 samples, 13 chemical features, 3 wine classes):

K-Nearest Neighbors (KNN): each prediction is a vote among a fixed number of nearest training points (k = 1, 5, 11, 15, 21).
Radius Neighbors (RNN): each prediction is a vote among all training points within a fixed distance (radius = 350, 400, 450, 500, 550, 600).

The goal is to see how each hyperparameter affects test accuracy, and to judge when each method is the better choice.



Method: explore the data, split it 80% train / 20% test (142 / 36 samples), train each model on the training set, record test accuracy for every setting, then plot and compare the results.



Key Insights
KNN outperformed RNN at every setting. Mean accuracy was about 80.0% for KNN and 69.0% for RNN.
KNN is stable. k = 1 was slightly worse because a single neighbor is sensitive to noise. From k = 5 upward, accuracy stayed flat at 80.6%.
RNN gets worse as the radius grows. A larger radius pulls in points from other classes, so the vote drifts toward the majority class. The smallest radius (350) performed best.



Challenges and Decisions
Unscaled features (deliberate): The assignment's radii (350–600) only make sense in raw distance units, so I did not standardize the data. Scaling would require much smaller radii (around 1–5) and would change the results.
Empty neighborhoods in RNN: RadiusNeighborsClassifier fails if a test point has no neighbors in the radius. I set outlier_label='most_frequent' as a safeguard and counted how many test points were affected. The answer was zero at every radius, so it never changed a prediction.