# Handwritten letter classification example using SciKit-Learn (Python 3)
Note that the dataset used here is preprocessed.

Given Dataset: The already extracted features:
1. lettr capital letter (26 values from A to Z)
2. x-box horizontal position of box (integer)
3. y-box vertical position of box (integer)
4. width width of box (integer)
5. high height of box (integer)
6. onpix total # on pixels (integer)
7. x-bar mean x of on pixels in box (integer)
8. y-bar mean y of on pixels in box (integer)
9. x2bar mean x variance (integer)
10. y2bar mean y variance (integer)
11. xybar mean x y correlation (integer)
12. x2ybr mean of x * x * y (integer)
13. xy2br mean of x * y * y (integer)
14. x-ege mean edge count left to right (integer)
15. xegvy correlation of x-ege with y (integer)
16. y-ege mean edge count bottom to top (integer)
17. yegvx correlation of y-ege with x (integer)

Testing several algorithms in SciKit-Learn to see which is best for this case.

**Usage:** python main.py [classifier=<your_choice>] [--normalized_matrix]
Default classifier is KNN.
Classifier choices:
1. classifier=knn
2. classifier=naïve_bayes or nb
3. classifier=mlp or neural_networks
4. classifier=svc or svm
5. classifier=decision_tree or dt
6. classifier=random_forest or rf
7. classifier=adaboost or ab
8. classifier=qda
9. classifier=lda
Add the "--normalized_matrix" option to get a normalized confusion matrix.
