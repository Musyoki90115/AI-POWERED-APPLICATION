
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#28_day_smp training data
#features:[sleep_hr, water_glasses, bench_kg]
X=np.array([
    [6.5,6,89],[7.2,8,85],[5.8,5,75],[8.0,10,90],
    [7.5,9,88],[6.0,6,78],[7.8,8,86],[8.2,10,92],
    [5.5,4,70],[7.0,7,82],[6.8,8,84],[8.5,11,95],
    [7.3,9,89],[6.2,6,76],[7.9,10,91],[5.9,5,73],
    [8.1,11,93],[7.4,8,87],[6.7,7,83],[8.3,10,94],
    [5.6,4,71],[7.1,8,85],[8.0,9,90],[6.4,6,77],
    [7.6,9,88],[8.4,11,96],[6.3,7,79],[7.7,10,91],

])

#Labels:1= hit 10000 steps, 0=did not
y = np.array([0,1,0,1,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,1,0,1,1,0,1,1,0,1])

#split 80/20
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)

#Train
clf=RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train,y_train)

#Evaluate
y_pred=clf.predict (X_test)
accuracy=accuracy_score(y_test,y_pred)

print(f"Training samples:{len(X_train)}")
print(f"Test samples:{len(X_test)}")
print(f"Accuracy on test:{accuracy:.0%}")

#feature importance
feature=["sleep_hr", "water_glasses", "bench_kg"]
importances=clf.feature_importances_
print()
print("feature importances:")
for name,imp in sorted(zip(feature,importances), key=lambda x: -x[1]):
    bar="#" *int(imp*40)
    print(f"{name:<16}{imp:3f} {bar}")