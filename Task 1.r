import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

df=sns.load_dataset("iris")

print("shape of dataset", df.shape)
print("column names", df.columns)

df.head()
df.info()
df.describe()

plt.figure(figsize=(6,4))
sns.scatterplot(
    data=df,
    x="sepal_length",
    y="sepal_width",
    hue="species"

)

plt.title("Scatter Plot: Sepal Length vs Sepal Width")
plt.show()

df.hist(figsize=(10,8))
plt.suptitle("Histrogram of Iris Features")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.title("Box Plot of Iris Features")
plt.show()
