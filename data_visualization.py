import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("extracted_data.csv")
sns.pairplot(df, hue='label', diag_kind='hist')
plt.show()
