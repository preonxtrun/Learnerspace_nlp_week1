import pandas as pd
import numpy as np
data = {
    'Name' : ['Alice', 'Bob', 'Charlie', 'David', 'Eve','Frank','Grace','Heidi','Ivan','Judy'],
    'Subject' : ['Math', 'Science', 'History', 'Math', 'Science', 'History', 'Math', 'Science', 'History', 'Math'],
    'Marks' : np.random.randint(50, 100, size=10),
    'Grade' : np.empty(10)
}
df = pd.DataFrame(data)
df['Grade'] = pd.cut(df['Marks'], bins=[49, 59, 69, 79, 89, 100], labels=['F', 'D', 'C', 'B', 'A'])  
df_=df.sort_values(by=['Marks'],ascending=False)
print(df_)
grouped = df.groupby('Subject')['Marks'].mean()
for subject, mean_marks in grouped.items():
    print(f"Average marks in {subject}: {mean_marks:.2f}")
def pandas_filter_pass(dataframe):
    df2 = dataframe[(dataframe['Grade'] == 'A') | (dataframe['Grade'] == 'B')]
    return df2
print(pandas_filter_pass(df))
