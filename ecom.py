import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
s=pd.read_csv('Ecommerce Customers')
df=pd.DataFrame(s)
# print(df.columns)
# t=s.corr()
# sns.heatmap(( s.isnull()),yticklabels="false",annot= True )#it will show weather null values are their or not
# plt.show()
#x will represnt input and y will represent label
x=df[[ 'Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y=df['Yearly Amount Spent']

x_trained, x_test , y_trained, y_test = train_test_split(x,y,test_size=.4,random_state=101)
# print(x_test)
lm = LinearRegression()#object created in linear regression
lm.fit(x_trained,y_trained)#method calling
pp=(lm.coef_)#it will give the corerelation of yearly amount spent with every other collumn value

l=pd.DataFrame(pp,index=['Avg. Session Length','Time on App','Time on Website','Length of Membership'],columns=["coef"])
print(l)