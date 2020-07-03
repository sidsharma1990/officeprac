# Pandas (panal data)
# series - 1d data
# dataframe = 2d data
# panal data = 3d and so on

# to update Spyder - conda update spyder
# Series

import pandas as pd
import numpy as np

list1 = [1,2,3,4]
ds1 = pd.Series(list1)
print (ds1)

type(ds1)

ds2 = pd.Series(list1, index = [1,2,3,4])
print (ds2)


list1 = [1,2,3,4]
list2 = [4,5,6,7]

ds3 = pd.Series(list1, dtype = int)
print (ds3)


ds3c = pd.Series(index = list2, data = list1)
print (ds3c)

ds2 + ds3

####### dict to data series
list1 = ['a','b','c','d']
list2 = [4,5,6,7]
dict1 = dict(zip(list1, list2))

ds4 = pd.Series(dict1)
print (ds4)

ds4['c']
ds4[2]

ds4[2:4]
ds4['a':'c']

#### Data Frame
list1 = [1,2,3,4,5]
df1 = pd.DataFrame(list1)
print (df1)

dict2 = {'fruits': ['Mangos', 'apple', 'muskmelon'],
         'counts': [10, 15, 20]}
df2 = pd.DataFrame(dict2)
print (df2)
df2[1:2]
df2[1:3]


######### series to dataframe
series1 = pd.Series([50,10], index = ['a', 'b'])
df3 = pd.DataFrame(series1)
print (df3)

#### numpy to dataframes
import numpy as np
arr1 = np.array([[50,100],['DS', 'ML']])
df4 = pd.DataFrame({'name':arr1[1], 'score': arr1[0]})
print (df4)

###################################
A = [1,2,3,4]
B = [5,6,7,8]
C = [9,1,2,3]
D = [4,5,6,7]
E = [8,9,1,2]

df5 = pd.DataFrame([A,B,C,D,E], ['a','b','c','d','e'],['w','x','y','z'])
print (df5)

df5['aa'] = [4,5,6,7,8]
df5['ab'] = df5 ['y'] + df5['z']

df5.drop('e')
df5.drop('a')
df5.drop('aa', axis = 1, inplace = True)
df5.drop('e', inplace = True)


###############
df5[df5>4]
df5[df5>=4]

df5[df5>4][['y']]

df5[df5>4][['y','z']]

df5[df5['y']>4][['y']]

df5[df5['y']>4][['y','z']]
#-==============================
# & | ^
df5[(df5['w']>4) & (df5['x']>4)]

df5[(df5['w']>4) | (df5['x']>4)]

df5[(df5['w']>4) ^ (df5['x']>4)]

######### Missing Data
import numpy as np
np.nan
dict3 = {'a':[1,2,3,4], 'b':[3,4,5,np.nan],
         'c': [1,2,np.nan,np.nan], 'd':[np.nan,np.nan,np.nan,np.nan]}
df6 = pd.DataFrame(dict3)
print (df6)

df6.dropna()
df6.fillna(2)

df6['b'].fillna(value = df6['b'].mean())
df6['c'].fillna(value = df6['c'].mean(), inplace=True)

df6.drop('d', axis = 1)

#================

shop = {'items': ['egg','milk','milk', 'bread', 'egg','tofu'],
        'days':['mon', 'tue','wed','thu', 'fri', 'sat'],
        'sales': [50,100, 200, 50, 200, 100]}
df7 = pd.DataFrame(shop)
print (df7)

df7g = df7.groupby('items')
df7g.mean()
df7g.std()
df7g.count()
df7g.max()
df7g.min()
df7g.describe()
df7g.describe().transpose()

############## Merge, Join and concatenation

player = ['kohli', 'rohit', 'dhoni']
status = ['captain', 'batsman', 'wckt-keeper']
score = [50, 76, 56]

df8 = pd.DataFrame({'player':player, 'status':status, 'score':score})
print (df8)


player1 = ['kohli', 'bumrah', 'jadeja']
status1 = ['captain', 'bowler', 'All-rounder']
wicket = [1, 5, 3]

df9 = pd.DataFrame({'player1':player, 'status1':status, 'wicket':wicket})
print (df9)

pd.merge(df8, df9)
df8.merge(df9)

pd.merge(df8, df9, how = 'inner')

pd.merge(df8, df9, how = 'inner', on = 'status')
pd.merge(df8, df9, how = 'inner', on = 'player')

pd.merge(df8, df9, how = 'outer')
df8.merge(df9, how = 'outer')

pd.merge(df8, df9, how = 'left')
df8.merge(df9, how = 'left')

pd.merge(df8, df9, how = 'right')
df8.merge(df9, how = 'right')


#-===================== Join ======================
import pandas as pd

player = ['kohli', 'rohit', 'dhoni']
status = ['captain', 'batsman', 'wckt-keeper']
score = [50, 76, 56]

df8 = pd.DataFrame({'player':player, 'status':status, 'score':score})
print (df8)

player1 = ['kohli', 'bumrah', 'jadeja']
status1 = ['captain', 'bowler', 'All-rounder']
wicket = [1, 5, 3]

df9 = pd.DataFrame({'player1':player1, 'status1':status1, 'wicket':wicket})
print (df9)

df8.join(df9, how = 'inner')

df8.join(df9)

df8.join(df9, how = 'outer')


##### Concatenation
# Axis = 0 means rows
# axis = 1 means columns
pd.concat([df8, df9], ignore_index=True)

pd.concat([df8, df9])

pd.concat([df8, df9], axis = 0)

pd.concat([df8, df9], axis = 1)

df8['score'].sum()

df8.sort_values('score')
df8['score'].unique()
df8['score'].nunique()
df8['score'].value_counts()
df8['score'].isnull()

#============================================
# Data analysis
import pandas as pd
import numpy as np

df = pd.read_excel('weatherc.xlsx', sheet_name = 'Sheet1')

df1 = pd.read_excel('C:\\Users\\sshar127\\Desktop\\Learnings\\Pandas Datasets\\weatherc.xlsx', sheet_name = 'Sheet1')

# to skip rows
df1 = pd.read_excel('weatherc.xlsx', 'Sheet1', skiprows = 1)

sliced_df1 = df1[2:7]
sliced2_df1 = df1[5:8]

dffff = pd.DataFrame()


df1 = pd.read_excel('weatherc.xlsx', 'Sheet1', header = 1)
df1 = pd.read_excel('weatherc.xlsx', 'Sheet1', header = 2)


#########========================================
import pandas as pd
import numpy as np

df = pd.read_excel('weatherc.xlsx', sheet_name = 'Sheet1')

df1 = pd.read_excel('C:\\Users\\sshar127\\Desktop\\Learnings\\Pandas Datasets\\weatherc.xlsx', sheet_name = 'Sheet1')

df = pd.read_excel('weatherc.xlsx', 'Sheet1', skiprows = 1)

df = pd.read_excel('weatherc.xlsx', header = None)

df = pd.read_excel('weatherc.xlsx', header = None,
                   names = ['Date', 'temp', 'Air', 'Event'])

df = pd.read_excel('weatherc.xlsx', header = 0,
                   names = ['Date', 'temp', 'Air', 'Event'])

df = pd.read_excel('weatherc.xlsx', skiprows = 0,
                   names = ['Date', 'temp', 'Air', 'Event'])

df = pd.read_excel('weatherc.xlsx', nrows = 3)

df.columns

df.shape

######### for a specific column
df.temperature
### or
df['temperature'] 
df['temperature', 'day'] #it wont work
df[['temperature', 'day', 'windspeed']]

type(df['temperature'] )

df.max()
df['temperature'].max()
df['temperature'].min()

df.mean()
df['temperature'].mean()

df.median()
df['temperature'].median()

df.mode()
df['temperature'].median()

df.std()
df['temperature'].median()

df.describe()

# conditional accessing
df[df['temperature']>=32]
df[(df['temperature']>=32) & (df['windspeed']>=4)]
df[(df['temperature']>=32) ^ (df['windspeed']>=4)]

df[(df['temperature']>=32) & (df['temperature']<=33)]

df[(df['temperature']==35) & (df['temperature']==35)]
df[(df['temperature']==35)]

df[df['temperature']==df['temperature'].max()]
df[df['temperature']==df['temperature'].min()]

df.index

df.set_index('day', inplace = True)

df.reset_index(inplace = True)

#### how to save
df.to_excel('test.xlsx')
df.to_csv('test.csv')
df.to_excel('test2.xlsx', index = False)

df.to_excel('test4.xlsx', columns = ['temperature', 'windspeed'], index = False)

#### renaming columns
df1 = df.rename(columns = {'temperature':'temp'})
df1 = df.rename(columns = {'temperature':'temp',
                           'windspeed': 'air'})

df.to_csv('test1.csv', header = None, index = False)


####### Dealing with Missing Data
import pandas as pd

df = pd.read_excel('weatherc.xlsx', parse_dates = ['day'], header = 0,
                   names = ['day', 'temp', 'Air', 'Event'])
type(df.day[0])

df.set_index('day', inplace = True)

### filling
df1 = df.fillna(0)
print (df1)

# mean
df2 = df.temp.fillna(df.temp.mean())
df.temp = df.temp.fillna(df.temp.mean())
df.temp = df.temp.fillna(df.temp.median())
df.temp = df.temp.fillna(df.temp.mode())

##### 
df3 = df.fillna({'temp':25,
                 'Air': 4,
                 'Event': 'No event'})

df.fillna({'temp':25, 'Air': 4, 'Event': 'No event'}, inplace = True)

df3 = df.fillna(method = 'ffill')

df4 = df.fillna(method = 'ffill', limit = 1)

df3 = df.fillna(method = 'ffill', axis = 0)

df5 = df.fillna(method = 'ffill', axis = 1)

df6 = df.fillna(method = 'backfill', axis = 0)

df7 = df.fillna(method = 'bfill', axis = 0)

df8 = df.interpolate()


df = pd.read_excel('weatherc.xlsx')

#### removing NA values

df1 = df.dropna()
df1 = df.dropna(how = 'all')

df1 = df.dropna(thresh = 1)
df1 = df.dropna(thresh = 2)

df1 = df.dropna(how = 'all', thresh = 2)

######################
dt = pd.date_range('1/1/2020', '1/6/2020')
dt = df.reindex(dt)

# Correlation (only for numbers)
# wont work on categorical data or date etc
df2 = df[['temperature', 'windspeed']].corr()
df2 = df[['temperature', 'windspeed', 'day']].corr()





























