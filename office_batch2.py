# Numpy

# 3d array
import numpy as np
list1 = [[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]]]
arr3d = np.array(list1)
print (arr3d)
arr3d.ndim
arr3d.reshape(9,3,1)

arr3d2 = np.arange(0,24)
print (arr3d2)
arr3d2.ndim

arr2d_C = arr3d2.reshape(8,3)
arr3d_C = arr3d2.reshape(8,3,1)

arr3d_C = arr3d2.reshape(1,8,3)


# Reg-ex
import re
string = 'This is just A string.,   '
print (re.sub('[A-Z]', '', string))
print (re.sub('[a-z]', '', string))
print (re.sub('[a-zA-Z ]', '', string))

print (re.sub('[A-Z ,.]', '', string))

print (re.sub('[^A-Z]', '', string))

# 
food = 'pizza chapati veg burger'
# x = re.compile('veg')
# food = x.sub('chicken', food)
food1 = re.sub('veg', 'non-veg', food)


text1 = '''I
love
python
and ML'''
y = re.compile('\n')
print (y.sub(' ', text1))

# \n - new line, \t - tab, \d = number, \D = non numbers

num1 = '12345 values'
print(len(re.findall('\d', num1)))
print(len(re.findall('\D', num1)))

111-222-3333
cont = re.compile('\d\d\d-\d\d\d-\d\d\d\d')
str1 = 'My contact number is 111-222-3333'
print (cont.search(str1))

if re.search('if', 'If I can do it, you all can do it'):
    print ('It is available')


str2 = 'set met get wet jet'
str2C = re.findall('[s,m,g,w,j]et', str2)
for i in str2C:
    print (i)

str2 = 'set met get wet jet'
str2C = re.findall('[a-o]et', str2)
for i in str2C:
    print (i)


str2 = 'set met get wet jet'
str2C = re.findall('[a-z]et', str2)
for i in str2C:
    print (i)
    
    
str2 = 'set met get wet jet'
str2C = re.findall('[l-z]et', str2)
for i in str2C:
    print (i)   
    
str2 = 'set met get wet jet'
str2C = re.findall('[^a-o]et', str2)
for i in str2C:
    print (i)    
    
str6 = 'Hi Hi, we we are working'
repeat = re.findall(r'\b\w[a-zA-Z]\b', str6)
print (repeat)    
    
    
    
    
    
    
    
    
    
