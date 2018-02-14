import pandas as pd
import io
import csv
import datetime
from pandas_datareader import data







# with open('/home/ubuntu/Downloads/sample_1.csv','rb') as csvfile:
#     data = csv.reader( csvfile )
#     # row_count = sum(1 for row in data)
#     print(data)

# read data from csv file

df=pd.read_csv('/home/ubuntu/Downloads/sample_1.csv')

# df=pd.read_csv('sample_1.csv')


# df=pd.read_csv(io.StringIO('/home/ubuntu/Downloads/sample_1.csv'))
print(df)


# index_col=0 means no index in output
# df=pd.read_csv('/home/ubuntu/Downloads/sample_1.csv',index_col=0)






#
# XYZ_web ={'day':[1,2,3,4,5,6],'visitors':[100,22,44,33,44,5],'bounce_rate':[22,33,44,5,55,66]}
#
# df=pd.DataFrame(XYZ_web)
# represent all data

# print(df)


# ...............................................................slicing of data..................


# represent a part of data
# to get data  top 2 vertical rows so head(2) is 2  represent 2 rows
# print(df.head(2))

# to get data from  lower  2 vertical rows so tail() is 2  represent 2 column
# print(df.tail(2))



# ................................................................merging of data..........



# there are two seperate data frame to merge
# work for all HPI if all values of two vector is same if any vaalue of vector is different it ommit that row
# df1=pd.DataFrame({'HPI':[2,3,4,5],"int_rate" :[2,4,5,6],"ind_gdp":[2,4,6,7]},index=[2001,2002,2003,2004])
# df2=pd.DataFrame({'HPI':[2,3,4,5],"int_rate" :[2,4,5,6],"ind_gdp":[2,4,6,7]},index=[2005,2006,2007,2008])
# to merge two seperate data frame
# df=pd.merge(df1,df2)
# print(df)
# to make certain coulumn common
# this will print all values of vector make new column to print all data . does not depend on same or different
# here only hpi column is common and other column is different

# df3=pd.DataFrame({'HPI':[2,3,4,5],"int_rate" :[2,4,5,6],"ind_gdp":[2,4,6,7]},index=[2001,2002,2003,2004])
# df4=pd.DataFrame({'HPI':[2,3,4,5],"int_rate" :[2,4,4,6],"ind_gdp":[2,1,6,7]},index=[2005,2006,2007,2008])
# df=pd.merge(df3,df4,on = 'HPI')
# print(df)





# ------------------------------------


# start = datetime.datetime(2010, 1, 1)
# end = datetime.datetime(2015, 8, 22)
# print(start)


# ....................................................................joining data frame....


# df1=pd.DataFrame({"int_rate" :[2,4,5,6],"ind_gdp":[2,4,6,7]},index=[2001,2002,2003,2004])
# df2=pd.DataFrame({"low_tire_HPI" :[2,4,5,6],"Unemployment":[2,4,6,7]},index=[2001,2003,2004,2005])
# join= df1.join(df2)
#
# print(join)

# -------------------------------------



# index and  coulumn header can be changed by pandas and concatinate








































# df=pd.read_csv("sample.csv")




#
# import matplotlib.pyplot as plt
# plt.plot([1,2,3],[5,7,4])
# plt.show()