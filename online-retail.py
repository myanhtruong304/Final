import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

import squarify
import datetime as datetime

df = pd.concat([pd.read_csv('OnlineRetail_1.csv', encoding='ISO-8859-1').dropna(), pd.read_csv('OnlineRetail_2.csv', encoding='ISO-8859-1').dropna()])

##EDA
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.dropna(inplace = True)
df['CustomerID'] = df['CustomerID'].astype('object')
df_1 = df[df['Quantity'] >0]
df_1['Revenue'] = df_1.Quantity*df_1.UnitPrice
max_date = df_1['InvoiceDate'].max().date()
recency = lambda x: (max_date - x.max().date()).days
frequency = lambda x: len(x.unique())
monetary = lambda x: round(sum(x), 2)

RFM = df_1.groupby('CustomerID').agg({'InvoiceDate': recency,
                                    'InvoiceNo': frequency,
                                    'Revenue': monetary})
RFM.columns = ['Recency', 'Frequency', 'Monetary']
RFM.reset_index(inplace = True)

r_label = range(4, 0, -1)
f_label = range(1, 5)
m_label = range(1, 5)

r_groups = pd.qcut(RFM['Recency'].rank(method='first'), q = 4, labels = r_label)
f_groups = pd.qcut(RFM['Frequency'].rank(method='first'), q = 4, labels = f_label)
m_groups = pd.qcut(RFM['Monetary'].rank(method='first'), q = 4, labels = m_label)
RFM = RFM.assign(R = r_groups.values, F = f_groups, M = m_groups.values)
def join_rfm(x) : return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
RFM['RFM_Segment'] = RFM.apply(join_rfm, axis = 1)
RFM['RFM_Score'] = RFM[['R', 'F', 'M']].sum(axis = 1)
##EDA
country = df_1.groupby(['Country']).agg({'CustomerID': 'count',
                               'Quantity': 'mean'}).sort_values('CustomerID', ascending = False)

##Model
wsse = []
K = range(1, 15)
for k in K:
  k_model = KMeans(n_clusters = k)
  k_model.fit(RFM[['Recency', 'Frequency', 'Monetary']])
  wsse.append(k_model.inertia_)

kmeans_model = KMeans(n_clusters = 4)
kmeans_model.fit(RFM[['Recency', 'Frequency', 'Monetary']])
RFM['KNN_cluster'] = kmeans_model.labels_


st.title('Topic 2: Online Retail')
menu = ['General', 'Basic Exploration', 'Customer Segmentation']
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'General':
    st.subheader('Team Members:')
    st.write('Nguyễn Trương Mỹ Anh')
    st.write('Ngô Đình Ngọc Thúy')

    st.subheader('Abstract')
    st.image('im.png')
    st.write('Customer segmentation is the activity of grouping your customers by several characteristics. It can be their personal information, spending behavior, demographic, etc. The purpose of doing customer segmentation is to understand each segment, so you can market and promote your brand effectively.')
    st.write('In this customer segmentation analysis, we use Online Retail customer segementation dataset. ')

    st.subheader('About Dataset')
    st.write('Number of records: ', len(df))
    st.write('Features:', list(df.columns))
    st.write('Number of orders:' ,len(df.InvoiceNo.unique()))
    st.write('Number of Customer:', len(df.CustomerID.unique()))
    st.write('Number of items:', len(df.StockCode.unique()))
    st.write('Invoice Date from {} to {}'.format(min(df.InvoiceDate), max(df.InvoiceDate)))
elif choice == 'Basic Exploration':
    st.header('Model Deployment')
    st.subheader('**RFM Method**')
    st.write("Recency, frequency, monetary value (RFM) is a marketing analysis tool used to identify a firm's best clients based on the nature of their spending habits.")
    st.dataframe(RFM.head(10))
    st.write('Australia, Netherlands, Japan, Sweden have highest average quantity per order while UK and German have highest number of customer.')
    
    fig = plt.figure(figsize = (20,8))
    sns.barplot(data = country.sort_values('Quantity', ascending = False).reset_index(), x = 'Country', y = 'Quantity', ci = 0)
    plt.xticks(rotation = 'vertical')
    plt.title('Average Quantity per Order.')
    st.pyplot(fig)

    st.write('Country with number of order.')
    st.dataframe(df_1.groupby(['Quantity', 'Country']).agg({'CustomerID': 'count'}).sort_values('Quantity', ascending = False)[:50])
    st.write('UK has number of orders which have > 1000 item. Assuming UK is wholesales market.')
    st.dataframe(df_1[df_1['Quantity'] >=1000].groupby('Country').agg({'InvoiceNo': 'count'}))
    st.write('With 107 orders which has >1000 items per order, UK should be considered as a wholesales market.')

    st.write('100 Best Sellers')
    fig_1 = plt.figure(figsize = (20, 8))
    sns.barplot(data = df_1.sort_values('Quantity', ascending = False)[:100], x = 'Description', y = 'Quantity', ci = 0)
    plt.xticks(rotation = 'vertical')
    st.pyplot(fig_1)

    st.write("Paper craft, Medium Ceramic Top is best sellers with total selled item is much higher than the others.")

    st.write('100 Worst Sellers')
    fig_1 = plt.figure(figsize = (20, 8))
    sns.barplot(data = df_1.sort_values('Quantity', ascending = True)[:100], x = 'Description', y = 'Quantity', ci = 0)
    plt.xticks(rotation = 'vertical')
    st.pyplot(fig_1)
elif choice == 'Customer Segmentation':
    kmeans_model_2 = KMeans(n_clusters = 4)
    kmeans_model_2.fit(RFM[['R', 'F', 'M']])
    RFM['KNN_cluster_2'] = kmeans_model_2.labels_
    
    st.write('RFM factors illustrate these facts:')
    st.write('- The more recent the purchase, the more responsive the customer is to promotions')
    st.write('- The more frequently the customer buys, the more engaged and satisfied they are')
    st.write('- Monetary value differentiates heavy spenders from low-value purchasers')
    st.write("Clustering algorithm will help us to better understand customers, in terms of both static demographics and dynamic behaviors by conducting RFM analysis, let's see how we can score these customers by ranking them based on each RFM attribute.")
    st.write("This pproach of scaling customers from 1-4 will result in many different RFM scores, ranging from 111(lowest) to 444(highest). Each RFM cell will differ in size and vary from one another, in terms of the customer's key habits, captured in the RFM score. We can't analyze all segments individually if each RFM cell is considered a segment.")
    fig_2 = plt.figure(figsize = (20, 5))
    sns.barplot(data=RFM.sort_values(by = 'KNN_cluster_2', ascending = False).reset_index(), x = 'KNN_cluster_2', y = 'RFM_Score', ci = 0)
    plt.xticks(rotation = 'vertical', fontsize = 16)
    plt.xlabel('Cluster', fontsize = 13)
    plt.ylabel('Frequecy', fontsize = 13)
    plt.title('Cluster')
    st.pyplot(fig_2)

    fig_3 = plt.figure(figsize = (15, 8))
    sns.scatterplot(data = RFM[['R', 'F', 'M', 'KNN_cluster_2']], x = 'R', y = 'F', hue = 'KNN_cluster_2', palette = 'rainbow')
    plt.title('Scatter plot R, F')

    fig_4 = plt.figure(figsize = (15, 8))
    sns.scatterplot(data = RFM[['R', 'F', 'M', 'KNN_cluster_2']], x = 'R', y = 'M', hue = 'KNN_cluster_2', palette = 'rainbow')
    plt.title('Scatter plot R, M')
    st.pyplot(fig_3)
    st.pyplot(fig_4)

    st.write('Number of customer per cluster.')
    st.dataframe(RFM.groupby('KNN_cluster_2').agg({'KNN_cluster': 'count'}))

    RFM_KNN_group = RFM.groupby('KNN_cluster_2').agg({'Recency':'mean',
                                'Frequency':'mean',
                                'Monetary':['mean', 'count']}).round(0)

    RFM_KNN_group.columns = RFM_KNN_group.columns.droplevel()
    RFM_KNN_group.columns = ['Recency', 'Frequency_Mean', 'Monetary_Mean', 'Count']
    RFM_KNN_group['Percent'] = round((RFM_KNN_group['Count']/RFM_KNN_group.Count.sum()*100), 2)

    RFM_KNN_group = RFM_KNN_group.reset_index()
    st.write('Customer Segmentation')
    st.dataframe(RFM_KNN_group)

    RFM_segmentation_KNN = RFM.groupby('KNN_cluster_2').agg({'Recency':'mean',
                                 'Frequency':'mean',
                                 'Monetary': ['mean', 'count']})

    RFM_segmentation_KNN.columns = RFM_segmentation_KNN.columns.droplevel()
    RFM_segmentation_KNN.columns = ['Recency', 'Frequency_Mean', 'Monetary_Mean', 'Count']
    RFM_segmentation_KNN['Percent'] = round((RFM_segmentation_KNN['Count']/RFM_segmentation_KNN.Count.sum()*100), 2)
    RFM_segmentation_KNN.reset_index(inplace = True)

    fig_5 = plt.gcf()
    ax = fig_5.add_subplot()
    fig_5.set_size_inches(14,10)

    colors_dicts = {'Active': 'green', 'Lost': 'darkgrey', 'Loyal': 'gold', 'New': 'lightgreen',
                    'Regular': 'pink', 'Rich':'purple', 'VIP': 'red'}
        
    squarify.plot(sizes=RFM_segmentation_KNN['Count'],
                text_kwargs = {'fontsize':12, 'weight': 'bold', 'fontname':'sans serif'},
                color=colors_dicts.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $\n{:.0f} customers ({}%)'.format(*RFM_segmentation_KNN.iloc[i])
                for i in range(0, len(RFM_segmentation_KNN))], alpha = 0.5)
    plt.title('Customers Segments', fontsize=26, fontweight = 'bold')
    plt.axis('off')

    st.pyplot(fig_5)
    st.image('im2.png')
    st.subheader("Let's delve into few interesting segments!")
    st.write('- At Rist: customer with high Receny, average Frequency, average Monetary')
    st.write('- Lost: customer with highest Recency, low Frequency, low Monetary.')
    st.write('- Casual: customer with low Recency, low Frequency, average to low Monetary.')
    st.write('- Potential: customer with lowest Recency, highest Frequency, highest Monetary.')
