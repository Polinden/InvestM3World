#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from IPython.display import Javascript
import ipywidgets as widgets
from scipy.stats import shapiro

#path. to datafile
p2df="/Users/denwolper/Desktop/Data_Extract_From_World_Development_Indicators vvishn.xlsx"
##################


# In[30]:


tb=widgets.ToggleButton(
    value=True,
    description='With Groups',
    disabled=False,
    button_style='info', 
    tooltip='Description',
    icon='check' 
)

df=pn.read_excel(p2df)
def on_button_clicked(value):
    global df
    if value['new']: 
        tb.description='With Groups'
        display(Javascript("Jupyter.notebook.execute_all_cells()"))
    else: 
        tb.description='No Groups'
        df=pn.read_excel(p2df)
        df.drop(df[-201:].index, inplace=True)
        display(Javascript("Jupyter.notebook.execute_cells([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])"))

display(tb)
pn.set_option('display.max_columns', None)
pn.set_option('display.max_rows', 70)
tb.observe(on_button_clicked, names='value')
df0=pn.read_excel(p2df)


# In[31]:


df.columns=list(map(lambda x: x[:4] if '[' in x  else x, df.columns))
df.columns=df.columns[:4].append(df.columns[4:].map(pn.to_numeric))


# In[32]:


df[df.columns[4:]]=df[df.columns[4:]].apply(pn.to_numeric, errors='coerce')
df[df.columns[:4]]=df[df.columns[:4]].astype(str) 


# In[33]:


md=df.set_index(['Country Name', 'Series Name'])
l1=md[(md.index.get_level_values('Series Name')=='Gross capital formation (% of GDP)')].iloc[:,0].unique()
l2=md[(md.index.get_level_values('Series Name')=='Broad money (% of GDP)')].iloc[:,0].unique()
l=np.unique(np.intersect1d(l1,l2))
df=df.loc[df['Country Code'].isin(l)]
df.drop(df.columns[3:4], axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
df.dropna(subset = ['Country Name'], how="any", inplace=True)


# In[34]:


s1=df.iloc[(df['Series Name']=='Broad money (% of GDP)').values, 3:].reset_index(drop=True)
s2=df.iloc[(df['Series Name']=='GDP, PPP (constant 2017 international $)').values, 3:].reset_index(drop=True)
df.iloc[(df['Series Name']=='Broad money (% of GDP)').values, 3:]=s1*s2/100
df.iloc[(df['Series Name']=='Broad money (% of GDP)').values, 2]='Broad money $'


# In[35]:


s1=df.iloc[(df['Series Name']=='Gross capital formation (% of GDP)').values, 3:].reset_index(drop=True)
df.iloc[(df['Series Name']=='Gross capital formation (% of GDP)').values, 3:]=s1*s2/100
df.iloc[(df['Series Name']=='Gross capital formation (% of GDP)').values, 2]='Gross capital formation USD'


# In[36]:


df.fillna(0, inplace=True)
print('\n Preprocessed World Statistics')
df.sort_values(by=['Country Code'])
df.reset_index()
#the preprocession above dropped empty rows and changed from relative (%) to absolut numbers
df  #print cleaned and preprocessed table


# In[37]:


#find and sort best candidateds
#calc Pearson corr for M2 and Investment
#score 1 for Pearson corr>threshold, otherwise 0

df2 = pn.DataFrame(columns=['Name','Code','K','GDPpercap2019', 'GOOD'])
for name in l:
    #settings
    i1=df[df['Country Code']==name].index[0]   #M2
    i2=i1+3                                    #Investment
    i3=i1+1                                    #GDP percap
    thresh=0.65                                #threshold for Pearson corr
    cut_year=-1                                #No2020
    top=40
    #########
    z=df.iloc[[i1,i2],3:cut_year].T
    first=(z.iloc[:,0]>0).idxmax()
    z=z.loc[first:]                            #drop heading zero columns 
    zz=z.corr() 
    if zz.iloc[0,1]>thresh:
        df2.loc[len(df2)] = [(df.iloc[i1,0]),name,zz.iloc[0,1],df.iloc[i3,-1], 1]
    else: 
        if pn.notna(zz.iloc[0,1]):
           df2.loc[len(df2)] = [(df.iloc[i1,0]),name,zz.iloc[0,1],df.iloc[i3,-1], 0]    
gb=len(df2.loc[df2['GOOD']==1])/len(df2)
print(f'\nGood to Bad Correlation Countries Rate {gb:0.2f}')
print(f'Top {top} Good Countries')
df2.set_index('Code').sort_values(['K'])[-top:]


# In[38]:


#experiment with the country selected
#input your country code (from the list above)

ls=(df2.loc[:,['Name', 'Code']].sort_values('Name').to_numpy())
cn=ls[0,1]
ls=[(l[0], l[1]) for l in ls]

drdw=widgets.Dropdown(
    options=ls,
    value=cn,
    description='Name:',
    disabled=False,
)

zpl=0 #first run
def fill_pl(): 
      global z, ax, fig, zpl, cn
      i1=df[df['Country Code']==cn].index[0]   #M2
      i2=i1+3                                  #Inv
      z=df.iloc[[i1,i2],3:cut_year].T
      first=(z.iloc[:,0]>0).idxmax()
      z=z.loc[first:]
      z.dropna(how="any", inplace=True) 
      z.reset_index(drop=True)
      z.columns=["M2","Inv"]
      sh1_s,sh1_p=shapiro(z.iloc[:,0].to_numpy())  
      sh2_s,sh2_p=shapiro(z.iloc[:,1].to_numpy()) 
      z.hist()
      plt.text(x=0.1, y=-0.1, s=f'Shapiro test:   {sh1_p:0.3f}   |   {sh2_p:0.3f}', fontsize=16, transform=plt.gcf().transFigure);
      
      clear_output(wait=True)
      ax.clear()
      ax.set_xlabel("Broad money, $", fontsize=12)
      ax.set_ylabel("Gross capital formation, $", fontsize=12)
      ax.set_title('Time series for M2 and Invest, USD', fontsize=16, y=-0.2)
      display(drdw) 
      if not zpl: 
        zpl=ax.plot(z)
        ax.legend([f'{cn}, Broad money, $', f'{cn}, Gross capital formation, $'], fontsize=12)
      else: 
         ax.plot(z)
         ax.legend([f'{cn}, Broad money, $', f'{cn}, Gross capital formation, $'], fontsize=12)
         display(ax.figure)
         display(Javascript("Jupyter.notebook.execute_cells([10,11,12,13,14,15,16])"))
      
             
        
def on_value_change(change):
      global cn
      cn=change['new']
      fill_pl() 
      
            
drdw.observe(on_value_change, names='value')  
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1,1,1)
fill_pl()


# In[39]:


f = plt.figure(figsize=(12, 4))
plt.matshow(z.corr(), fignum=f.number)
plt.xticks(range(z.select_dtypes(['number']).shape[1]), z.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(z.select_dtypes(['number']).shape[1]), z.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Pearson correlation Matrix, '+cn, fontsize=16, y=-0.1);


# In[40]:


print(f'\nPearson correlation for {cn} M2 and Investment')
z.corr()


# In[41]:


from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
reg  = linear_model.LassoLars(alpha=.1, normalize=True)


# In[42]:


get_ipython().run_cell_magic('capture', '', 'data=z[:-1].to_numpy()\nreg.fit(data[:,0,np.newaxis], data[:,1])')


# In[43]:


s=data.shape[0]
m1=data[:,0].max()
m2=data[:,0].min()
mm=np.linspace(m2, m1, num=s)
pp=reg.predict(mm[:,np.newaxis])
k1=reg.coef_[0]
k0=reg.intercept_
r2=r2_score(data[:,1], pp)


# In[44]:


plt.figure(figsize=(9, 6))
plt.scatter(data[:,0],data[:,1])
plt.plot(mm,pp, color='r')
tfy=r'$\frac{\mathtt{'+f'{cn}'+r'}}{'+f'y={k0:.4f}+{k1:.4f}x '+r'\longrightarrow r^2' +f'={r2:.4f}'+r'}$'
plt.title(tfy, y=-0.2, fontsize=20, color='brown')
plt.xlabel("Broad money, $", fontsize=12)
plt.ylabel("Gross capital formation, $", fontsize=12)
plt.legend(["prediction",'fact'], fontsize=12)
plt.text(x=0.1, y=-0.1, s='Investment=f(M2) regression  | y=f(x) | Coefficient of determination '+r'$r^2$', fontsize=16, transform=plt.gcf().transFigure);


# In[45]:


get_ipython().run_cell_magic('capture', '', "from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.model_selection import StratifiedShuffleSplit\nfrom sklearn.preprocessing import normalize\nneigh = KNeighborsClassifier(n_neighbors=3)\nsss = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=0)\nX=normalize(df2.loc[:,['K','GDPpercap2019']].to_numpy(), axis=0)\ny=df2.loc[:,['GOOD']].to_numpy().astype('int').ravel()\nsss.get_n_splits(X, y)")


# In[46]:


get_ipython().run_cell_magic('capture', '', 'for train_index, test_index in sss.split(X, y):\n      X_train, X_test = X[train_index], X[test_index]\n      y_train, y_test = y[train_index], y[test_index]\nneigh.fit(X_train, y_train)     ')


# In[47]:


x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/50), np.arange(y_min, y_max, (y_max-y_min)/50))
Z = neigh.predict(np.c_[xx.ravel(), yy.ravel()])


# In[48]:


from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

Z = Z.reshape(xx.shape)
plt.figure(figsize=(9, 7))
plt.contourf(xx, yy, Z, cmap=cmap_light)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Corr (M2 and Inv)')
plt.ylabel('GDP per capita ')
plt.title(f'Nearest neighbors classification for {len(df2)} countries: \nGDP per capita and Corr(M2 and Inv) normalized', fontsize=16, y=-0.2)
ce=1-sum(neigh.predict(X_test)-y_test)/y_test.shape[0]
print(f'Classification accuracy {ce:0.2f}')


# In[49]:


from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

plt.figure(figsize=(9, 7))
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

plt.subplot(111)
cdict={'b': 'blue','g': 'green','r': 'red','c': 'cyan','m': 'magenta','y': 'yellow','k': 'black','w': 'white'}
cmask=list(cdict.keys())
colors = cycle(cmask)
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + ".")
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )
plt.xlabel('Corr (M2 and Inv)')
plt.ylabel('GDP per capita ')
plt.title(f"Mean-shift clustering for {len(df2)} countries: \nGDP per capita and Corr(M2 and Inv) normalized'", fontsize=16, y=-0.2)

#count clusters
from collections import Counter
cntt=Counter(labels)
res=np.c_[X,labels]
cnt={k: v for k, v in sorted(cntt.items(), key=lambda item: item[1])}
mm=list(cnt.keys())[-1]                                                 #biggest claster name
mc=cntt[mm]                                                             #biggest claster size
df3 = pn.DataFrame(columns=['Name','Code','K','GDPpercap2019', 'GOOD']) #biggest claster content table
for i,v in enumerate(res): 
    if abs(v[2]-mm)<0.00001: df3.loc[len(df3)]=df2.iloc[i,:]


# In[50]:


print(f'The Biggest Cluster has color {cdict[cmask[mm]]} and contains {mc} or {mc/len(df2):.2%} of all countries')
print(f'Top of the Biggest Cluster is the following:')
#df3 =df3[df3['GDPpercap2019'] !=0]
df3.iloc[:,[0,2,3]].sort_values(by=['K'], ascending=False).reset_index(drop=True).T


# In[51]:


from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# In[ ]:




