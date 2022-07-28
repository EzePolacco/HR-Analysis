#!/usr/bin/env python
# coding: utf-8

# ## HR Analysis

# In[1]:


# 1) Importar librerías

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.pyplot import figure
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# 2) Cargar dataset.

mainpath = "/Users/ezequielpolacco/Desktop/Data Science/Kaggle"
filename = "/HR Dataset/HRDataset_v14.csv"
fullpath = mainpath + "/" + filename


# In[3]:


data = pd.read_csv(fullpath)


# In[4]:


data.head()


# In[5]:


data.columns # 3) Para conocer los nombres de las columnas


# In[6]:


data.describe().transpose() # 'Describe' otorga un breve resumen estadístico de los datos a analizar.


# In[7]:


# 4) Corroboramos si existen campos vacíos en cada columna y cuántos son en cada una.
data.isnull().sum()


# ## DATA CLEANING

# ##### Comienzo analizando cómo están escritas variables categóricas.

# In[8]:


data["RecruitmentSource"].unique()


# In[9]:


data["PerformanceScore"].unique()


# In[10]:


data["Position"].unique()


# In[11]:


data["Position"].replace('Data Analyst ', 'Data Analyst', inplace=True)


# In[12]:


data["Sex"].unique()


# In[13]:


data["MaritalDesc"].unique()


# In[14]:


data["CitizenDesc"].unique()


# In[15]:


data["HispanicLatino"].unique()


# In[16]:


data["HispanicLatino"].replace('yes', 'Yes', inplace=True)


# In[17]:


data["HispanicLatino"].replace('no', 'No', inplace=True)


# In[18]:


data["RaceDesc"].unique()


# In[19]:


data["TermReason"].unique().tolist()


# In[20]:


data["EmploymentStatus"].unique()


# In[21]:


data["Department"].unique()


# In[22]:


data["Department"].replace('Production       ', 'Production', inplace=True)


# #### Dentro de MANAGER ID encuentro valores faltantes. Los cuales a simple vista corresponden a 1 sólo individuo.

# In[23]:


data[data["ManagerID"].isna()]


# #### Hallo el ID, que corresponde a Webster Butler

# In[24]:


data[data["ManagerName"]=='Webster Butler'][['ManagerName', 'ManagerID']]


# #### Procedo a reemplazar valores nulos por el ID = 39 correspondiente.

# In[25]:


data["ManagerID"] = data["ManagerID"].replace(np.nan, 39.0) # reemplazo


# In[26]:


# Chequeo que se haya realizado correctamente
data[data["ManagerName"]=='Webster Butler'][['ManagerName', 'ManagerID']] 


# In[27]:


data["ManagerID"].isnull().sum()


# In[28]:


data.info() # 'Info' nos informa el tipo de dato que corresponde a cada variable contenida en columnas.


# In[29]:


data.shape


# In[30]:


plt.figure(figsize=(16,16))
sns.heatmap(data.corr(), annot=True)


# ##                         ANÁLISIS DE EMPLEADOS EN LA COMPAÑÍA
# ###### Para el caso se pretende analizar cuántos hombres y mujeres trabajan en la compañía, en qué sectores y posiciones tienen más presencia, etc. 
# ###### PAY EQUITY: A su vez, se relaciona la variable de Género respecto a Salario, Departamentos y Posición con el fin de obtener una idea de cuan equitativa es la nómina.

# In[31]:


# Filtro datos originales, quedándome sólo con las filas de aquellos que continúan trabajandi
employees = data.drop(data[data["TermReason"]!= "N/A-StillEmployed"].index)
employees.shape


# In[32]:


sex = employees.groupby(["Sex"]).size()
print(sex)


# In[33]:


# Distribución por sexo
plt.figure(figsize=(8,4))
sns.countplot(x="Sex", data = data)
plt.title("Distribución por sexo")


# In[34]:


# Distribución por sexo a través de los departamentos.

plt.figure(figsize=(12,6))
plt.title("Distribución por SEXO a través de los DEPARTAMENTOS")
sns.countplot(x="Department", hue="Sex", data = data)


# In[35]:


female = employees["GenderID"] == 0
female = female.sum()
print("Total género femenino: ", female)


# In[36]:


male = employees["GenderID"] == 1
male = male.sum()
print("Total género masculino: ", male)


# In[37]:


# Distribución por sexo a través de las posiciones

sex_data = employees.groupby(['Sex','Position']).size().reset_index()
sex_data.columns = ['Sex','Position','Count']
sex_data = sex_data.sort_values('Count', ascending = False)
sex_data


# In[38]:


# Gráfico donde se muestra la distribución de mujeres y hombres para cada posición.

fem_data = sex_data[sex_data.Sex=="F"]
male_data = sex_data[sex_data.Sex!="F"]

plt.figure(figsize=(15,5))
plt.bar(fem_data["Position"], fem_data["Count"], color="orange", label="Female")
plt.bar(male_data["Position"], male_data["Count"], label="Male")

plt.xticks(rotation=90)
plt.xlabel("Posición")
plt.ylabel("Cantidad de personas")
plt.title("Distribución de mujeres y hombres por cada Posición")

plt.legend()
plt.show()


# In[39]:


# Cómo se distribuyen mujeres y hombres dentro de cada Departamento?

sex_data_dept = employees.groupby(['Sex','Department']).size().reset_index()
sex_data_dept.columns = ['Sex','Department','Count']
sex_data_dept = sex_data_dept.sort_values('Count', ascending = False)
sex_data_dept


# In[40]:


# Gráfico donde se muestra la distribución de mujeres y hombres para cada posición.

fem_data_dept = sex_data_dept[sex_data_dept.Sex == "F"]
male_data_dept = sex_data_dept[sex_data_dept.Sex != "F"]

plt.figure(figsize=(15,5))
plt.bar(fem_data_dept["Department"], fem_data_dept["Count"], color = "orange", label = "Female")
plt.bar(male_data_dept["Department"], male_data_dept["Count"], label = "Male")

plt.xticks(rotation = 90)
plt.xlabel("Departamento")
plt.ylabel("Cantidad de personas")
plt.title("Distribución de mujeres y hombres por cada Departamento")

plt.legend()
plt.show()


# ## ANÁLISIS DE SALARIO Y GÉNERO (Pay equity)

# ### SALARIO POR SECTOR Y GÉNERO

# In[41]:


# Agrupo datos de variables a analizar.
dept_sex_salary = employees.groupby(["Department", "Sex"])["Salary"].mean()
dept_sex_salary


# In[42]:


# Cuál es el salario promedio para cada Departamento de acuerdo al sexo de la persona?

sex_data_dept = employees.groupby(['Sex','Department', 'Salary']).size().reset_index()
sex_data_dept.columns = ['Sex','Department', 'Salary','Count']
sex_data_dept = sex_data_dept.sort_values('Count', ascending = False)
sex_data_dept
plt.figure(figsize = (12,6))
sns.barplot(x= "Department", y="Salary", hue= "Sex", data=sex_data_dept)
plt.title("Salario promedio por Departamento de acuerdo a Sexo")
plt.show()


# ## agregar medidas estadística descriptiva para analizar sueldos M y H

# In[43]:


print("Salario máximo: ", max(employees["Salary"]))
print("Salario mínimo: ", min(employees["Salary"]))


# In[44]:


""""""salary_hist = employees["Salary"]

intervalos = range(min(employees["Salary"]), max(employees["Salary"]) + 2) #calculamos los extremos de los intervalos

plt.hist(x=salary_hist, bins=intervalos, color='#F2AB6D', rwidth=0.85)
plt.title('Histograma de Salarios')
plt.xlabel('Salario')
plt.ylabel('Frecuencia')
plt.xticks(intervalos)

plt.show()"""""" #dibujamos el histograma


# ### SALARIO PROMEDIO POR POSICIÓN Y GÉNERO

# In[45]:


dept_salary = employees.groupby(["Position", "Sex"])["Salary"].mean()
dept_salary


# In[46]:


sex_data_pos = employees.groupby(['Sex','Position', 'Salary']).size().reset_index()
sex_data_pos.columns = ['Sex','Position', 'Salary','Count']
sex_data_pos = sex_data_pos.sort_values('Count', ascending = False)
sex_data_pos
fig = px.bar(sex_data_pos, x = 'Position', y = 'Salary', color='Sex', title = 'Salario de empleados a través de las posiciones')
fig.show()


# ### SALARIO PROMEDIO POR SECTOR

# In[47]:


# Cuál es el salario promedio por sector?

dept_salary = employees.groupby(["Department"])["Salary"].mean()
dept_salary


# In[48]:


dept_salary = employees.groupby(["Department"])["Salary"].mean().reset_index()
dept_salary.columns = ["Department", "Salary"]
dept_salary = dept_salary.sort_values("Salary", ascending = False)
plt.figure(figsize = (12,6))
sns.barplot(x="Department", y="Salary", data= dept_salary)
plt.title("Salario promedio por Sector")
plt.show()


# ### SALARIO PROMEDIO POR PUESTO

# In[49]:


pos_salary = employees.groupby(["Position"])["Salary"].mean().sort_values(ascending = False)
pos_salary


# In[50]:


pos_salary = employees.groupby(["Position"])["Salary"].mean().reset_index()
pos_salary.columns = ["Position", "Salary"]
pos_salary = pos_salary.sort_values("Salary", ascending=False)
plt.figure(figsize = (50,6))
sns.barplot(x="Position", y="Salary", data=pos_salary)
plt.title("Salario promedio por Posición")
plt.show()


# ## DIVERSIDAD
# ###### Las variables del data set analizadas son CitizenDesc, RaceDesc y Género. 
# ###### El objetivo es conocer a través de estas variables qué tan diversa es la nómina respecto al origen de las personas, género, qué departamentos y posiciones ocupan y cuánto es la paga en promedio para cada caso. También observar las fuentes de reclutamiento en función de conocer por dónde llegan las personas a la compañía.

# ####           CIUDADANÍA

# In[51]:


# Dataset contiene información referida a ciudadanos estadounidenses y no ciudadanos. 
plt.figure(figsize=(8,4))
sns.countplot(x="CitizenDesc", data = employees)
plt.title("Distribución por Ciudadanía")


# In[52]:


plt.figure(figsize=(12,6))
plt.title("Ciudadanía por departamento")
sns.countplot(x="Department", hue="CitizenDesc", data = employees)
plt.title("Distribución de Ciudadanía a través de cada Departamento")
plt.legend()


# In[53]:


grouped_data = employees.groupby(['CitizenDesc','Department']).size().reset_index()
grouped_data.columns = ['CitizenDesc','Department','Count']
grouped_data = grouped_data.sort_values('Count', ascending = False)
fig = px.bar(grouped_data, x = 'Department', y = 'Count', color='CitizenDesc', title = 'Ciudadanía de los empleados para cada Departamento')
fig.show()


# In[54]:


## Distribución a través de cada Departamento
citi_dept = employees.groupby(["CitizenDesc", "Department"]).size().reset_index()
citi_dept


# In[55]:


## FUENTE DE RECLUTAMIENTO
recruiting = employees.groupby(["RecruitmentSource"]).size().reset_index()
recruiting.columns = ["RecruitmentSource", "Distribución"]
recruiting = recruiting.sort_values("Distribución", ascending = False)
fig = px.pie(recruiting, values = "Distribución", names = "RecruitmentSource")
fig.show()


# In[56]:


## Cuáles son las fuentes de reclutamiento que nos dan más diversidad?


# In[57]:


sex_recruited = employees.groupby(["RecruitmentSource", "Sex"]).size().reset_index()
sex_recruited.columns = ["RecruitmentSource", "Sex", "Distribución"]
sex_recruited = sex_recruited.sort_values("Distribución", ascending = False)
sex_recruited


# In[58]:


sex_recruited = employees.groupby(["RecruitmentSource", "Sex"]).size().reset_index()
sex_recruited.columns = ["RecruitmentSource", "Sex", "Distribución"]
sex_recruited = sex_recruited.sort_values("Distribución", ascending = False)
plt.figure(figsize = (12,6))
sns.barplot(x= "RecruitmentSource", y="Distribución", hue= "Sex", data=sex_recruited)
plt.show()


# ###### El 72,46% de las contrataciones de personal tiene como fuente de reclutamiento a Indeed, LinkedIn y Referidos por empleados. 
# ##### Indeed representa la vía de llegada del 18,35% de mujeres (66 sobre un total de 207 empleados), mientras que LinkedIn, en segundo lugar, aporta un 16,90% (35 mujeres). En el caso de Recomendaciones de empleados, un 7,24% de hombres llega por esta vía, siendo mayoría respecto a mujeres.

# In[ ]:





# In[59]:


# RECLUTAMIENTO POR CIUDADANÍA


# In[60]:


citi_recruited = employees.groupby(["RecruitmentSource", "CitizenDesc"]).size().reset_index()
citi_recruited.columns = ["RecruitmentSource", "CitizenDesc", "Distribución"]
citi_recruited = citi_recruited.sort_values("Distribución", ascending = False)
plt.figure(figsize = (12,6))
sns.barplot(x= "RecruitmentSource", y="Distribución", hue= "CitizenDesc", data=citi_recruited)
plt.show()


# In[ ]:





# In[78]:


# Nivel de desempeño de cada departamento
plt.figure(figsize=(12,6))
sns.countplot(data=employees, x = "Department", hue = "PerfScoreID", palette="Set2")


# In[79]:


performance_dept = employees.groupby(["Department","PerfScoreID"]).size().reset_index()
performance_dept.columns = ["Department","PerfScoreID", "Distribución"]
performance_dept = performance_dept.sort_values("PerfScoreID", ascending = False)
performance_dept.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## ANÁLISIS DE DESVINCULACIONES
# 
# ###### Se busca conocer motivos de desvinculación de los empleados, además de observar qué sectores o managers reciben más renuncias

# In[61]:


# Para este análisis, se crea un segundo dataset borrando las filas que coincidan con el valor "N/A-StillEmployed" para la columna "TermReason" 

# El propósito es analizar variables por las cuales las personas se desvinculan de la compañía.

attrition_data = data.drop(data[data["TermReason"]== "N/A-StillEmployed"].index)
attrition_data


# In[62]:


# CUÁLES SON LAS CAUSAS DE LAS RENUNCIAS?

term_reason = attrition_data.groupby(['TermReason']).size().reset_index()
term_reason.columns = ['TermReason','Distributions']
term_reason = term_reason.sort_values('Distributions', ascending = False)
fig = px.pie(term_reason, values = 'Distributions', names = 'TermReason')
fig.show()


# In[63]:


# CUÁLES SON LOS DEPARTAMENTOS DONDE MÁS SE RENUNCIA?

attrition_dept = attrition_data.groupby(["Department","TermReason"]).size().reset_index()
attrition_dept.columns = ["Department","TermReason", "Distribución"]
attrition_dept = attrition_dept.sort_values("Distribución", ascending = False)
attrition_dept
#plt.figure(figsize = (12,6))
#sns.barplot(x= "TermReason", y="Distribución", hue= "Department", data=sex_recruited)
#plt.show()


# ##### En cuanto a los motivos de desvinculación, el principal se debe a cambios de posición hacia fuera de la compañía (15,38%), seguido por el descontento (13,46%), y en tercer lugar se encuentra el cambio de compañía por mayor salario (10,57%).

# In[64]:


# A QUÉ MANAGER LE RENUNCIARON MAYOR CANTIDAD DE PERSONAS?

attrition_manager = attrition_data.groupby(["ManagerName"]).size().reset_index()
attrition_manager.columns = ["ManagerName","Distribución"]
attrition_manager = attrition_manager.sort_values("Distribución", ascending = False)
attrition_manager
plt.figure(figsize = (25,6))
sns.barplot(x= "ManagerName", y="Distribución", data=attrition_manager)
plt.show()


# In[65]:


# ¿Qué manager recibe más cantidad de renuncias?

attrition_manager.head(5)


# # QUÉ POSICIONES RENUNCIAN MÁS?

# In[66]:


# CUÁLES SON LAS POSICIONES RENUNCIADAS?

pos_attrition = attrition_data.groupby(["Position"]).size().reset_index()
pos_attrition.columns = ["Position", "Distribución"]
pos_attrition = pos_attrition.sort_values("Distribución", ascending = False)
pos_attrition
plt.figure(figsize = (25,6))
sns.barplot(x= "Position", y="Distribución", data=pos_attrition)
plt.show()


# ##### Los puestos que presentan mayor rotación son: 
# ##### - Production Technician I (50%) 
# ##### - Production Technician II (25%)
# ##### - Production Manager (4,8%)

# In[71]:


pos_attrition = attrition_data.groupby(["Position", "Department",]).size().reset_index()
pos_attrition.columns = ["Position","Department", "Distribución"]
pos_attrition = pos_attrition.sort_values("Distribución", ascending = False)
pos_attrition.head(5)


# In[72]:


attrition_sex_dept = attrition_data.groupby(["Sex","Department"]).size().reset_index()
attrition_sex_dept.columns = ["Sex", "Department","Distribución"]
attrition_sex_dept = attrition_dept.sort_values("Department", ascending = False)
attrition_sex_dept.head()


# In[ ]:





# In[184]:


attrition_dept = attrition_data.groupby(["ManagerName","Department","TermReason"]).size().reset_index()
attrition_dept.columns = ["ManagerName", "Department","TermReason", "Distribución"]
attrition_dept = attrition_dept.sort_values("ManagerName", ascending = False)
attrition_dept.head()


# In[73]:


# RENUNCIAN MÁS HOMBRES O MUJERES?

sex_attrition = attrition_data.groupby(["Sex"]).size().reset_index()
sex_attrition.columns = ["Sex", "Distribución"]
sex_attrition = sex_attrition.sort_values("Distribución", ascending = False)
sex_attrition
plt.figure(figsize = (6,4))
sns.barplot(x= "Sex", y="Distribución", data=sex_attrition)
plt.show()


# ##### 

# In[ ]:





# In[74]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




