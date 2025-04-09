import streamlit as st

st.title('🎈 App Name')

st.write('Hello world!')

from sklearn import datasets, ensemble

st.write ("""### gas consumption """)


st.write ("""Welcome to gas consumption app!""")

with st.expander('Data'):

  st.sidebar.header('Enter your information:')

  df = pd.read_csv('datashahr4-st.csv')



def users_input_features():
   global N_of_consumers
   global ave_temp_year
   global population
   global min_min_temp_cold

min_min_temp_cold =st.sidebar.slider('min_min_temp_cold',0.0,10.0,1.0,0.1)
ave_temp_year=st.sidebar.slider('ave_temp_year',0.0,20.0,1.0,0.1)
population =st.sidebar.slider('population',800.0,2000.0,1000.0,100.0)
N_of_consumers =st.sidebar.slider('N_of_consumers',800.0,10000.0,1000.0,100.0)



df=df.iloc[:9,:]

X = df.iloc[:9,3:11].values
y = df.iloc[:9,1].values



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)

params = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 0.65,
    "learning_rate": 0.1,
    }

reg = ensemble.GradientBoostingRegressor (**params)
reg.fit(X_train, y_train)

with st.expander('inputs'):
 st.write ("""### Please input your data using sidebars: """)

 x1= (N_of_consumers)
 x2= (ave_temp_year)
 x3= (population)
 x4= (min_min_temp_cold)

 X= np.array([ x1,0,0,x2,x3,x4,0,0])
 X=X.reshape(1, -1)
 X

with st.expander('prediction'):


 y_pred = reg.predict(X)

 y_pred
