

import streamlit as st
import streamlit.components.v1 as stc

# File Processing Pkgs
import pandas as pd
import numpy as np

import xgboost as xgb
#from sklearn.linear_model import LinearRegression,Ridge
#from sklearn.metrics import mean_squared_error,mean_absolute_error

#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression


def main():
    st.title("File Upload ")

    menu = ["Dataset","About"]
    choice = st.sidebar.selectbox("Menu",menu)
       
    measures = ['axl1_l_w_flange', 'axl1_r_w_flange', 'axl2_l_w_flange', 
'axl2_r_w_flange', 'axl3_l_w_flange', 'axl3_r_w_flange', 'axl4_l_w_flange', 'axl4_r_w_flange']


    if choice == "Dataset":
        st.subheader("Dataset")
        data_file = st.file_uploader("Upload CSV",type=['csv'])
        if st.button("Process"):
            if data_file is not None:
                file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
                st.write(file_details)

                df = pd.read_csv(data_file)
                st.dataframe(df)
                       
                # prepare test data
                df['mes_id'] = df['wagnum'].apply(str) + '_' + df['ts_id'].apply(str)
                     
                lin_regr_dict = {}
                lin_regr_dict["mes_id"] = []
                lin_regr_dict["ts_id"] = []
                lin_regr_dict["milleage_all"] = []

                for i in measures:
                    lin_regr_dict[i+"_"+"a"] = []
                    lin_regr_dict[i+"_"+"b"] = []
                    

                
                for m_id in df["mes_id"].unique():
                    lin_regr_dict["mes_id"].append(m_id)
                    lin_regr_dict["ts_id"].append(df[df["mes_id"] ==m_id]["ts_id"].max())
                    lin_regr_dict["milleage_all"].append(df[df["mes_id"] ==m_id]["milleage_all"].max())

                    for m in measures:
                        lg = LinearRegression(n_jobs = -1)
                        x = df[df['mes_id'] == m_id]["milleage_all"].to_numpy()#.reshape(-1,1)
                        y = df[df['mes_id'] == m_id][m].to_numpy()
                        not_na_indexes = ~np.isnan(y) & ~np.isnan(x)
                        lg.fit(x[not_na_indexes].reshape(-1,1),y[not_na_indexes].reshape(-1,1))


                        a = lg.coef_
                        b = lg.intercept_
                        lin_regr_dict[m+"_"+"a"].append(float(a))
                        lin_regr_dict[m+"_"+"b"].append(float(b))

                test_lin_regr_np = pd.DataFrame(lin_regr_dict).drop(columns = ['mes_id'])
                
                
                for m in measures:
                    test_lin_regr_np[m] = test_lin_regr_np["milleage_all"] *test_lin_regr_np[m+"_a"]+test_lin_regr_np[m+"_b"]
                
                
                for m in measures:
                    ar = []
                    eps = 2

                    for index,row in test_lin_regr_np.iterrows():
                        #print(row["mes_id"])
                        av_target = 0
                        current_m_value = row[m]
                        mask = (current_m_value-eps<test_lin_regr_np[m])&(test_lin_regr_np[m]<current_m_value+eps)
                        common_values = test_lin_regr_np[mask][m]
                        ar.append(common_values.mean())
                    test_lin_regr_np["av_target_for_"+m] = ar

                
                
                
                
                test_lin_regr_np = test_lin_regr_np.to_numpy()
                
                
                
                
                
                
                
                
                
                lin_regr_df = pd.read_csv('data/lin_regr_df_upd_with_mean_target_feature.csv')

                #train 
                params = {
                    "objective": "reg:squarederror",
                    "n_estimators":100,
                    "max_depth": 5,#5
                    "eval_metric":'mae'
                }


                y = lin_regr_df["target"]
                #x = scaler.fit_transform(lin_regr_df.drop(columns=["target","mes_id", 'wagnum']))
                x = lin_regr_df.drop(columns=["target","mes_id", 'wagnum'])
                model = xgb.XGBRegressor( **params)
                model.fit(x,y)
                y_pred = model.predict(x)

                error = np.abs(y_pred - y)
                mask = error<30000#Здесь не менять

                model = xgb.XGBRegressor(**params)
                model.fit(x[mask],y[mask])


                #scaled_test_lin_regr_np = scaler.transform(test_lin_regr_np)
                pred = model.predict(test_lin_regr_np)
                pred[pred<600] = 5000



                
                for i in range(len(pred)):
                    st.write('(wagnum)_(ts_id)',lin_regr_dict['mes_id'][i])
                    st.write('Predict', pred[i] )
                

                



    else:
        st.subheader("About")
        st.info("ML ботать нельзя спать")



if __name__ == '__main__':
    main()



