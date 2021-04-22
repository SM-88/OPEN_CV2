import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import argparse
from weasyprint import HTML,CSS
from weasyprint.fonts import FontConfiguration


def modeling(df):
    df1= pd.read_csv("cc163_.csv") # DB data
    def belt_mapping(df,df1):    
        df=df.drop(0)
        df=df.drop(['index'],axis=1)
        df=df.reset_index()
        for i in range(len(df1)):

            if i==0:
                df.loc[df[:df1.loc[i,"value"]].index,"Name"]=df1.loc[i,"TagName"][6:17]
                df.loc[df[:df1.loc[i,"value"]].index,"Time"]=df1.loc[i,"Scalemin"]/(df1.loc[i,"ScanTime"]*100)
                #df.loc[df[:df1.loc[i,"value"]].index,"Time"]=df1.loc[i,"Scalemin"]/(df1.loc[i,"ScanTime"]*100)
            else:
                df.loc[df1.loc[0:i-1,"value"].sum():df1.loc[0:i,"value"].sum(),"Name"]=df1.loc[i,"TagName"][6:17]
                df.loc[df1.loc[0:i-1,"value"].sum():df1.loc[0:i,"value"].sum(),"Time"]=df1.loc[i,"Scalemin"]/(df1.loc[i,"ScanTime"]*100)

        df.dropna(inplace=True)
        f1 = lambda x : x[0:2]
        dd=df.drop(['index','b','c','d'],axis=1)
        return dd

    df=belt_mapping(df,df1)
    belt_list=df["Name"].unique()
    df_split = dict(list(df.groupby(df["Name"])))
    
    for i in range(len(belt_list)):
        tmp=df_split[belt_list[i]][df_split[belt_list[i]]['a']>7]
        if len(tmp)!=0:
            if len(tmp)>5:
                peak_value=np.sum(tmp['a']*0.8)/len(tmp)
            else:
                peak_value=(np.sum(tmp['a']*0.8)/len(tmp))*0.8
            df.loc[df['Name']==belt_list[i],"peak"]=peak_value
        else:
            df.loc[df['Name']==belt_list[i],"peak"]=np.max(df_split[belt_list[i]]['a'])   
    dff=df.copy()
    re=dff.drop_duplicates(["Name"])
    ree=re.reset_index()

    for i in range(len(ree)):
        if ree.loc[i,'peak']>=8:
            ree.loc[i,'Status']="심각"

        elif ree.loc[i,'peak']>=7:
            ree.loc[i,'Status']="경고"

        elif ree.loc[i,'peak']>=6:
            ree.loc[i,'Status']="주의"

        elif ree.loc[i,'peak']>=5:
            ree.loc[i,'Status']="관심"
        else:
            ree.loc[i,'Status']="정상"
    ree.drop(["index","a"],axis=1,inplace=True)
    #ree.rename(columns = {'a' : 'sensor'}, inplace = True)
    #HTML export
    ree.to_html("belt_status.html",encoding='EUC-KR',justify='center',classes='df_style.css')

    html = HTML('./belt_status.html') #html file location

if __name__ == "__main__":
    print("Statiscal Modeling...")
    parser = argparse.ArgumentParser(description="Statistical_Model")
    parser.add_argument("--data_path", required=True, help="test data path")
    
    args = parser.parse_args()       

    df=pd.read_csv(args.data_path,names=["index","a","b","c","d"])
    modeling(df)
    print("Exporting belt_status.html,")
    print("===============END==============")
