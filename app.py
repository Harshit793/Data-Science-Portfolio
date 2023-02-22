import streamlit as st
import joblib
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
import six
import sys
sys.modules['sklearn.externals.six'] = six
PAGE_CONFIG={"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)
def main():
#title
   st.header("Healthcare Provider Fraud Detection App")

# input bar 1
   Clm=st.number_input("Enter Claim Amt Reimbursed ",key=0)

# input bar 2
   Ded=st.number_input("Enter Deductible Amt Paid ",key=1)

# input bar 3
   admit_0=st.selectbox("Enter 1 if non-admitted ",np.array([0,1]),key=2)

# input bar 
   admit_1=st.selectbox("Enter 1 if admitted ",np.array([0,1]),key=3)

# input bar 4
   day=st.number_input("Enter no of days admitted ",key=4)

# input bar 5
   gender_0=st.selectbox("Enter 0 if female  ",np.array([0,1]),key=5)

# input bar 6
   gender_1=st.selectbox("Enter 1 if  male",np.array([0,1]),key=6)

# input bar 7
   race_0=st.selectbox("Enter 1 if  race 0",np.array([0,1]),key=7)

# input bar 8
   race_1=st.selectbox("Enter 1 if  race 1 ",np.array([0,1]),key=8)

#  input bar 
   race_2=st.selectbox("Enter 1 if  race 2 ",np.array([0,1]),key=9)

# input bar 9
   Rdi_0=st.selectbox("Enter 1 if renal disease condition non-exist",np.array([0,1]),key=10)

# input bar 10
   Rdi_1=st.selectbox("Enter 1 if  renal disease condition exist",np.array([0,1]),key=11)

# input bar 11
   state=st.number_input("Enter state",key=12)

# input bar 12
   country=st.number_input("Enter country",key=13)

# input bar 13
   part_a=st.number_input("Enter no of months part A covered ",key=14)

# input bar 14
   part_b=st.number_input("Enter no of months part B covered ",key=15)

# input bar 15
   chr_alz_0=st.selectbox("Enter 1 if chronic alzheimer condition  non-exist",np.array([0,1]),key=16)

# input bar 16
   chr_alz_1=st.selectbox("Enter 1 if  chronic alzheimer condition exist",np.array([0,1]),key=17)

# input bar 17
   chr_heart_0=st.selectbox("Enter 1 if chronic heart condition non-exist",np.array([0,1]),key=18)

# input bar 18
   chr_heart_1=st.selectbox("Enter 1 if chronic heart condition exist",np.array([0,1]),key=19)

# input bar 19
   chr_kidney_0=st.selectbox("Enter 1 if chronic kidney condition non-exist",np.array([0,1]),key=20)

# input bar 20
   chr_kidney_1=st.selectbox("Enter 1 if chronic kidney condition exist",np.array([0,1]),key=21)

# input bar 21
   chr_cancer_0=st.selectbox("Enter 1 if chronic cancer condition non-exist",np.array([0,1]),key=22)

# input bar 22
   chr_cancer_1=st.selectbox("Enter 1 if chronic cancer condition exist",np.array([0,1]),key=23)

# input bar 23
   chr_pulmonary_0=st.selectbox("Enter 1 if  chronic pulmonary condition non-exist",np.array([0,1]),key=24)

# input bar 24
   chr_pulmonary_1=st.selectbox("Enter 1 if  chronic pulmonary condition exist",np.array([0,1]),key=25)

# input bar  25
   chr_depression_0=st.selectbox("Enter 1 if chronic depression condition  non-exist",np.array([0,1]),key=26)

# input bar 26
   chr_depression_1=st.selectbox("Enter 1 if chronic depression condition  exist",np.array([0,1]),key=27)

# input bar 27
   chr_diabetes_0=st.selectbox("Enter 1 if  chronic diabetes condition non-exist",np.array([0,1]),key=28)

# input bar 28
   chr_diabetes_1=st.selectbox("Enter 1 if  chronic diabetes condition exist",np.array([0,1]),key=29)

# input bar 29
   chr_ischemicheart_0=st.selectbox("Enter 1 if chronic ischemicheart condition non-exist ",np.array([0,1]),key=30)

# input bar 30
   chr_ischemicheart_1=st.selectbox("Enter 1 if chronic ischemicheart condition exist ",np.array([0,1]),key=31)

# input bar 31
   chr_osteoporasis_0=st.selectbox("Enter 1 if   chronic Osteoporasis condition non-exist ",np.array([0,1]),key=32)

# input bar 32
   chr_osteoporasis_1=st.selectbox("Enter 1 if   chronic Osteoporasis condition exist ",np.array([0,1]),key=33)

# input bar 33
   chr_arthritis_0=st.selectbox(" Enter 1 if chronic rheumatoidarthritis  condition non-exist ",np.array([0,1]),key=34)

# input bar 34
   chr_arthritis_1=st.selectbox(" Enter 1 if chronic rheumatoidarthritis  condition exist ",np.array([0,1]),key=35)

# input bar 35
   chr_stroke_0=st.selectbox("Enter 1 if chronic stroke  condition non-exist ",np.array([0,1]),key=36)

# input bar 36
   chr_stroke_1=st.selectbox("Enter 1 if chronic stroke  condition exist ",np.array([0,1]),key=37)

#input bar 37
   inpann_re=st.number_input("Enter IPAnnualReimbursementAmt",key=38)

#input bar 38
   inpann_ded=st.number_input("Enter IPAnnualDeductibleAmt",key=39)

#input bar 39
   opann_re=st.number_input("Enter OPAnnualReimbursementAmt",key=40)

#input bar 40
   opann_de=st.number_input("Enter OPAnnualDeductibleAmt",key=41)

#input bar 41
   ben_id=st.number_input("Enter Beneficiary ID",key=42)

#input bar 42
   clm_id=st.number_input("Enter Claim ID",key=43)

#input bar 43
   provider=st.number_input("Enter Provider id",key=44)

#input bar 44
   att_phy=st.number_input("Enter attending physician id ",key=45)

#input bar 45
   clm_dur=st.number_input("Enter claim duration",key=46)

#input bar 46
   Age=st.number_input("Enter Age",key=47)


# If button is pressed
   if st.button("Submit"):

# Unpickle classifier
     clf=joblib.load("clf.pkl")

#creating dict with input values
     di={'InscClaimAmtReimbursed':Clm,'DeductibleAmtPaid':Ded,'Is_admitted_0':admit_0,'Is_admitted_1':admit_1,\
         'No_of_days_admitted':day,'Gender_0':gender_0,'Gender_1':gender_1,'Race_0':race_0,'Race_1':race_1,'Race_2':race_2,\
         'RenalDiseaseIndicator_0':Rdi_0,'RenalDiseaseIndicator_1':Rdi_1,'State':state,'County':country,\
         'NoOfMonths_PartACov':part_a,'NoOfMonths_PartBCov':part_b,'ChronicCond_Alzheimer_0':chr_alz_0,'ChronicCond_Alzheimer_1':chr_alz_1,\
         'ChronicCond_Heartfailure_0':chr_heart_0,'ChronicCond_Heartfailure_1':chr_heart_1,'ChronicCond_KidneyDisease_0':chr_kidney_0,'ChronicCond_KidneyDisease_1':chr_kidney_1,\
         'ChronicCond_Cancer_0':chr_cancer_0,'ChronicCond_Cancer_1':chr_cancer_1,'ChronicCond_ObstrPulmonary_0':chr_pulmonary_0,'ChronicCond_ObstrPulmonary_1':chr_pulmonary_1,\
         'ChronicCond_Depression_0':chr_depression_0,'ChronicCond_Depression_1':chr_depression_1,'ChronicCond_Diabetes_0':chr_diabetes_0,\
         'ChronicCond_Diabetes_1':chr_diabetes_1,'ChronicCond_IschemicHeart_0':chr_ischemicheart_0,'ChronicCond_IschemicHeart_1':chr_ischemicheart_1,\
         'ChronicCond_Osteoporasis_0':chr_osteoporasis_0,'ChronicCond_Osteoporasis_1':chr_osteoporasis_1,'ChronicCond_rheumatoidarthritis_0':chr_arthritis_0,\
         'ChronicCond_rheumatoidarthritis_1':chr_arthritis_1,'ChronicCond_stroke_0':chr_stroke_0,'ChronicCond_stroke_1':chr_stroke_1,\
         'IPAnnualReimbursementAmt':inpann_re,'IPAnnualDeductibleAmt':inpann_ded,'OPAnnualReimbursementAmt':opann_re,\
         'OPAnnualDeductibleAmt':opann_de,'BenID':ben_id,'claimID':clm_id,'provider':provider,'Attendingphysician':att_phy,'Claim_Duration':clm_dur,'Age':Age}
     
     ind=np.arange(47)

# Store inputs into dataframe
     X=pd.DataFrame(di,index=ind)
     
     
    

     
     col=['Is_admitted_1','Gender_1','Race_2','RenalDiseaseIndicator_1','ChronicCond_Alzheimer_1',\
     'ChronicCond_Heartfailure_1','ChronicCond_KidneyDisease_1','ChronicCond_Cancer_1',\
     'ChronicCond_ObstrPulmonary_1', 'ChronicCond_Depression_1','ChronicCond_Diabetes_1',\
     'ChronicCond_IschemicHeart_1','ChronicCond_Osteoporasis_1','ChronicCond_rheumatoidarthritis_1',\
     'ChronicCond_stroke_1']

#Removing encoded column as one best represents others.
     X.drop(columns=col,axis=1,inplace=True)
     colu=['InscClaimAmtReimbursed','DeductibleAmtPaid','No_of_days_admitted','NoOfMonths_PartACov',\
          'NoOfMonths_PartBCov','IPAnnualReimbursementAmt','IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',\
          'OPAnnualDeductibleAmt', 'BenID', 'claimID', 'provider','Attendingphysician', 'Claim_Duration', 'Age']

#Normalisation  
     scale=MinMaxScaler()
     scale.fit(X[colu])
     scaled_X=scale.transform(X[colu])
     X[colu]=scaled_X
     

# Get prediction
     prediction=clf['Stack'].predict(X.values)

# Output prediction
     st.text(f"This instance is a {(lambda x : 'Fraud' if(x==1).any() else 'legitimate')(prediction)}")
    
if __name__ == '__main__':
	main()    

 

  




