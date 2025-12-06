import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="arifshora/tourism-model", filename="tour_model.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Purchase prediction
st.title("Tourism Package Predictor")
st.write("This is a model that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.")
st.write("Kindly enter the details to check whether a customer will purchase the Wellness Tourism Package.")

# Collect user input
Age = st.number_input(" Age of the customer.",min_value=15,max_value=70,value=30)
CityTier = st.number_input(" The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).",min_value=1,max_value=3,value=2)
NumberOfPersonVisiting = st.number_input(" Total number of people accompanying the customer on the trip.",min_value=1,max_value=10,value=5)
PreferredPropertyStar = st.number_input(" Preferred hotel rating by the customer.",min_value=1,max_value=5,value=5)
NumberOfTrips = st.number_input(" Average number of trips the customer takes annually.",min_value=0,max_value=50,value=10)
NumberOfChildrenVisiting = st.number_input("Number of children below age 5 accompanying the customer.",min_value=0,max_value=5,value=3)
MonthlyIncome = st.number_input(" Gross monthly income of the customer.",min_value=100,max_value=1000000,value=1000)
PitchSatisfactionScore = st.number_input(" Score indicating the customer's satisfaction with the sales pitch.",min_value=1,max_value=5,value=3)
NumberOfFollowups = st.number_input(" Total number of follow-ups by the salesperson after the sales pitch.-",min_value=1,max_value=6,value=3)
DurationOfPitch = st.number_input(" Duration of the sales pitch delivered to the customer.",min_value=0,max_value=240,value=10)

TypeofContact = st.selectbox(" The method by which the customer was contacted (Company Invited or Self Inquiry).",["Self Enquiry","Company Invited"])
Occupation = st.selectbox(" Customer's occupation (e.g., Salaried, Freelancer).",["Salaried","Free Lancer","Small Business","Large Business"])
Gender = st.selectbox(" Gender of the customer (Male, Female).",["Female","Male"])
MaritalStatus = st.selectbox(" Marital status of the customer (Single, Married, Divorced).",["Single","Divorced","Married","Unmarried"])
Passport = st.selectbox(" Whether the customer holds a valid passport",["Yes","No"])
OwnCar = st.selectbox(" Whether the customer owns a car",["Yes","No"])
Designation = st.selectbox(" Customer's designation in their current organization.",["Manager","Executive","Senior Manager","AVP","VP"])
ProductPitched = st.selectbox(" The type of product pitched to the customer.",["Deluxe","Basic","Standard","Super Deluxe","King"])

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    
    'Age': Age,
    'CityTier': CityTier,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch,    
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Passport':  1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'Designation': Designation,
    'ProductPitched': ProductPitched
                        
                            
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase the Package" if prediction == 1 else "NOT purchase the package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
