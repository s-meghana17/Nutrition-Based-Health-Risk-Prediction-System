import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Health Risk Prediction System",
    page_icon="🥗",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    linear_model = joblib.load('../models/linear_regression_model.pkl')
    logistic_model = joblib.load('../models/logistic_regression_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    scaler_log = joblib.load('../models/scaler_logistic.pkl')
    try:
        feature_info = joblib.load('../models/feature_info.pkl')
        feature_columns = feature_info['feature_columns']
    except:
        # Fallback to fixed feature list
        feature_columns = ['Gender', 'Age', 'Daily meals frequency', 'Physical exercise',
                          'Height', 'Weight', 'BMI', 'BMR', 'Calories']
    return linear_model, logistic_model, scaler, scaler_log, feature_columns

try:
    linear_model, logistic_model, scaler, scaler_log, feature_columns_linear = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"⚠️ Error loading models: {str(e)}")

# Logistic features (original set with macros)
feature_columns_logistic = ['Gender', 'Age', 'Daily meals frequency', 'Physical exercise',
                           'Height', 'Weight', 'BMI', 'BMR', 
                           'Carbs', 'Proteins', 'Fats', 'Calories']

# Header
st.title("🥗 Nutrition-Based Health Risk Prediction System")
st.markdown("### AI-Powered Health Assessment Tool")
st.markdown("---")

# Sidebar - Info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This system uses **Machine Learning** to predict:
    - 📊 **Health Score** (0-100)
    - ⚠️ **Health Risk Level** (Low/High)
    
    Based on your nutrition and lifestyle data.
    
    **Models Used:**
    - Linear Regression (R² = 84.9%)
    - Logistic Regression (Accuracy = 97.1%)
    """)
    
    st.markdown("---")
    st.markdown("**⚠️ Disclaimer:**")
    st.markdown("This is a decision-support tool, NOT medical diagnosis. Consult healthcare professionals for medical advice.")

# Main content
if models_loaded:
    # Input section
    st.header("📝 Enter Your Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Personal Info")
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender_val = 1 if gender == "Male" else 0
        age = st.number_input("Age (years)", min_value=15, max_value=100, value=25)
        height = st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    
    with col2:
        st.subheader("🍽️ Lifestyle")
        meals = st.selectbox("Daily Meal Frequency", [2, 3, 4], index=1)
        exercise = st.slider("Physical Exercise Level", 0, 4, 0, 
                           help="0=None, 1=Light, 2=Moderate, 3=Active, 4=Very Active")
        
        st.markdown("**Exercise Levels:**")
        st.markdown("- 0️⃣ Sedentary")
        st.markdown("- 1️⃣ Light (1-2 days/week)")
        st.markdown("- 2️⃣ Moderate (3-4 days/week)")
        st.markdown("- 3️⃣ Active (5-6 days/week)")
        st.markdown("- 4️⃣ Very Active (daily)")
    
    with col3:
        st.subheader("🥗 Daily Nutrition")
        calories = st.number_input("Calories (kcal)", min_value=1000, max_value=4000, value=2000, step=50)
        carbs = st.number_input("Carbohydrates (g)", min_value=50, max_value=500, value=250, step=10)
        proteins = st.number_input("Proteins (g)", min_value=30, max_value=250, value=100, step=5)
        fats = st.number_input("Fats (g)", min_value=20, max_value=150, value=70, step=5)
    
    st.markdown("---")
    
    # Calculate BMI and BMR
    bmi = weight / ((height / 100) ** 2)
    
    # BMR calculation (Mifflin-St Jeor Equation)
    if gender_val == 1:  # Male
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:  # Female
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    # Predict button
    if st.button("🔍 Predict Health Risk", type="primary", use_container_width=True):
        
        # Prepare input for LINEAR regression (without individual macros)
        input_data_linear = pd.DataFrame({
            'Gender': [gender_val],
            'Age': [age],
            'Daily meals frequency': [meals],
            'Physical exercise': [exercise],
            'Height': [height],
            'Weight': [weight],
            'BMI': [bmi],
            'BMR': [bmr],
            'Calories': [calories]
        })
        
        # Prepare input for LOGISTIC regression (with macros)
        input_data_logistic = pd.DataFrame({
            'Gender': [gender_val],
            'Age': [age],
            'Daily meals frequency': [meals],
            'Physical exercise': [exercise],
            'Height': [height],
            'Weight': [weight],
            'BMI': [bmi],
            'BMR': [bmr],
            'Carbs': [carbs],
            'Proteins': [proteins],
            'Fats': [fats],
            'Calories': [calories]
        })
        
        # Scale and predict
        input_scaled = scaler.transform(input_data_linear)
        input_scaled_log = scaler_log.transform(input_data_logistic)
        
        health_score = linear_model.predict(input_scaled)[0]
        health_risk = logistic_model.predict(input_scaled_log)[0]
        risk_proba = logistic_model.predict_proba(input_scaled_log)[0]
        
        # Clip health score to valid range
        health_score = np.clip(health_score, 0, 100)
        
        # Display results
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Health Score", f"{health_score:.1f}/100")
            
        with col2:
            risk_label = "🔴 HIGH RISK" if health_risk == 1 else "🟢 LOW RISK"
            st.metric("⚠️ Risk Level", risk_label)
        
        with col3:
            risk_percentage = risk_proba[1] * 100
            st.metric("📈 Risk Probability", f"{risk_percentage:.1f}%")
        
        # Gauge chart for health score
        st.markdown("### 🎯 Health Score Visualization")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = health_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Health Score", 'font': {'size': 24}},
            delta = {'reference': 84.5, 'increasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#ffcccc'},
                    {'range': [50, 75], 'color': '#ffffcc'},
                    {'range': [75, 100], 'color': '#ccffcc'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 84.5}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk probability bar
        st.markdown("### 📊 Risk Probability Breakdown")
        
        prob_df = pd.DataFrame({
            'Risk Level': ['Low Risk', 'High Risk'],
            'Probability': [risk_proba[0] * 100, risk_proba[1] * 100]
        })
        
        fig2 = px.bar(prob_df, x='Risk Level', y='Probability', 
                     color='Risk Level',
                     color_discrete_map={'Low Risk': 'green', 'High Risk': 'red'},
                     text='Probability')
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Interpretation
        st.markdown("---")
        st.header("💡 Interpretation & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Your Metrics")
            st.markdown(f"**BMI:** {bmi:.2f}")
            bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
            st.markdown(f"**BMI Category:** {bmi_category}")
            st.markdown(f"**BMR:** {bmr:.0f} kcal/day")
            calorie_balance = calories - bmr
            st.markdown(f"**Calorie Balance:** {calorie_balance:+.0f} kcal")
        
        with col2:
            st.subheader("💪 Recommendations")
            
            if health_risk == 1:
                st.warning("⚠️ **High Risk Detected**")
                st.markdown("**Suggestions:**")
                if bmi > 25:
                    st.markdown("- 🏃 Focus on weight management")
                if exercise < 2:
                    st.markdown("- 💪 Increase physical activity")
                if abs(calorie_balance) > 300:
                    st.markdown("- 🍽️ Balance calorie intake with BMR")
                st.markdown("- 🩺 Consider consulting a healthcare professional")
            else:
                st.success("✅ **Low Risk - Keep it up!**")
                st.markdown("**Maintain your healthy habits:**")
                st.markdown("- ✅ Continue balanced nutrition")
                st.markdown("- ✅ Stay physically active")
                st.markdown("- ✅ Regular health check-ups")
        
        # Macro breakdown
        st.markdown("---")
        st.subheader("🥗 Your Macro Distribution")
        
        protein_cal = proteins * 4
        carbs_cal = carbs * 4
        fats_cal = fats * 9
        total_cal = protein_cal + carbs_cal + fats_cal
        
        macro_df = pd.DataFrame({
            'Macro': ['Protein', 'Carbs', 'Fats'],
            'Percentage': [
                protein_cal / total_cal * 100,
                carbs_cal / total_cal * 100,
                fats_cal / total_cal * 100
            ]
        })
        
        fig3 = px.pie(macro_df, values='Percentage', names='Macro',
                     color='Macro',
                     color_discrete_map={'Protein': '#ff6b6b', 
                                       'Carbs': '#4ecdc4', 
                                       'Fats': '#ffe66d'},
                     hole=0.4)
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("**Ideal Ranges:**")
        st.markdown("- Protein: 15-30% | Carbs: 45-65% | Fats: 20-35%")

else:
    st.warning("⚠️ Please train the models first by running the notebook scripts.")
    st.info("📝 Run the fixed model training script to regenerate models.")

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit | Machine Learning Project**")










# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import plotly.graph_objects as go
# import plotly.express as px

# # Page config
# st.set_page_config(
#     page_title="Health Risk Prediction System",
#     page_icon="🥗",
#     layout="wide"
# )

# # Load models
# @st.cache_resource
# def load_models():
#     linear_model = joblib.load('../models/linear_regression_model.pkl')
#     logistic_model = joblib.load('../models/logistic_regression_model.pkl')
#     scaler = joblib.load('../models/scaler.pkl')
#     scaler_log = joblib.load('../models/scaler_logistic.pkl')
#     return linear_model, logistic_model, scaler, scaler_log

# try:
#     linear_model, logistic_model, scaler, scaler_log = load_models()
#     models_loaded = True
# except:
#     models_loaded = False
#     st.error("⚠️ Models not found! Please train the models first.")

# # Header
# st.title("🥗 Nutrition-Based Health Risk Prediction System")
# st.markdown("### AI-Powered Health Assessment Tool")
# st.markdown("---")

# # Sidebar - Info
# with st.sidebar:
#     st.header("ℹ️ About")
#     st.markdown("""
#     This system uses **Machine Learning** to predict:
#     - 📊 **Health Score** (0-100)
#     - ⚠️ **Health Risk Level** (Low/High)
    
#     Based on your nutrition and lifestyle data.
    
#     **Models Used:**
#     - Linear Regression (R² = 84.9%)
#     - Logistic Regression (Accuracy = 97.1%)
#     """)
    
#     st.markdown("---")
#     st.markdown("**⚠️ Disclaimer:**")
#     st.markdown("This is a decision-support tool, NOT medical diagnosis. Consult healthcare professionals for medical advice.")

# # Main content
# if models_loaded:
#     # Input section
#     st.header("📝 Enter Your Information")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.subheader("👤 Personal Info")
#         gender = st.selectbox("Gender", ["Female", "Male"])
#         gender_val = 1 if gender == "Male" else 0
#         age = st.number_input("Age (years)", min_value=15, max_value=100, value=25)
#         height = st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
#         weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    
#     with col2:
#         st.subheader("🍽️ Lifestyle")
#         meals = st.selectbox("Daily Meal Frequency", [2, 3, 4], index=1)
#         exercise = st.slider("Physical Exercise Level", 0, 4, 0, 
#                            help="0=None, 1=Light, 2=Moderate, 3=Active, 4=Very Active")
        
#         st.markdown("**Exercise Levels:**")
#         st.markdown("- 0️⃣ Sedentary")
#         st.markdown("- 1️⃣ Light (1-2 days/week)")
#         st.markdown("- 2️⃣ Moderate (3-4 days/week)")
#         st.markdown("- 3️⃣ Active (5-6 days/week)")
#         st.markdown("- 4️⃣ Very Active (daily)")
    
#     with col3:
#         st.subheader("🥗 Daily Nutrition")
#         calories = st.number_input("Calories (kcal)", min_value=1000, max_value=4000, value=2000, step=50)
#         carbs = st.number_input("Carbohydrates (g)", min_value=50, max_value=500, value=250, step=10)
#         proteins = st.number_input("Proteins (g)", min_value=30, max_value=250, value=100, step=5)
#         fats = st.number_input("Fats (g)", min_value=20, max_value=150, value=70, step=5)
    
#     st.markdown("---")
    
#     # Calculate BMI and BMR
#     bmi = weight / ((height / 100) ** 2)
    
#     # BMR calculation (Mifflin-St Jeor Equation)
#     if gender_val == 1:  # Male
#         bmr = 10 * weight + 6.25 * height - 5 * age + 5
#     else:  # Female
#         bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
#     # Predict button
#     if st.button("🔍 Predict Health Risk", type="primary", use_container_width=True):
        
#         # Prepare input
#         input_data = pd.DataFrame({
#             'Gender': [gender_val],
#             'Age': [age],
#             'Daily meals frequency': [meals],
#             'Physical exercise': [exercise],
#             'Height': [height],
#             'Weight': [weight],
#             'BMI': [bmi],
#             'BMR': [bmr],
#             'Carbs': [carbs],
#             'Proteins': [proteins],
#             'Fats': [fats],
#             'Calories': [calories]
#         })
        
#         # Scale and predict
#         input_scaled = scaler.transform(input_data)
#         input_scaled_log = scaler_log.transform(input_data)
        
#         health_score = linear_model.predict(input_scaled)[0]
#         health_risk = logistic_model.predict(input_scaled_log)[0]
#         risk_proba = logistic_model.predict_proba(input_scaled_log)[0]
        
#         # Display results
#         st.markdown("---")
#         st.header("📊 Prediction Results")
        
#         # Metrics row
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.metric("🎯 Health Score", f"{health_score:.1f}/100")
            
#         with col2:
#             risk_label = "🔴 HIGH RISK" if health_risk == 1 else "🟢 LOW RISK"
#             risk_color = "red" if health_risk == 1 else "green"
#             st.metric("⚠️ Risk Level", risk_label)
        
#         with col3:
#             risk_percentage = risk_proba[1] * 100
#             st.metric("📈 Risk Probability", f"{risk_percentage:.1f}%")
        
#         # Gauge chart for health score
#         st.markdown("### 🎯 Health Score Visualization")
        
#         fig = go.Figure(go.Indicator(
#             mode = "gauge+number+delta",
#             value = health_score,
#             domain = {'x': [0, 1], 'y': [0, 1]},
#             title = {'text': "Health Score", 'font': {'size': 24}},
#             delta = {'reference': 84.5, 'increasing': {'color': "green"}},
#             gauge = {
#                 'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
#                 'bar': {'color': "darkblue"},
#                 'bgcolor': "white",
#                 'borderwidth': 2,
#                 'bordercolor': "gray",
#                 'steps': [
#                     {'range': [0, 50], 'color': '#ffcccc'},
#                     {'range': [50, 75], 'color': '#ffffcc'},
#                     {'range': [75, 100], 'color': '#ccffcc'}],
#                 'threshold': {
#                     'line': {'color': "red", 'width': 4},
#                     'thickness': 0.75,
#                     'value': 84.5}}))
        
#         fig.update_layout(height=300)
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Risk probability bar
#         st.markdown("### 📊 Risk Probability Breakdown")
        
#         prob_df = pd.DataFrame({
#             'Risk Level': ['Low Risk', 'High Risk'],
#             'Probability': [risk_proba[0] * 100, risk_proba[1] * 100]
#         })
        
#         fig2 = px.bar(prob_df, x='Risk Level', y='Probability', 
#                      color='Risk Level',
#                      color_discrete_map={'Low Risk': 'green', 'High Risk': 'red'},
#                      text='Probability')
#         fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
#         fig2.update_layout(showlegend=False, height=300)
#         st.plotly_chart(fig2, use_container_width=True)
        
#         # Interpretation
#         st.markdown("---")
#         st.header("💡 Interpretation & Recommendations")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("📋 Your Metrics")
#             st.markdown(f"**BMI:** {bmi:.2f}")
#             bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
#             st.markdown(f"**BMI Category:** {bmi_category}")
#             st.markdown(f"**BMR:** {bmr:.0f} kcal/day")
#             calorie_balance = calories - bmr
#             st.markdown(f"**Calorie Balance:** {calorie_balance:+.0f} kcal")
        
#         with col2:
#             st.subheader("💪 Recommendations")
            
#             if health_risk == 1:
#                 st.warning("⚠️ **High Risk Detected**")
#                 st.markdown("**Suggestions:**")
#                 if bmi > 25:
#                     st.markdown("- 🏃 Focus on weight management")
#                 if exercise < 2:
#                     st.markdown("- 💪 Increase physical activity")
#                 if abs(calorie_balance) > 300:
#                     st.markdown("- 🍽️ Balance calorie intake with BMR")
#                 st.markdown("- 🩺 Consider consulting a healthcare professional")
#             else:
#                 st.success("✅ **Low Risk - Keep it up!**")
#                 st.markdown("**Maintain your healthy habits:**")
#                 st.markdown("- ✅ Continue balanced nutrition")
#                 st.markdown("- ✅ Stay physically active")
#                 st.markdown("- ✅ Regular health check-ups")
        
#         # Macro breakdown
#         st.markdown("---")
#         st.subheader("🥗 Your Macro Distribution")
        
#         protein_cal = proteins * 4
#         carbs_cal = carbs * 4
#         fats_cal = fats * 9
#         total_cal = protein_cal + carbs_cal + fats_cal
        
#         macro_df = pd.DataFrame({
#             'Macro': ['Protein', 'Carbs', 'Fats'],
#             'Percentage': [
#                 protein_cal / total_cal * 100,
#                 carbs_cal / total_cal * 100,
#                 fats_cal / total_cal * 100
#             ]
#         })
        
#         fig3 = px.pie(macro_df, values='Percentage', names='Macro',
#                      color='Macro',
#                      color_discrete_map={'Protein': '#ff6b6b', 
#                                        'Carbs': '#4ecdc4', 
#                                        'Fats': '#ffe66d'},
#                      hole=0.4)
#         fig3.update_traces(textposition='inside', textinfo='percent+label')
#         st.plotly_chart(fig3, use_container_width=True)
        
#         st.markdown("**Ideal Ranges:**")
#         st.markdown("- Protein: 15-30% | Carbs: 45-65% | Fats: 20-35%")

# else:
#     st.warning("⚠️ Please train the models first by running the notebook scripts.")
#     st.info("📝 Run steps 1-5 in the notebooks folder to generate the models.")

# # Footer
# st.markdown("---")
# st.markdown("**Built with ❤️ using Streamlit | Machine Learning Project**")