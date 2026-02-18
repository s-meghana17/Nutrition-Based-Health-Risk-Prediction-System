import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from llm.llm_assistant import generate_llm_response

# Page config
st.set_page_config(
    page_title="AI Health Risk Prediction",
    page_icon="🥗",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

# Load models
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    linear_model = joblib.load(os.path.join(MODEL_DIR, "linear_regression_model.pkl"))
    logistic_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    scaler_log = joblib.load(os.path.join(MODEL_DIR, "scaler_logistic.pkl"))

    try:
        feature_info = joblib.load(os.path.join(MODEL_DIR, "feature_info.pkl"))
        feature_columns = feature_info['feature_columns']
    except:
        feature_columns = ['Gender', 'Age', 'Daily meals frequency', 'Physical exercise',
                          'Height', 'Weight', 'BMI', 'BMR', 'Calories']

    return linear_model, logistic_model, scaler, scaler_log, feature_columns

try:
    linear_model, logistic_model, scaler, scaler_log, feature_columns_linear = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"⚠️ Error loading models: {str(e)}")

feature_columns_logistic = ['Gender', 'Age', 'Daily meals frequency', 'Physical exercise',
                           'Height', 'Weight', 'BMI', 'BMR', 
                           'Carbs', 'Proteins', 'Fats', 'Calories']

# Header
st.title("🥗 AI-Powered Health Risk Prediction System")
st.markdown("### ML Predictions + Intelligent AI Assistant")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🤖 AI Assistant")
    
    api_key = st.text_input(
        "Gemini API Key", 
        type="password",
        help="Get FREE at: https://aistudio.google.com/app/apikey"
    )
    
    if api_key:
        os.environ['GEMINI_API_KEY'] = api_key
        st.success("✅ API Connected!")
    else:
        st.warning("⚠️ Enter API key to use AI chat")
    
    st.markdown("---")
    
    st.header("ℹ️ About")
    st.markdown("""
    **ML Models:**
    - Linear Regression (R²≈0.85)
    - Logistic Regression (Acc≈0.97)
    
    **AI Chat:**
    - Discuss your results
    - Get personalized advice
    - Ask nutrition questions
    """)
    
    st.markdown("---")
    st.markdown("**⚠️ Disclaimer:**")
    st.markdown("Decision-support tool, not medical diagnosis.")

# Main content
if models_loaded:
    st.header("📝 Enter Your Health Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Personal Info")
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender_val = 1 if gender == "Male" else 0
        age = st.number_input("Age (years)", 15, 100, 25)
        height = st.number_input("Height (cm)", 120, 220, 170)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, 0.5)
    
    with col2:
        st.subheader("🍽️ Lifestyle")
        meals = st.selectbox("Daily Meals", [2, 3, 4], index=1)
        exercise = st.slider("Exercise Level (0-4)", 0, 4, 0)
        
        st.markdown("**Exercise Guide:**")
        st.markdown("0️⃣ None | 1️⃣ Light | 2️⃣ Moderate")
        st.markdown("3️⃣ Active | 4️⃣ Very Active")
    
    with col3:
        st.subheader("🥗 Daily Nutrition")
        calories = st.number_input("Calories (kcal)", 1000, 4000, 2000, 50)
        carbs = st.number_input("Carbs (g)", 50, 500, 250, 10)
        proteins = st.number_input("Proteins (g)", 30, 250, 100, 5)
        fats = st.number_input("Fats (g)", 20, 150, 70, 5)
    
    st.markdown("---")
    
    # Calculate
    bmi = weight / ((height / 100) ** 2)
    if gender_val == 1:
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    # Predict button
    if st.button("🔍 Analyze My Health", type="primary", use_container_width=True):
        
        # Prepare inputs
        input_linear = pd.DataFrame({
            'Gender': [gender_val], 'Age': [age], 'Daily meals frequency': [meals],
            'Physical exercise': [exercise], 'Height': [height], 'Weight': [weight],
            'BMI': [bmi], 'BMR': [bmr], 'Calories': [calories]
        })
        
        input_logistic = pd.DataFrame({
            'Gender': [gender_val], 'Age': [age], 'Daily meals frequency': [meals],
            'Physical exercise': [exercise], 'Height': [height], 'Weight': [weight],
            'BMI': [bmi], 'BMR': [bmr], 'Carbs': [carbs], 'Proteins': [proteins],
            'Fats': [fats], 'Calories': [calories]
        })
        
        # Predictions
        input_scaled = scaler.transform(input_linear)
        input_scaled_log = scaler_log.transform(input_logistic)
        
        health_score = np.clip(linear_model.predict(input_scaled)[0], 0, 100)
        health_risk = logistic_model.predict(input_scaled_log)[0]
        risk_proba = logistic_model.predict_proba(input_scaled_log)[0]
        
        # Store data
        st.session_state.user_data = {
            'health_score': health_score,
            'bmi': bmi,
            'bmr': bmr,
            'risk_level': "High" if health_risk == 1 else "Low",
            'risk_probability': risk_proba[1] * 100,
            'calories': calories,
            'carbs': carbs,
            'proteins': proteins,
            'fats': fats,
            'exercise': exercise,
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'meals': meals
        }
        st.session_state.prediction_done = True
        st.session_state.chat_history = []
        
        # Display results
        st.markdown("---")
        st.header("📊 Your Health Analysis")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Health Score", f"{health_score:.1f}/100")
        col2.metric("⚠️ Risk Level", "🔴 HIGH" if health_risk == 1 else "🟢 LOW")
        col3.metric("📈 Risk Probability", f"{risk_proba[1]*100:.1f}%")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={'text': "Health Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': '#ffcccc'},
                    {'range': [50, 75], 'color': '#ffffcc'},
                    {'range': [75, 100], 'color': '#ccffcc'}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'value': 84.5}}))
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Your Metrics")
            bmi_cat = "Underweight" if bmi<18.5 else "Normal" if bmi<25 else "Overweight" if bmi<30 else "Obese"
            st.markdown(f"**BMI:** {bmi:.2f} ({bmi_cat})")
            st.markdown(f"**BMR:** {bmr:.0f} kcal/day")
            st.markdown(f"**Calorie Balance:** {calories-bmr:+.0f} kcal")
        
        with col2:
            st.subheader("🥗 Macros")
            p_cal = proteins * 4
            c_cal = carbs * 4
            f_cal = fats * 9
            total = p_cal + c_cal + f_cal
            st.markdown(f"**Protein:** {p_cal/total*100:.1f}%")
            st.markdown(f"**Carbs:** {c_cal/total*100:.1f}%")
            st.markdown(f"**Fats:** {f_cal/total*100:.1f}%")
        
        # Macro pie chart
        st.markdown("---")
        macro_df = pd.DataFrame({
            'Macro': ['Protein', 'Carbs', 'Fats'],
            'Percentage': [p_cal/total*100, c_cal/total*100, f_cal/total*100]
        })
        fig3 = px.pie(macro_df, values='Percentage', names='Macro', hole=0.4,
                     color_discrete_map={'Protein':'#ff6b6b','Carbs':'#4ecdc4','Fats':'#ffe66d'})
        st.plotly_chart(fig3, use_container_width=True)
    
    # AI CHAT
    if st.session_state.prediction_done:
        st.markdown("---")
        st.header("💬 Chat with AI About Your Results")
        
        if not api_key:
            st.warning("⚠️ Enter Gemini API key in sidebar to chat!")
            st.info("Get FREE key: https://aistudio.google.com/app/apikey")
        else:
            # Chat history
            for speaker, message in st.session_state.chat_history:
                with st.chat_message("user" if speaker == "You" else "assistant"):
                    st.markdown(message)
            
            # Chat input
            if prompt := st.chat_input("Ask about your results..."):
                st.session_state.chat_history.append(("You", prompt))
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("AI thinking..."):
                        try:
                            response = generate_llm_response(
                                prompt,
                                st.session_state.user_data,
                                api_key
                            )
                            st.markdown(response)
                            st.session_state.chat_history.append(("AI", response))
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            # Suggested questions
            st.markdown("**💡 Quick Questions:**")
            col1, col2, col3 = st.columns(3)
            
            if col1.button("Why this score?"):
                st.session_state.chat_history.append(("You", "Why is my health score this value?"))
                st.rerun()
            
            if col2.button("What to improve?"):
                st.session_state.chat_history.append(("You", "What should I improve first?"))
                st.rerun()
            
            if col3.button("Meal suggestions?"):
                st.session_state.chat_history.append(("You", "Give me meal suggestions"))
                st.rerun()

else:
    st.error("⚠️ Models not loaded!")
    st.info("Run: `python retrain_models.py` or the training notebook")

st.markdown("---")
st.markdown("**🤖 Powered by ML + Gemini AI**")













# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import plotly.graph_objects as go
# import plotly.express as px
# import sys
# import os

# # Fix import path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

# from llm.llm_assistant import generate_llm_response

# # Page config
# st.set_page_config(
#     page_title="AI Health Risk Prediction",
#     page_icon="🥗",
#     layout="wide"
# )

# # Initialize session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'prediction_done' not in st.session_state:
#     st.session_state.prediction_done = False
# if 'user_data' not in st.session_state:
#     st.session_state.user_data = {}

# # Load models
# @st.cache_resource
# def load_models():
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     MODEL_DIR = os.path.join(BASE_DIR, "models")

#     linear_model = joblib.load(os.path.join(MODEL_DIR, "linear_regression_model.pkl"))
#     logistic_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
#     scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
#     scaler_log = joblib.load(os.path.join(MODEL_DIR, "scaler_logistic.pkl"))

#     try:
#         feature_info = joblib.load(os.path.join(MODEL_DIR, "feature_info.pkl"))
#         feature_columns = feature_info['feature_columns']
#     except:
#         feature_columns = ['Gender', 'Age', 'Daily meals frequency', 'Physical exercise',
#                           'Height', 'Weight', 'BMI', 'BMR', 'Calories']

#     return linear_model, logistic_model, scaler, scaler_log, feature_columns

# try:
#     linear_model, logistic_model, scaler, scaler_log, feature_columns_linear = load_models()
#     models_loaded = True
# except Exception as e:
#     models_loaded = False
#     st.error(f"⚠️ Error loading models: {str(e)}")

# feature_columns_logistic = ['Gender', 'Age', 'Daily meals frequency', 'Physical exercise',
#                            'Height', 'Weight', 'BMI', 'BMR', 
#                            'Carbs', 'Proteins', 'Fats', 'Calories']

# # Header
# st.title("🥗 AI-Powered Health Risk Prediction System")
# st.markdown("### Get Predictions + Chat with AI About Your Results")
# st.markdown("---")

# # Sidebar
# with st.sidebar:
#     st.header("🤖 AI Assistant Settings")
    
#     api_key = st.text_input(
#         "Gemini API Key", 
#         type="password",
#         help="Get FREE key at: https://aistudio.google.com/app/apikey"
#     )
    
#     if api_key:
#         os.environ['GEMINI_API_KEY'] = api_key
#         st.success("✅ API Key Set!")
#     else:
#         st.warning("⚠️ Enter API key to chat with AI")
    
#     st.markdown("---")
    
#     st.header("ℹ️ About")
#     st.markdown("""
#     **ML Models:**
#     - Linear Regression (R² = 84.9%)
#     - Logistic Regression (Acc = 97.1%)
    
#     **AI Features:**
#     - 🤖 Chat about your results
#     - 💡 Get personalized advice
#     - 📊 Understand your metrics
#     """)
    
#     st.markdown("---")
#     st.markdown("**⚠️ Disclaimer:**")
#     st.markdown("Decision-support tool, not medical diagnosis.")

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
#         exercise = st.slider("Physical Exercise Level", 0, 4, 0)
        
#         st.markdown("**Exercise Levels:**")
#         st.markdown("- 0️⃣ Sedentary")
#         st.markdown("- 1️⃣ Light")
#         st.markdown("- 2️⃣ Moderate")
#         st.markdown("- 3️⃣ Active")
#         st.markdown("- 4️⃣ Very Active")
    
#     with col3:
#         st.subheader("🥗 Daily Nutrition")
#         calories = st.number_input("Calories (kcal)", min_value=1000, max_value=4000, value=2000, step=50)
#         carbs = st.number_input("Carbohydrates (g)", min_value=50, max_value=500, value=250, step=10)
#         proteins = st.number_input("Proteins (g)", min_value=30, max_value=250, value=100, step=5)
#         fats = st.number_input("Fats (g)", min_value=20, max_value=150, value=70, step=5)
    
#     st.markdown("---")
    
#     # Calculate BMI and BMR
#     bmi = weight / ((height / 100) ** 2)
    
#     if gender_val == 1:
#         bmr = 10 * weight + 6.25 * height - 5 * age + 5
#     else:
#         bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
#     # Predict button
#     if st.button("🔍 Analyze My Health", type="primary", use_container_width=True):
        
#         # Prepare inputs
#         input_data_linear = pd.DataFrame({
#             'Gender': [gender_val],
#             'Age': [age],
#             'Daily meals frequency': [meals],
#             'Physical exercise': [exercise],
#             'Height': [height],
#             'Weight': [weight],
#             'BMI': [bmi],
#             'BMR': [bmr],
#             'Calories': [calories]
#         })
        
#         input_data_logistic = pd.DataFrame({
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
        
#         # Predictions
#         input_scaled = scaler.transform(input_data_linear)
#         input_scaled_log = scaler_log.transform(input_data_logistic)
        
#         health_score = np.clip(linear_model.predict(input_scaled)[0], 0, 100)
#         health_risk = logistic_model.predict(input_scaled_log)[0]
#         risk_proba = logistic_model.predict_proba(input_scaled_log)[0]
        
#         # Store in session
#         st.session_state.user_data = {
#             'health_score': health_score,
#             'bmi': bmi,
#             'bmr': bmr,
#             'risk_level': "High" if health_risk == 1 else "Low",
#             'risk_probability': risk_proba[1] * 100,
#             'calories': calories,
#             'carbs': carbs,
#             'proteins': proteins,
#             'fats': fats,
#             'exercise': exercise,
#             'age': age,
#             'gender': gender,
#             'weight': weight,
#             'height': height,
#             'meals': meals
#         }
#         st.session_state.prediction_done = True
#         st.session_state.chat_history = []  # Reset chat on new prediction
        
#         # Display results
#         st.markdown("---")
#         st.header("📊 Your Health Analysis Results")
        
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.metric("🎯 Health Score", f"{health_score:.1f}/100")
#         with col2:
#             risk_label = "🔴 HIGH RISK" if health_risk == 1 else "🟢 LOW RISK"
#             st.metric("⚠️ Risk Level", risk_label)
#         with col3:
#             st.metric("📈 Risk Probability", f"{risk_proba[1]*100:.1f}%")
        
#         # Gauge chart
#         st.markdown("### 🎯 Health Score Visualization")
        
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number+delta",
#             value=health_score,
#             domain={'x': [0, 1], 'y': [0, 1]},
#             title={'text': "Health Score", 'font': {'size': 24}},
#             delta={'reference': 84.5, 'increasing': {'color': "green"}},
#             gauge={
#                 'axis': {'range': [None, 100]},
#                 'bar': {'color': "darkblue"},
#                 'steps': [
#                     {'range': [0, 50], 'color': '#ffcccc'},
#                     {'range': [50, 75], 'color': '#ffffcc'},
#                     {'range': [75, 100], 'color': '#ccffcc'}],
#                 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 84.5}}))
        
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Risk probability bar
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
        
#         # Metrics
#         st.markdown("---")
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
#             st.subheader("🥗 Macro Distribution")
#             protein_cal = proteins * 4
#             carbs_cal = carbs * 4
#             fats_cal = fats * 9
#             total_cal = protein_cal + carbs_cal + fats_cal
            
#             st.markdown(f"**Protein:** {protein_cal/total_cal*100:.1f}% (ideal: 15-30%)")
#             st.markdown(f"**Carbs:** {carbs_cal/total_cal*100:.1f}% (ideal: 45-65%)")
#             st.markdown(f"**Fats:** {fats_cal/total_cal*100:.1f}% (ideal: 20-35%)")
        
#         # Macro pie chart
#         st.markdown("---")
#         st.subheader("📊 Macro Breakdown")
        
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
    
#     # AI CHAT SECTION (Shows after prediction)
#     if st.session_state.prediction_done:
#         st.markdown("---")
#         st.header("💬 Chat with AI About Your Results")
        
#         if not api_key:
#             st.warning("⚠️ Please enter your Gemini API key in the sidebar to chat with AI!")
#             st.info("👈 Get a FREE API key at: https://aistudio.google.com/app/apikey")
#         else:
#             st.markdown("Ask me anything about your health analysis, nutrition, or fitness goals!")
            
#             # Display chat history
#             for i, (speaker, message) in enumerate(st.session_state.chat_history):
#                 if speaker == "You":
#                     with st.chat_message("user"):
#                         st.markdown(message)
#                 else:
#                     with st.chat_message("assistant", avatar="🤖"):
#                         st.markdown(message)
            
#             # Chat input
#             if prompt := st.chat_input("Ask me anything about your results..."):
#                 # Add user message
#                 st.session_state.chat_history.append(("You", prompt))
                
#                 with st.chat_message("user"):
#                     st.markdown(prompt)
                
#                 # Generate AI response
#                 with st.chat_message("assistant", avatar="🤖"):
#                     with st.spinner("AI is thinking..."):
#                         try:
#                             response = generate_llm_response(
#                                 prompt,
#                                 st.session_state.user_data,
#                                 api_key
#                             )
#                             st.markdown(response)
#                             st.session_state.chat_history.append(("AI", response))
#                         except Exception as e:
#                             error_msg = f"Error: {str(e)}"
#                             st.error(error_msg)
#                             st.session_state.chat_history.append(("AI", error_msg))
            
#             # Suggested questions
#             st.markdown("---")
#             st.markdown("**💡 Suggested Questions:**")
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 if st.button("Why is my health score this value?"):
#                     st.session_state.chat_history.append(("You", "Why is my health score this value?"))
#                     st.rerun()
            
#             with col2:
#                 if st.button("What should I improve first?"):
#                     st.session_state.chat_history.append(("You", "What should I improve first?"))
#                     st.rerun()
            
#             with col3:
#                 if st.button("Give me a meal plan"):
#                     st.session_state.chat_history.append(("You", "Can you suggest a meal plan for me?"))
#                     st.rerun()
            
#             col4, col5, col6 = st.columns(3)
            
#             with col4:
#                 if st.button("How much should I exercise?"):
#                     st.session_state.chat_history.append(("You", "How much should I exercise?"))
#                     st.rerun()
            
#             with col5:
#                 if st.button("Is my BMI healthy?"):
#                     st.session_state.chat_history.append(("You", "Is my BMI healthy?"))
#                     st.rerun()
            
#             with col6:
#                 if st.button("Food recommendations?"):
#                     st.session_state.chat_history.append(("You", "What foods do you recommend?"))
#                     st.rerun()

# else:
#     st.error("⚠️ Models not loaded. Please retrain models.")
#     st.info("Run: `python retrain_models.py`")

# # Footer
# st.markdown("---")
# st.markdown("**🤖 Powered by ML + Google Gemini AI | Built with ❤️ using Streamlit**")




















