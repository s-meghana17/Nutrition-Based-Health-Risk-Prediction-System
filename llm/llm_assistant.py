"""
LLM Assistant Module for Nutrition Health System
Using NEW Google GenAI Package (google.genai)
"""

import os
from google import genai
from google.genai import types

# Configure Gemini with new API
def configure_gemini(api_key=None):
    """Configure Gemini API with new google.genai package"""
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY or pass api_key parameter.")
    
    client = genai.Client(api_key=api_key)
    return client


# Universal LLM response function
def get_llm_response(prompt, api_key=None, temperature=0.7):
    """
    Get response from Gemini 2.0 Flash using NEW API
    
    Args:
        prompt: The prompt to send
        api_key: Gemini API key (optional if set in environment)
        temperature: Creativity level (0.0-1.0)
    
    Returns:
        Response text from Gemini
    """
    
    try:
        client = configure_gemini(api_key)
        
        # Generation config
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=config
        )
        
        return response.text
    
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease check your API key and internet connection."


# Feature 1: Basic Chat - FIXED SIGNATURE
def generate_llm_response(user_question, user_context, api_key=None):
    """
    Generate response based on user's health context
    Uses Gemini 2.0 Flash for fast, accurate responses
    
    Args:
        user_question: The user's question
        user_context: Dict with health data
        api_key: Gemini API key (optional)
    """
    
    prompt = f"""You are a friendly, professional nutrition and health AI assistant powered by Gemini.

USER'S HEALTH PROFILE:
- Health Score: {user_context.get('health_score', 'N/A')}/100
- BMI: {user_context.get('bmi', 'N/A')}
- Risk Level: {user_context.get('risk_level', 'N/A')}
- Daily Calories: {user_context.get('calories', 'N/A')} kcal
- Exercise Level: {user_context.get('exercise', 'N/A')}/4

USER QUESTION: {user_question}

Provide a helpful, personalized answer based on their health data. Be:
- Specific and actionable
- Encouraging and supportive
- Evidence-based
- Concise (2-4 paragraphs)

Use emojis where appropriate for better readability."""

    return get_llm_response(prompt, api_key, temperature=0.7)


# Feature 2: Personalized Diet Explanation
def generate_diet_explanation(user_data, health_score, risk_level, api_key=None):
    """Generate comprehensive diet explanation using Gemini 2.0 Flash"""
    
    bmi = user_data.get('bmi', 0)
    bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    
    prompt = f"""You are a certified nutritionist AI. Provide a detailed health analysis.

PERSON'S COMPLETE PROFILE:
👤 Demographics:
- Age: {user_data.get('age')} years
- Gender: {user_data.get('gender')}
- Height: {user_data.get('height')}cm
- Weight: {user_data.get('weight')}kg

📊 Body Metrics:
- BMI: {bmi:.1f} ({bmi_category})
- BMR: {user_data.get('bmr', 'N/A')} kcal/day

🏃 Lifestyle:
- Exercise Level: {user_data.get('exercise')}/4
- Meal Frequency: {user_data.get('meals', 'N/A')} meals/day

🥗 Current Nutrition:
- Daily Calories: {user_data.get('calories')}kcal
- Protein: {user_data.get('proteins')}g
- Carbohydrates: {user_data.get('carbs')}g
- Fats: {user_data.get('fats')}g

🎯 ML PREDICTION RESULTS:
- Health Score: {health_score:.1f}/100
- Risk Classification: {"🚨 HIGH RISK" if risk_level == 1 else "✅ LOW RISK"}

Provide a comprehensive analysis with these sections:

## 📋 What Your Numbers Mean
Explain their health status in simple, clear terms.

## 🔍 Key Findings
Identify 2-3 main insights from their data.

## 🥗 Missing Nutrients
What nutrients they might be lacking based on their intake.

## 💡 3 Quick Wins
Simple changes they can implement TODAY.

## 🎯 Long-term Strategy
What to focus on for sustained health improvement.

## ⚠️ Important Notes
Any concerns or red flags (if applicable).

Be encouraging, specific, and use emojis. Keep it under 500 words total."""

    return get_llm_response(prompt, api_key, temperature=0.6)


# Feature 3: Food Recommendations
def generate_food_recommendations(goal, user_data, dataset_stats=None, api_key=None):
    """Generate smart food recommendations using Gemini"""
    
    dataset_context = ""
    if dataset_stats:
        dataset_context = f"""
📊 INSIGHTS FROM 2,098 REAL USERS:
- Average calories (healthy users): {dataset_stats.get('healthy_avg_calories', 'N/A'):.0f} kcal/day
- Average protein intake: {dataset_stats.get('avg_protein', 'N/A'):.0f}g
- Your calories vs healthy average: {user_data.get('calories', 0) - dataset_stats.get('healthy_avg_calories', 0):+.0f} kcal
"""
    
    prompt = f"""You are a nutrition expert AI. Recommend specific foods for this person's goal.

{dataset_context}

USER'S CURRENT STATUS:
- Daily Calories: {user_data.get('calories', 'N/A')} kcal
- Exercise Level: {user_data.get('exercise', 0)}/4
- BMI: {user_data.get('bmi', 'N/A')}
- Current Protein: {user_data.get('proteins', 'N/A')}g
- Current Carbs: {user_data.get('carbs', 'N/A')}g
- Current Fats: {user_data.get('fats', 'N/A')}g

🎯 GOAL: {goal}

Provide:

## 🥗 Top 7 Recommended Foods
List specific foods with brief reasons why they help this goal.

## 📏 Portion Guidelines
Practical portion sizes for 2-3 key foods.

## 🍽️ Sample Meal
One complete meal example (breakfast, lunch, OR dinner) that supports their goal.

## 💡 Pro Tips
2-3 insider tips for success with this goal.

## ⚠️ Foods to Limit
2-3 foods they should reduce or avoid.

Be specific, practical, and actionable. Keep it under 400 words."""

    return get_llm_response(prompt, api_key, temperature=0.7)


# Feature 4: Meal Plan Generator
def generate_meal_plan(days, calorie_target, dietary_prefs, user_data=None, api_key=None):
    """Generate detailed meal plan using Gemini 2.0 Flash"""
    
    user_context = ""
    if user_data:
        user_context = f"""
USER CONTEXT:
- Current average: {user_data.get('calories', 'N/A')} kcal/day
- BMI: {user_data.get('bmi', 'N/A')}
- Exercise: {user_data.get('exercise', 0)}/4
- Preferences: {dietary_prefs}
"""
    
    prompt = f"""You are a professional meal planning AI. Create a {days}-day meal plan.

{user_context}

REQUIREMENTS:
- Duration: {days} days
- Daily calorie target: {calorie_target} kcal (±100 kcal is acceptable)
- Dietary preferences/restrictions: {dietary_prefs}
- Must be practical, affordable, and delicious
- Balanced macronutrients
- Variety across days

FORMAT (for EACH day):

**Day X: [Theme name]**

🌅 **Breakfast** (~XXX kcal)
- [Meal name and simple description]
- Ingredients: [key ingredients]

🍽️ **Lunch** (~XXX kcal)
- [Meal name and simple description]
- Ingredients: [key ingredients]

🌙 **Dinner** (~XXX kcal)
- [Meal name and simple description]
- Ingredients: [key ingredients]

🥤 **Snack** (~XXX kcal)
- [1-2 snack options]

📊 **Daily Total:** ~XXXX kcal | P: XXg | C: XXg | F: XXg

---

Include simple preparation tips where helpful. Make it realistic and appealing!"""

    return get_llm_response(prompt, api_key, temperature=0.8)


# Feature 5: Health Warnings
def generate_health_warnings(user_data, health_score, risk_level, api_key=None):
    """Generate comprehensive health warnings using Gemini"""
    
    bmi = user_data.get('bmi', 0)
    bmr = user_data.get('bmr', 0)
    calories = user_data.get('calories', 0)
    calorie_balance = calories - bmr
    exercise = user_data.get('exercise', 0)
    
    # Calculate macro percentages
    protein_cal = user_data.get('proteins', 0) * 4
    carbs_cal = user_data.get('carbs', 0) * 4
    fats_cal = user_data.get('fats', 0) * 9
    total_cal = protein_cal + carbs_cal + fats_cal
    
    if total_cal > 0:
        protein_pct = (protein_cal / total_cal) * 100
        carbs_pct = (carbs_cal / total_cal) * 100
        fats_pct = (fats_cal / total_cal) * 100
    else:
        protein_pct = carbs_pct = fats_pct = 0
    
    prompt = f"""You are a health risk assessment AI. Provide a comprehensive risk analysis.

COMPLETE HEALTH ASSESSMENT:

📊 Body Metrics:
- BMI: {bmi:.1f} ({"🚨 UNDERWEIGHT" if bmi<18.5 else "✅ NORMAL" if bmi<25 else "⚠️ OVERWEIGHT" if bmi<30 else "🚨 OBESE"})
- Age: {user_data.get('age')} years
- Gender: {user_data.get('gender')}

⚡ Energy Balance:
- BMR: {bmr:.0f} kcal/day (baseline needs)
- Actual Intake: {calories} kcal/day
- Balance: {calorie_balance:+.0f} kcal ({"🚨 SURPLUS" if calorie_balance>300 else "⚠️ SURPLUS" if calorie_balance>100 else "✅ BALANCED" if abs(calorie_balance)<=100 else "⚠️ DEFICIT" if calorie_balance>-300 else "🚨 DEFICIT"})

🏃 Activity Level:
- Exercise: {exercise}/4 ({"🚨 SEDENTARY" if exercise==0 else "⚠️ LIGHT" if exercise==1 else "✅ MODERATE" if exercise==2 else "✅ ACTIVE"})

🥗 Macro Distribution:
- Protein: {protein_pct:.1f}% (ideal: 15-30%)
- Carbs: {carbs_pct:.1f}% (ideal: 45-65%)
- Fats: {fats_pct:.1f}% (ideal: 20-35%)

🤖 ML ASSESSMENT:
- Health Score: {health_score:.1f}/100
- Risk Classification: {"🚨 HIGH RISK" if risk_level==1 else "✅ LOW RISK"}

Provide a detailed risk report with these sections:

## 🚨 IMMEDIATE WARNINGS
Critical issues requiring immediate attention (if any).

## ⚠️ RISK FACTORS
Concerning patterns in their health data (2-3 key issues).

## 📊 NUTRITIONAL IMBALANCES
Specific macro/calorie issues and their implications.

## 💡 PREVENTIVE ACTIONS
Concrete steps to reduce risk (prioritized list of 3-5 actions).

## 🩺 WHEN TO SEE A DOCTOR
Clear guidance on when professional help is needed.

## ✅ POSITIVE NOTES
Acknowledge what they're doing well (if applicable).

Be direct but supportive. Use clear risk indicators (🚨⚠️✅). Be specific about numbers and actions."""

    return get_llm_response(prompt, api_key, temperature=0.6)


# Feature 6: Dataset Q&A (RAG)
def answer_dataset_question(question, dataset_stats, api_key=None):
    """Answer questions about the dataset using RAG approach"""
    
    prompt = f"""You are a data analyst AI with access to a real nutrition dataset. Answer the user's question with precision.

DATASET STATISTICS (2,098 Real Users):

📊 Overall Metrics:
- Total users analyzed: {dataset_stats.get('total_users', 'N/A')}
- Average age: {dataset_stats.get('avg_age', 'N/A'):.1f} years

🥗 Nutrition Data:
- Average daily calories: {dataset_stats.get('avg_calories', 'N/A'):.0f} kcal
- Calorie range: {dataset_stats.get('min_calories', 'N/A'):.0f} - {dataset_stats.get('max_calories', 'N/A'):.0f} kcal
- Average protein: {dataset_stats.get('avg_protein', 'N/A'):.0f}g
- Maximum protein: {dataset_stats.get('max_protein', 'N/A'):.0f}g
- Average carbs: {dataset_stats.get('avg_carbs', 'N/A'):.0f}g
- Average fats: {dataset_stats.get('avg_fats', 'N/A'):.0f}g

📈 Health Metrics:
- Average BMI: {dataset_stats.get('avg_bmi', 'N/A'):.1f}
- Low Risk users: {dataset_stats.get('low_risk_count', 'N/A')} ({dataset_stats.get('low_risk_pct', 'N/A'):.1f}%)
- High Risk users: {dataset_stats.get('high_risk_count', 'N/A')} ({dataset_stats.get('high_risk_pct', 'N/A'):.1f}%)

✅ Healthy Users (Low Risk) Averages:
- Calories: {dataset_stats.get('healthy_avg_calories', 'N/A'):.0f} kcal
- BMI: {dataset_stats.get('healthy_avg_bmi', 'N/A'):.1f}
- Exercise: {dataset_stats.get('healthy_avg_exercise', 'N/A'):.1f}/4

🚨 High Risk Users Averages:
- Calories: {dataset_stats.get('high_risk_avg_calories', 'N/A'):.0f} kcal
- BMI: {dataset_stats.get('high_risk_avg_bmi', 'N/A'):.1f}
- Exercise: {dataset_stats.get('high_risk_avg_exercise', 'N/A'):.1f}/4

USER QUESTION: {question}

Provide a data-driven answer:
1. Cite specific numbers from the dataset above
2. Make comparisons where relevant
3. Explain patterns or trends
4. Be precise and factual
5. Keep it concise (under 250 words)

Always ground your answer in the actual data provided."""

    return get_llm_response(prompt, api_key, temperature=0.5)


# Utility function to get enhanced dataset stats
def get_enhanced_dataset_stats(df):
    """
    Extract comprehensive statistics from the dataset
    Call this when loading the dataset
    """
    
    try:
        stats = {
            'total_users': len(df),
            'avg_age': df['Age'].mean(),
            'avg_calories': df['Calories'].mean(),
            'min_calories': df['Calories'].min(),
            'max_calories': df['Calories'].max(),
            'avg_protein': df['Proteins'].mean(),
            'max_protein': df['Proteins'].max(),
            'avg_carbs': df['Carbs'].mean(),
            'avg_fats': df['Fats'].mean(),
            'avg_bmi': df['BMI'].mean(),
            'low_risk_count': (df['Health_Risk']==0).sum(),
            'high_risk_count': (df['Health_Risk']==1).sum(),
            'low_risk_pct': (df['Health_Risk']==0).sum() / len(df) * 100,
            'high_risk_pct': (df['Health_Risk']==1).sum() / len(df) * 100,
            
            # Healthy users stats
            'healthy_avg_calories': df[df['Health_Risk']==0]['Calories'].mean(),
            'healthy_avg_bmi': df[df['Health_Risk']==0]['BMI'].mean(),
            'healthy_avg_exercise': df[df['Health_Risk']==0]['Physical exercise'].mean(),
            
            # High risk users stats
            'high_risk_avg_calories': df[df['Health_Risk']==1]['Calories'].mean(),
            'high_risk_avg_bmi': df[df['Health_Risk']==1]['BMI'].mean(),
            'high_risk_avg_exercise': df[df['Health_Risk']==1]['Physical exercise'].mean(),
        }
        
        return stats
    except Exception as e:
        print(f"Error calculating stats: {e}")
        return None