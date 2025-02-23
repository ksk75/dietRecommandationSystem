from flask import Flask, request, jsonify
import spacy
from spacy.matcher import PhraseMatcher
from flask_cors import CORS
import pandas as pd
import random

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/chatbot": {"origins": "http://127.0.0.1:5500"}})

# Load SpaCy model and set up the matcher
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Load Food Dataset
df = pd.read_csv("/Users/kiruthik/Documents/Machine Learning/dietRecommandationSystem/Tamil_Food_Dataset_15000.csv")

# Define patterns for NER categories
dietary_preferences = ["balanced diet", "immunity boost", "weight gain", "weight loss"]
restrictions = ["Diabetic Friendly"]
health_goals = ["weight loss", "muscle gain", "weight gain", "lose weight"]
allergies = ["nut allergy", "lactose intolerance", "peanut allergy"]

# Add patterns to matcher
matcher.add("DIETARY_PREFERENCE", [nlp.make_doc(text) for text in dietary_preferences])
matcher.add("RESTRICTION", [nlp.make_doc(text) for text in restrictions])
matcher.add("HEALTH_GOAL", [nlp.make_doc(text) for text in health_goals])
matcher.add("ALLERGY", [nlp.make_doc(text) for text in allergies])

# Entity extraction function
def extract_entities(text):
    doc = nlp(text)
    matches = matcher(doc)
    
    entities = {"dietary_preferences": [], "restrictions": [], "health_goals": [], "allergies": []}
    
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        matched_text = doc[start:end].text
        
        if label == "DIETARY_PREFERENCE":
            entities["dietary_preferences"].append(matched_text)
        elif label == "RESTRICTION":
            entities["restrictions"].append(matched_text)
        elif label == "HEALTH_GOAL":
            entities["health_goals"].append(matched_text)
        elif label == "ALLERGY":
            entities["allergies"].append(matched_text)
    
    return entities

# Generate a 1-week meal plan
def generate_meal_plan(entities):
    filtered_df = df.copy()
    
    if entities["dietary_preferences"]:
        pattern = '|'.join(entities["dietary_preferences"])
        filtered_df = filtered_df[filtered_df["Dietary Suitability"].str.contains(pattern, case=False, na=False)]
    
    if entities["health_goals"] and not filtered_df.empty:
        pattern = '|'.join(entities["health_goals"])
        goal_filtered = filtered_df[filtered_df["Dietary Suitability"].str.contains(pattern, case=False, na=False)]
        if not goal_filtered.empty:
            filtered_df = goal_filtered
    
    if entities["restrictions"] and not filtered_df.empty:
        for restriction in entities["restrictions"]:
            filtered_df = filtered_df[~filtered_df["Ingredients"].str.contains(restriction, case=False, na=False)]
    
    if entities["allergies"] and not filtered_df.empty:
        for allergy in entities["allergies"]:
            filtered_df = filtered_df[~filtered_df["Ingredients"].str.contains(allergy, case=False, na=False)]
    
    if filtered_df.empty:
        return {"error": "No suitable meals found. Try adjusting your preferences."}
    
    meal_plan = {}
    
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        meal_plan[day] = {}
        for meal in ["Breakfast", "Lunch", "Dinner"]:
            main_dish = filtered_df[filtered_df["Category"].str.lower() == meal.lower()].sample(1) if not filtered_df[filtered_df["Category"].str.lower() == meal.lower()].empty else None
            side_dish = filtered_df[filtered_df["Category"].str.lower() == "side dish"].sample(1) if not filtered_df[filtered_df["Category"].str.lower() == "side dish"].empty else None
            
            if main_dish is not None and side_dish is not None:
                total_calories = int(main_dish["Calories (per 100g)"].values[0]) + int(side_dish["Calories (per 100g)"].values[0])
                total_protein = float(main_dish["Protein (g)"].values[0]) + float(side_dish["Protein (g)"].values[0])
                
                meal_plan[day][meal] = {
                    "Main Dish": main_dish["Food Name"].values[0],
                    "Side Dish": side_dish["Food Name"].values[0],
                    "Total Calories": total_calories,
                    "Total Protein": total_protein
                }
            else:
                meal_plan[day][meal] = {"error": "No available meal options."}
    
    return meal_plan

# Format meal plan for chatbot response
def format_meal_plan(meal_plan):
    # If there's an error, just wrap it in a paragraph
    if "error" in meal_plan:
        return f"<p>{meal_plan['error']}</p>"

    # Start building HTML
    table_html = "<h3>Here is your 7-day meal plan:</h3>\n"
    table_html += """
    <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
      <thead>
        <tr>
          <th>Day</th>
          <th>Meal</th>
          <th>Main Dish</th>
          <th>Side Dish</th>
          <th>Calories (kcal)</th>
          <th>Protein (g)</th>
        </tr>
      </thead>
      <tbody>
    """

    for day, meals in meal_plan.items():
        for meal, details in meals.items():
            if "error" in details:
                table_html += f"""
                <tr>
                  <td>{day}</td>
                  <td>{meal}</td>
                  <td colspan="4">{details['error']}</td>
                </tr>
                """
            else:
                calories = round(details["Total Calories"], 1)
                protein = round(details["Total Protein"], 1)
                table_html += f"""
                <tr>
                  <td>{day}</td>
                  <td>{meal}</td>
                  <td>{details['Main Dish']}</td>
                  <td>{details['Side Dish']}</td>
                  <td>{calories}</td>
                  <td>{protein}</td>
                </tr>
                """

    table_html += """
      </tbody>
    </table>
    """

    return table_html



# Route to process chatbot input
@app.route('/chatbot', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({'response': "Please provide some input."}), 400
    
    # If user just greets
    if user_input.lower() in ["hi", "hello", "hey"]:
        return jsonify({'response': "Hello! How can I help you with your diet today?"})
    
    # Extract entities
    entities = extract_entities(user_input)
    
    # If no entities recognized, return a prompt instead of a meal plan
    if not any(entities.values()):
        # This means all lists in `entities` are empty
        return jsonify({'response': "I didn't catch any dietary preferences. Could you please specify them?"})
    
    # Otherwise, generate meal plan as before
    meal_plan = generate_meal_plan(entities)
    formatted_response = format_meal_plan(meal_plan)
    
    print("Extracted Entities:", entities)
    print("Generated Meal Plan:", formatted_response)
    
    return jsonify({'response': formatted_response})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
