from flask import Flask, request, jsonify
import spacy
from spacy.matcher import PhraseMatcher
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/chatbot": {"origins": "http://127.0.0.1:5500"}})

# Load SpaCy model and set up the matcher
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Define patterns for NER categories
dietary_preferences = ["vegetarian", "vegan", "pescatarian"]
restrictions = ["gluten-free", "dairy-free", "sugar-free"]
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

# Route to process chatbot input
@app.route('/chatbot', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({'response': "Please provide some input."}), 400
    
    # Extract entities and generate response
    entities = extract_entities(user_input)
    response = generate_response(entities)
    
    return jsonify({'response': response})

# Response generation function
def generate_response(entities):
    response_parts = []
    
    if entities["dietary_preferences"]:
        response_parts.append(f"Noted your dietary preference: {', '.join(entities['dietary_preferences'])}.")
    if entities["restrictions"]:
        response_parts.append(f"I'll make sure to consider your restrictions: {', '.join(entities['restrictions'])}.")
    if entities["health_goals"]:
        response_parts.append(f"Your health goal is {', '.join(entities['health_goals'])}.")
    if entities["allergies"]:
        response_parts.append(f"I'll be cautious about your allergy: {', '.join(entities['allergies'])}.")
    
    if not response_parts:
        return "Could you provide more details about your dietary preferences or goals?"
    
    return " ".join(response_parts)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
