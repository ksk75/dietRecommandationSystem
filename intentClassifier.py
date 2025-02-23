import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Load pre-trained LLaMA model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")  # Adjust model name if needed
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf", num_labels=5)  # Change num_labels as per your intents


  # Extended dataset with diverse intents
data = [
    # Diet Recommendation Intents (Class 0)
    ("Recommend a meal plan for keto beginners", 0)
    ("What’s a good low-sugar diet plan?", 0)
    ("I need a balanced diet for overall health", 0)
    ("Suggest a diet for reducing body fat", 0)
    ("Can you provide a low-calorie meal plan?", 0)
    ("I’m looking for a protein-rich vegetarian diet", 0)
    ("What’s a good Mediterranean meal plan?", 0)
    ("I need a meal plan for quick muscle recovery", 0)
    ("Suggest a diet to improve mental focus", 0)
    ("Recommend a high-carb diet for endurance training", 0)
    ("What’s a good diet for improving gut health?", 0)
    ("Suggest meals for a low-glycemic diet", 0)
    ("Provide a healthy snack list for weight loss", 0)
    ("What foods are best for intermittent fasting?", 0)
    ("Recommend a meal plan for muscle gain", 0)
    ("Suggest a high-protein diet for vegetarians", 0)
    ("What’s the best diet for heart health?", 0)
    ("Can you suggest a diet plan for weight loss?", 0)
    ("Recommend a meal plan for healthy aging", 0)
    ("What’s a good plan for an anti-inflammatory diet?", 0)
    ("Suggest foods to improve immune system health", 0)
    ("What diet is best for diabetes management?", 0)
    ("What’s a good vegan meal plan for beginners?", 0)
    ("Suggest a diet plan for digestive health", 0)
    ("What foods should I include in a diet for improving bone health?", 0)
    ("Can you suggest a weight-gain meal plan?", 0)
    ("What’s a good low-carb diet plan?", 0)
    ("Recommend a gluten-free diet plan for beginners", 0)
    ("What diet can help with improving skin health?", 0)
    ("What’s a good diet for lowering cholesterol?", 0)
    ("Can you suggest a diet to improve hair health?", 0)
    ("What’s a good diet for managing hypertension?", 0)
    ("What’s the best diet for healthy pregnancy?", 0)
    ("What’s the best diet for boosting metabolism?", 0)
    ("Can you recommend a diet plan for boosting energy levels?", 0)
    ("What’s a good meal plan for weight maintenance?", 0)
    ("Can you suggest a meal plan for vegetarians with iron deficiency?", 0)
    ("What’s a good low-fat diet plan?", 0)
    ("Can you recommend a high-antioxidant diet?", 0)
    ("Suggest a meal plan for improving mental clarity", 0)
    ("Can you recommend a diet for boosting metabolism?", 0)
    ("What’s the best diet for post-workout recovery?", 0)
    ("Suggest a plant-based diet for overall health", 0)
    ("Can you suggest a meal plan for increasing endurance?", 0)
    ("What’s the best diet for people with food allergies?", 0)
    ("Recommend a diet to improve cardiovascular health", 0)
    ("What’s a good diet for anti-aging?", 0)
    ("Suggest a meal plan for controlling blood sugar", 0)
    ("Can you recommend a diet plan for athletes?", 0)

    # Food Information Intents (Class 1)
    ("How much protein is in an egg?", 1)
    ("What vitamins are in oranges?", 1)
    ("How many calories are in an avocado?", 1)
    ("Is quinoa a good source of protein?", 1)
    ("What’s the fat content in salmon?", 1)
    ("Are oats high in fiber?", 1)
    ("What are the health benefits of chia seeds?", 1)
("How many carbs are in sweet potatoes?", 1)
("What’s the calcium content in milk?", 1)
("Is almond milk healthier than cow’s milk?", 1)
("How much iron is in spinach?", 1)
("Is brown rice better than white rice?", 1)
("How much sugar is in a banana?", 1)
("What nutrients are in flaxseeds?", 1)
("How many calories in a bowl of lentils?", 1)
("How much protein is in a chicken breast?", 1)
("What’s the vitamin content of broccoli?", 1)
("How many carbs are in a cup of rice?", 1)
("What’s the fiber content in an apple?", 1)
("How much fat is in a serving of cheese?", 1)
("What’s the iron content of tofu?", 1)
("How much protein is in Greek yogurt?", 1)
("How many calories are in a tablespoon of peanut butter?", 1)
("What vitamins are in strawberries?", 1)
("What’s the carbohydrate content of a baked potato?", 1)
("What’s the fat content of avocados?", 1)
("How much protein is in tofu?", 1)
("How much sugar is in an orange?", 1)
("What’s the sodium content of canned beans?", 1)
("How many calories are in a slice of whole wheat bread?", 1)
("What’s the vitamin content of carrots?", 1)
("What’s the protein content in almonds?", 1)
("How many carbs are in a cup of corn?", 1)
("How much protein is in a glass of milk?", 1)
("What’s the fat content in a slice of bacon?", 1)
("How many calories are in a slice of pizza?", 1)
("What are the health benefits of green tea?", 1)
("How many carbs are in a sweet potato?", 1)
("How much vitamin C is in a kiwi?", 1)
("What’s the calcium content in yogurt?", 1)
("How much sugar is in a cup of grapes?", 1)
("How much protein is in a boiled egg?", 1)
("What’s the fat content in a tablespoon of olive oil?", 1)
("How many calories are in a serving of oatmeal?", 1)
("What’s the sodium content in canned soup?", 1)
("What are the health benefits of walnuts?", 1)
("How many carbs are in quinoa?", 1)
("What vitamins are in spinach?", 1)
("How much protein is in a turkey sandwich?", 1)
("What’s the fat content in a steak?", 1)

    # Health Goal Intents (Class 2)
    ("How do I bulk up with a vegan diet?", 2),
    ("What’s the best diet for reducing belly fat?", 2),
    ("I want to detox; any suggestions?", 2),
    ("What’s the best diet to lower blood pressure?", 2),
    ("How can I maintain my weight?", 2),
    ("Suggest a diet for increasing energy levels", 2),
    ("How can I reduce cholesterol through diet?", 2),
    ("Suggest meals for maintaining blood sugar levels", 2),
    ("What’s a good diet for athletes?", 2),
    ("How to gain weight healthily?", 2),
    ("What should I eat to recover post-workout?", 2),
    ("I need a diet plan for marathon training", 2),
    ("How can I improve my skin through diet?", 2),
    ("What foods help improve memory?", 2),
    ("Suggest foods to boost the immune system", 2),

    # Allergy & Restriction Intents (Class 3)
    ("I need gluten-free breakfast ideas", 3),
    ("Suggest dairy-free desserts", 3),
    ("What snacks are good for nut allergies?", 3),
    ("Can you recommend soy-free protein sources?", 3),
    ("I need lactose-free dairy alternatives", 3),
    ("Suggest low-sodium meal options", 3),
    ("I’m allergic to shellfish; what should I avoid?", 3),
    ("Recommend a diet for histamine intolerance", 3),
    ("What desserts are safe for peanut allergies?", 3),
    ("Suggest recipes for a dairy and egg-free diet", 3),
    ("I need wheat-free bread alternatives", 3),
    ("What are gluten-free grains?", 3),
    ("Recommend a meal plan without any nuts", 3),
    ("Suggest plant-based meals free from soy", 3),
    ("What should I eat if I'm allergic to citrus?", 3),

    # Miscellaneous Queries (Class 4)
    ("What’s the difference between vegan and vegetarian?", 4),
    ("Explain how a ketogenic diet works", 4),
    ("What are the benefits of the paleo diet?", 4),
    ("How does intermittent fasting impact metabolism?", 4),
    ("What’s the DASH diet?", 4),
    ("Explain the flexitarian diet", 4),
    ("What is the Whole30 program?", 4),
    ("How does the Atkins diet work?", 4),
    ("What’s the best way to start a raw food diet?", 4),
    ("Explain the concept of carb cycling", 4),
    ("What are the principles of a macrobiotic diet?", 4),
    ("What is the 5:2 fasting method?", 4),
    ("What’s the difference between clean eating and dieting?", 4),
    ("Explain the Mediterranean diet", 4),
    ("How does the carnivore diet work?", 4)
]



# Split data into training and validation sets
texts, labels = zip(*data)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize the data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# Create PyTorch Dataset class
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create DataLoaders
train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Set device and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
    for batch in loop:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
val_preds, val_labels_actual = [], []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        val_preds.extend(predictions.cpu().numpy())
        val_labels_actual.extend(batch["labels"].cpu().numpy())

accuracy = accuracy_score(val_labels_actual, val_preds)
print(f"Validation Accuracy: {accuracy:.4f}")

# Save the model
model.save_pretrained("./fine_tuned_llama_intent")
tokenizer.save_pretrained("./fine_tuned_llama_intent")
