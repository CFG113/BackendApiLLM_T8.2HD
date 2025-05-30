import os
from flask import Flask, request, jsonify
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
# http://localhost:5050/getMeals?dish=green%20curry
# Load the model and tokenizer
# MODEL = "meta-llama/Llama-3.2-1B"
MODEL = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

# Determine device (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the appropriate device
print(f"Using device: {device}")

def fetchMealsFromLlama(dish_name):
    print(f"Generating meal suggestions for dish: {dish_name} using {MODEL}")
    prompt = (
        f"Suggest 5 healthy meals similar to the dish '{dish_name}'.\n"
        f"Format your response exactly as follows:\n"
        f"**MEAL 1:** [meal name]\n"
        f"**MEAL 2:** [meal name]\n"
        f"**MEAL 3:** [meal name]\n"
        f"**MEAL 4:** [meal name]\n"
        f"**MEAL 5:** [meal name]\n\n"
        f"Ensure text is properly formatted. It needs to start with '**MEAL 1:**' and include exactly 5 meals in this structure. "
        f"Do not include any commentary, duplicate meals, or additional formatting. "
        f"Follow this pattern for all meal names. "
        f"Here is the target dish:\n{dish_name}"
    )
    try:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move to the same device as model

        # Generate text
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Maximum number of new tokens to generate
            temperature=0.7,  # Control randomness
            top_p=0.9,  # Nucleus sampling
            do_sample=True,  # Enable sampling for diversity
            pad_token_id=tokenizer.eos_token_id  # Handle padding (optional, avoids warning)
        )

        # Decode the generated tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract from "**MEAL 1:**" onwards
        meal_start = generated_text.find("**MEAL 1:**")
        if meal_start == -1:
            raise Exception("Failed to generate a properly formatted meal list")
        meal_text = generated_text[meal_start:]
        return meal_text
    except Exception as e:
        raise Exception(f"Failed to generate Meals with model: {str(e)}")

def fetchSuggestedMeals():
    print(f"Generating healthy meal suggestions using {MODEL}")
    prompt = (
        f"Suggest 5 healthy meals that include one protein-rich ingredient, are delicious, and easy to prepare.'.\n"
        f"Format your response exactly as follows:\n"
        f"**MEAL 1:** [meal name]\n"
        f"**MEAL 2:** [meal name]\n"
        f"**MEAL 3:** [meal name]\n"
        f"**MEAL 4:** [meal name]\n"
        f"**MEAL 5:** [meal name]\n\n"
        f"Ensure text is properly formatted. It needs to start with '**MEAL 1:**' and include exactly 5 meals in this structure. "
        f"Do not include any commentary, duplicate meals, or additional formatting. "
        f"Follow this pattern for all meal names. "
        f"Here is the target dish:\n"
    )
    try:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move to the same device as model

        # Generate text
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Maximum number of new tokens to generate
            temperature=0.7,  # Control randomness
            top_p=0.9,  # Nucleus sampling
            do_sample=True,  # Enable sampling for diversity
            pad_token_id=tokenizer.eos_token_id  # Handle padding (optional, avoids warning)
        )

        # Decode the generated tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract from "**MEAL 1:**" onwards
        meal_start = generated_text.find("**MEAL 1:**")
        if meal_start == -1:
            raise Exception("Failed to generate a properly formatted meal list")
        meal_text = generated_text[meal_start:]
        return meal_text
    except Exception as e:
        raise Exception(f"Failed to generate Meals with model: {str(e)}")

def process_meals(meal_text):
    meals = []
    pattern = re.compile(r'\*\*MEAL \d+:\*\* (.+?)(?=\n|$)', re.DOTALL)
    matches = pattern.findall(meal_text)

    for match in matches:
        meal = match.strip()
        if "[meal name]" in meal.lower():
            continue
        meals.append(meal)
        if len(meals) == 5:
            break

    return meals


@app.route('/getMeals', methods=['GET'])
def get_meals():
    print("Request received")
    dish_name = request.args.get('dish')
    if not dish_name:
        return jsonify({'error': 'Missing dish parameter'}), 400
    try:
        meals = fetchMealsFromLlama(dish_name)
        print(meals)
        processed_meals = process_meals(meals)
        if not processed_meals:
            return jsonify({'error': 'Failed to parse meal data', 'raw_response': meals}), 500
        return jsonify({'meals': processed_meals}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/getSuggestMeals', methods=['GET'])
def get_suggested_meals():
    print("Suggested meals request received")
    try:
        meals = fetchSuggestedMeals()
        print(meals)
        processed_meals = process_meals(meals)
        if not processed_meals:
            return jsonify({'error': 'Failed to parse meal data', 'raw_response': meals}), 500
        return jsonify({'meals': processed_meals}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/test', methods=['GET'])
def run_test():
    return jsonify({'Meal': "test"}), 200

if __name__ == '__main__':
    port_num = 5050
    print(f"App running on port {port_num}")
    app.run(port=port_num, host="0.0.0.0")
