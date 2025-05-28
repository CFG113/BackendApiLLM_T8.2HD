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

def fetchQuizFromLlama(student_topic):
    print(f"Generating quiz for topic: {student_topic} using {MODEL}")
    prompt = (
        f"Generate a quiz with 3 questions to test students on the provided topic. "
        f"For each question, generate 4 options where only one of the options is correct. "
        f"Format your response as follows:\n"
        f"**QUESTION 1:** [Your question here]?\n"
        f"**OPTION A:** [First option]\n"
        f"**OPTION B:** [Second option]\n"
        f"**OPTION C:** [Third option]\n"
        f"**OPTION D:** [Fourth option]\n"
        f"**ANS:** [Correct answer letter]\n\n"
        f"**QUESTION 2:** [Your question here]?\n"
        f"**OPTION A:** [First option]\n"
        f"**OPTION B:** [Second option]\n"
        f"**OPTION C:** [Third option]\n"
        f"**OPTION D:** [Fourth option]\n"
        f"**ANS:** [Correct answer letter]\n\n"
        f"**QUESTION 3:** [Your question here]?\n"
        f"**OPTION A:** [First option]\n"
        f"**OPTION B:** [Second option]\n"
        f"**OPTION C:** [Third option]\n"
        f"**OPTION D:** [Fourth option]\n"
        f"**ANS:** [Correct answer letter]\n\n"
        f"Ensure text is properly formatted. It needs to start with a question, then the options, and finally the correct answer. "
        f"Follow this pattern for all questions. "
        f"Here is the student topic:\n{student_topic}"
    )

    try:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move to the same device as model

        # Generate text
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,  # Maximum number of new tokens to generate
            temperature=0.7,     # Control randomness
            top_p=0.9,           # Nucleus sampling
            do_sample=True,      # Enable sampling for diversity
            pad_token_id=tokenizer.eos_token_id  # Handle padding (optional, avoids warning)
        )

        # Decode the generated tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the quiz part from the full output
        quiz_start = generated_text.find("**QUESTION 1:**")
        if quiz_start == -1:
            raise Exception("Failed to generate a properly formatted quiz")
        quiz_text = generated_text[quiz_start:]
        return quiz_text
    except Exception as e:
        raise Exception(f"Failed to generate quiz with model: {str(e)}")

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
            max_new_tokens=125,  # Maximum number of new tokens to generate
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
        if len(meals) == 5:  # ✅ Limit to 5 meals only
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

def process_quiz(quiz_text):
    questions = []
    pattern = re.compile(
        r'\*\*QUESTION \d+:\*\* (.+?)\n'
        r'\*\*OPTION A:\*\* (.+?)\n'
        r'\*\*OPTION B:\*\* (.+?)\n'
        r'\*\*OPTION C:\*\* (.+?)\n'
        r'\*\*OPTION D:\*\* (.+?)\n'
        r'\*\*ANS:\*\* (.+?)(?=\n|$)',
        re.DOTALL
    )
    matches = pattern.findall(quiz_text)

    for match in matches:
        question = match[0].strip()
        options = [match[1].strip(), match[2].strip(), match[3].strip(), match[4].strip()]
        correct_ans = match[5].strip()

        # Skip questions with placeholder text
        if "[Your question here]" in question or any("[First option]" in opt for opt in options):
            continue

        question_data = {
            "question": question,
            "options": options,
            "correct_answer": correct_ans
        }
        questions.append(question_data)

    return questions

@app.route('/getQuiz', methods=['GET'])
def get_quiz():
    print("Request received")
    student_topic = request.args.get('topic')
    if not student_topic:
        return jsonify({'error': 'Missing topic parameter'}), 400
    try:
        quiz = fetchQuizFromLlama(student_topic)
        print(quiz)
        processed_quiz = process_quiz(quiz)
        if not processed_quiz:
            return jsonify({'error': 'Failed to parse quiz data', 'raw_response': quiz}), 500
        return jsonify({'quiz': processed_quiz}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def run_test():
    return jsonify({'quiz': "test"}), 200

if __name__ == '__main__':
    port_num = 5050
    print(f"App running on port {port_num}")
    app.run(port=port_num, host="0.0.0.0")
