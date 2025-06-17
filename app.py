from flask import Flask, render_template, request, send_from_directory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Initialize environment
load_dotenv()

# Create Flask app with explicit configuration
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')

# Debugging setup
DEBUG = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Verify critical paths
TEMPLATE_DIR = Path(__file__).parent / 'templates'
print(f"\n=== DEBUG INFO ===")
print(f"Template directory: {TEMPLATE_DIR}")
print(f"Templates found: {list(TEMPLATE_DIR.glob('*.html'))}")
print(f"GROQ_API_KEY present: {bool(os.environ.get('GROQ_API_KEY'))}\n")

# Initialize LLM with error handling
try:
    llm_resto = ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # Using confirmed working model
        temperature=0.0
    )
    print("LLM initialized successfully")
except Exception as e:
    print(f"LLM initialization failed: {str(e)}")
    llm_resto = None

# Define prompt template
prompt_template_resto = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 
                    'disease', 'region', 'allergics', 'foodtype'],
    template="""Diet Recommendation System:
I want you to provide output in the following format using the input criteria:

Restaurants:
- name1
- name2
- name3
- name4
- name5
- name6

Breakfast:
- item1
- item2
- item3
- item4
- item5
- item6

Dinner:
- item1
- item2
- item3
- item4
- item5

Workouts:
- workout1
- workout2
- workout3
- workout4
- workout5
- workout6

Criteria:
Age: {age}, Gender: {gender}, Weight: {weight} kg, Height: {height} ft,
Vegetarian: {veg_or_nonveg}, Disease: {disease}, Region: {region},
Allergics: {allergics}, Food Preference: {foodtype}.
"""
)

@app.route('/')
def index():
    """Render main form page with enhanced debugging"""
    try:
        print("\n=== INDEX ROUTE ===")
        template_path = TEMPLATE_DIR / 'index.html'
        print(f"Attempting to render: {template_path}")
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found at {template_path}")
        
        return render_template("index.html")
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        return f"Error loading page: {str(e)}", 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handle form submission with robust error handling"""
    print("\n=== RECOMMEND ROUTE ===")
    
    if not llm_resto:
        return render_template("error.html", 
                            error="AI service unavailable"), 503

    try:
        # Collect and validate form data
        form_data = {
            'age': request.form.get('age', ''),
            'gender': request.form.get('gender', ''),
            'weight': request.form.get('weight', ''),
            'height': request.form.get('height', ''),
            'veg_or_nonveg': request.form.get('veg_or_nonveg', ''),
            'disease': request.form.get('disease', ''),
            'region': request.form.get('region', ''),
            'allergics': request.form.get('allergics', ''),
            'foodtype': request.form.get('foodtype', '')
        }

        print("Form data received:", form_data)

        # Convert height from meters to feet
        try:
            height_m = float(form_data['height'])
            height_ft = height_m * 3.28084
            form_data['height'] = f"{height_ft:.1f}"
        except (ValueError, KeyError) as e:
            raise ValueError("Invalid height value") from e

        # Generate recommendations
        chain = LLMChain(llm=llm_resto, prompt=prompt_template_resto)
        results = chain.run(form_data)
        print("LLM response received (first 200 chars):", results[:200])

        # Parse results
        def extract_section(pattern):
            match = re.search(pattern, results, re.DOTALL)
            return [line.strip("- ") for line in match.group(1).strip().split("\n") if line.strip()] if match else []

        recommendations = {
            'restaurant_names': extract_section(r'Restaurants:\s*(.*?)\n\n'),
            'breakfast_names': extract_section(r'Breakfast:\s*(.*?)\n\n'),
            'dinner_names': extract_section(r'Dinner:\s*(.*?)\n\n'),
            'workout_names': extract_section(r'Workouts:\s*(.*?)\n\n')
        }

        print("Recommendations prepared:", {k: len(v) for k, v in recommendations.items()})

        return render_template('result.html', **recommendations)

    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return render_template("error.html", error=str(e)), 400

@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return "OK", 200

if __name__ == "__main__":
    # Run with enhanced debugging
    app.run()
# , debug=DEBUG, use_reloader=DEBUG, port=5000)