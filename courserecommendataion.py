import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import uvicorn
import json
# Load environment variables from .env file
load_dotenv()

# Initialize Google Gemini Pro
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_KEY"))

# Create the model with the configuration
generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# FastAPI app initialization
app = FastAPI()

# Pydantic model for request payload
class CourseRecommendationRequest(BaseModel):
    student_level: str
    course: str
@app.post("/course-recommendation")
async def recommend_course(request: CourseRecommendationRequest):
    # Extract student level and course from the request
    student_level = request.student_level
    course = request.course

    # Construct the prompt based on student level and course
    prompt = f"""
    You are an intelligent assistant specializing in educational course recommendations. 
    Based on the student's level and the specified course, recommend 4 appropriate courses with the following details: 
    Subject, Number of Units, Focus Area, and Difficulty Level. Respond in JSON format.

    Input:
    1. Student Level: {student_level}
    2. Course: {course}

    Output:
        {{
            "subject": "Python",
            "units": 3,
            "focus_area": "Python Basics",
            "difficulty": "Beginner"
        }},
        {{
            "subject": "Data Structures",
            "units": 3,
            "focus_area": "Arrays and Linked Lists",
            "difficulty": "Intermediate"
        }},
        {{
            "subject": "Algorithms",
            "units": 3,
            "focus_area": "Sorting and Searching",
            "difficulty": "Intermediate"
        }},
        {{
            "subject": "Advanced Python",
            "units": 3,
            "focus_area": "Python for Data Science",
            "difficulty": "Advanced"
        }}
    """

    try:
        # Generate the response from the model
        gemini_response = model.generate_content(prompt)

        # Clean and parse the response
        response_text = gemini_response.text.strip()

        # Remove any extraneous markers
        if response_text.startswith("```json"):
            response_text = response_text[7:]  # Remove the opening ```json
        if response_text.endswith("```"):
            response_text = response_text[:-3]  # Remove the closing ```
        if response_text.startswith("'''"):
            response_text = response_text[3:]  # Remove the opening '''
        if response_text.endswith("'''"):
            response_text = response_text[:-3]  # Remove the closing '''
        
        # Wrap the text in square brackets if not already valid JSON
        response_text = f"[{response_text.strip().strip('[]')}]"

        # Parse the response into JSON
        recommendations = json.loads(response_text)
        return {"recommendations": recommendations}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse recommendations: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)