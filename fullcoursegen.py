import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
import google.generativeai as genai
from dotenv import load_dotenv
import json
import uvicorn
import re
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
import requests
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_KEY"))

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

# Pydantic model for request payload
class SubjectRequest(BaseModel):
    ques: str

class CourseRequest(BaseModel):
    subject: str = Field(..., description="The subject of the course")
    difficulty: DifficultyLevel = Field(..., description="Difficulty level of the course")
    focus_area: str = Field(..., description="Specific area to focus on within the subject")
    units: int = Field(..., ge=1, le=10, description="Number of units desired")

generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

app = FastAPI()

# Add CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change to specific origins as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

YOUTUBE_API_KEY = "AIzaSyAoV0_ILuFFE8WyfxbifJtk2asH2HFj9Do"

async def fetch_youtube_video(query: str) -> str:
    youtube_api_url = (
        f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=1&q={query}&type=video&key={YOUTUBE_API_KEY}"
    )
    try:
        response = requests.get(youtube_api_url, timeout=10)  # Add timeout
        response.raise_for_status()
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            video_id = data["items"][0]["id"]["videoId"]
            return f"https://www.youtube.com/watch?v={video_id}"
        return "No relevant video found."
    except requests.exceptions.Timeout:
        return "YouTube fetch timeout."
    except Exception as e:
        print(f"Error fetching YouTube video: {e}")
        return "Error fetching video."

async def generate_unit_content(unit_data: dict, subject: str, difficulty: str, focus_area: str):
    """Generate detailed content for each topic in a unit"""
    content_prompt = f"""
    Generate detailed educational content for the unit "{unit_data['unitTitle']}" in {subject}.
    Topics to cover: {', '.join(unit_data['topicsCovered'])}
    Learning objectives: {', '.join(unit_data['learningObjectives'])}
    Difficulty level: {difficulty}
    Focus area: {focus_area}

    Return the response in this JSON format:
    {{
        "topicContents": [
            {{
                "topic": "Topic Name",
                "content": "Detailed explanation and educational content",
                "examples": ["example 1", "example 2"],
                "exercises": ["exercise 1", "exercise 2"]
            }}
        ]
    }}

    Ensure content is practical and matches the specified difficulty level.
    Give the content in about minimum 6000 words.
    """
    
    try:
        response = model.generate_content(content_prompt)
        cleaned_json = re.sub(r"^```json|```$", "", response.text, flags=re.MULTILINE).strip()
        print(f"Content generation response for {unit_data['unitTitle']}: {cleaned_json}")
        
        content_data = json.loads(cleaned_json)
        return content_data
    except Exception as e:
        print(f"Error generating content for unit {unit_data['unitTitle']}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate content for unit {unit_data['unitTitle']}: {str(e)}"
        )

async def get_unit_details(unit_title: str, subject: str, difficulty: str, focus_area: str):
    """Generate initial unit structure with objectives and topics"""
    unit_prompt = f"""
    Generate a detailed unit structure for "{unit_title}" in {subject} course.
    Difficulty level: {difficulty}
    Focus area: {focus_area}

    Return the response in this JSON format:
    {{
        "unitTitle": "{unit_title}",
        "learningObjectives": ["detailed objective 1", "detailed objective 2", ...],
        "topicsCovered": ["detailed topic 1", "detailed topic 2", ...],
        "resources": ["resource 1", "resource 2", ...],
        "estimatedDuration": "X weeks"
    }}

    Ensure content matches the difficulty level and focuses on practical applications.
    """
    try:
        response = model.generate_content(unit_prompt)
        raw_response = response.text
        print(f"Raw response for {unit_title}: {raw_response}")

        # Clean and validate JSON
        cleaned_json = re.sub(r"^```json|```$", "", raw_response, flags=re.MULTILINE).strip()
        try:
            unit_data = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing error for {unit_title}: {str(e)}")


        #NEW EDIT
        # Generate detailed content for the unit
        detailed_content = await generate_unit_content(unit_data, subject, difficulty, focus_area)
        # Merge the unit structure with detailed content
        unit_data["detailedContent"] = detailed_content

        # Fetch YouTube video URL
        youtube_query = f"{unit_title} {subject} {focus_area}"
        youtube_video_url = await fetch_youtube_video(youtube_query)
        unit_data["youtube_video_url"] = youtube_video_url

        return unit_data
    except Exception as e:
        print(f"Error in get_unit_details for {unit_title}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate unit details for {unit_title}: {str(e)}"
        )


# Route to generate the course syllabus
@app.post("/doubt-chatbot")
async def generate_syllabus(request: SubjectRequest):
    # Get the subject from the request body
    subject = request.ques
    
    # Construct the prompt with the provided subject
    prompt = f"""
    You are a doubt chatbot for students and you have to resolve students doubts, The question is:{subject}
    """
    
    # Generate the response from the model
    response = model.generate_content(prompt)
    
    # Return the generated syllabus as a JSON response
    return {"answer": response.text}

@app.post("/generate-course")
async def generate_course(request: CourseRequest):
    """Generate a complete course with detailed unit content"""
    try:
        # Generate course structure
        structure_prompt = f"""
        Generate a comprehensive course structure for {request.subject} with exactly {request.units} units.
        Focus area: {request.focus_area}
        Difficulty: {request.difficulty}

        Return ONLY unit titles in this JSON format:
        {{
            "courseTitle": "",
            "difficultyLevel": "",
            "description": "",
            "prerequisites": ["prerequisite 1", "prerequisite 2"],
            "learningOutcomes": ["outcome 1", "outcome 2"],
            "units": [
                {{
                    "unitTitle": "",
                    "unitDescription": ""
                }}
            ],
            "overview": "",
            "assessmentMethods": ["method 1", "method 2"]
        }}
        """
        
        structure_response = model.generate_content(structure_prompt)
        cleaned_json = re.sub(r"^```json|```$", "", structure_response.text, flags=re.MULTILINE).strip()
        print(f"Course structure response: {cleaned_json}")
        
        course_structure = json.loads(cleaned_json)
        
        # Generate detailed content for each unit
        detailed_units = []
        for unit in course_structure["units"]:
            try:
                unit_details = await get_unit_details(
                    unit["unitTitle"],
                    request.subject,
                    request.difficulty,
                    request.focus_area
                )
                detailed_units.append(unit_details)
                print(f"Successfully processed unit: {unit['unitTitle']}")
            except Exception as e:
                print(f"Error processing unit {unit['unitTitle']}: {str(e)}")
                continue
        
        if not detailed_units:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate any unit details"
            )
        
        # Update course structure with detailed units
        course_structure["units"] = detailed_units
        
        return course_structure
        
    except Exception as e:
        print(f"Error in generate_course: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
