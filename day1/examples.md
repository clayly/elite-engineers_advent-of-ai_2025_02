# Structured Output Examples

This document demonstrates how to use the new structured output functionality added to the Day 1 chat application.

## Installation

Make sure you have the updated dependencies:

```bash
# Make sure you have pydantic installed
pip install pydantic>=2.0.0
```

## Usage

1. **Start the chat application:**
   ```bash
   python main.py
   ```

2. **Enable structured output mode:**
   ```
   /structured
   ```

3. **Try these example prompts:**

### Basic Response Example
```
What is the capital of France?
```

Expected output:
```json
{
  "answer": "The capital of France is Paris.",
  "followup_question": "Would you like to know more about Paris's history or landmarks?",
  "confidence": 0.95
}
```

### Creative Response Example
```
Tell me a joke about programming
```

Expected output:
```json
{
  "content": "Why do programmers prefer dark mode? Because light attracts bugs!",
  "genre": "joke",
  "mood": "humorous",
  "rating": 7.0
}
```

### Structured Answer Example
```
Explain the benefits of exercise
```

Expected output:
```json
{
  "summary": "Exercise provides numerous physical and mental health benefits that improve overall quality of life.",
  "details": [
    "Improves cardiovascular health and reduces risk of heart disease",
    "Helps maintain healthy weight and metabolism",
    "Strengthens muscles and bones",
    "Reduces stress and improves mental health",
    "Enhances sleep quality and energy levels"
  ],
  "confidence": 0.9
}
```

### Data Extraction Example
```
Extract key information from this text: John Smith, age 35, works as a software engineer at TechCorp. He lives in New York and has 10 years of experience.
```

Expected output:
```json
{
  "extracted_data": {
    "name": "John Smith",
    "age": 35,
    "profession": "software engineer",
    "company": "TechCorp",
    "location": "New York",
    "experience_years": 10
  },
  "extraction_notes": "Successfully extracted personal and professional information from the provided text.",
  "confidence": 0.85
}
```

## Features

- **Automatic Schema Detection**: The system automatically chooses the most appropriate schema based on your input
- **Rich JSON Display**: Structured responses are displayed with syntax highlighting
- **Confidence Scores**: Many responses include confidence ratings
- **Follow-up Questions**: Basic responses suggest relevant follow-up questions
- **Error Handling**: Errors are also displayed in structured format
- **Markdown Parsing**: Automatically extracts JSON from markdown code blocks if needed
- **Robust Fallbacks**: Multiple parsing strategies ensure reliable structured output

## Toggle Modes

- `/structured` - Enable structured output mode
- `/structured` again - Disable and return to normal mode
- `/help` - See all available commands
- `/clear` - Clear conversation history
- `/exit` - Exit the application