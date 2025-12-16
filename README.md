# StressPredict - Real-Time Stress Monitoring with AI

Hey! This is StressPredict - a real-time stress detection system that monitors physiological signals and gives you AI-powered wellness coaching when stress gets high.

Built with Python and Streamlit so you can run it anywhere, no backend needed.

## What This Does

Stress is invisible until it's too late. This app watches your body's signals in real-time and alerts you the moment stress levels spike - then tells you exactly what to do about it.

It's like having a wellness coach that analyzes your vitals and jumps in when you need help.

## The Tech Stack

- **Streamlit** - Web UI framework
- **Scikit-learn** - Machine learning (Random Forest classifier, 87% accuracy)
- **OpenAI/Anthropic API** - LLM for personalized wellness coaching
- **Plotly** - Interactive signal charts
- **NumPy/Pandas** - Signal processing and feature extraction
- **SQLite** - Local database for session history

Everything runs locally on your machine.

## Setup Instructions

### What You Need
- Python 3.9 or higher
- OpenAI API key OR Anthropic API key (for AI coaching)

### Installation

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/stresspredict.git
cd stresspredict
Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Set up your API key
Create a .env file in the project root:

OPENAI_API_KEY=your_openai_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_key_here
Get your API key from:

OpenAI: https://platform.openai.com/api-keys
Anthropic: https://console.anthropic.com/
Run the app
streamlit run app.py
The app opens at localhost:8501 automatically.
