# Installation Guide

Follow these steps to set up the MRM_OS environment locally on Windows.

## Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Extremely fast Python package installer and resolver)

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MRM_OS
   ```

2. **Create a virtual environment and install dependencies**
   Using `uv` for fast installation:
   ```powershell
   uv venv
   # Activate the virtual environment
   .venv\Scripts\activate
   
   # Install required packages
   uv pip install -r requirements.txt
   uv pip install python-dotenv
   ```

3. **Configure Environment Variables**
   The application uses Grok (xAI) for generating interpretative narratives.
   Create a `.env` file in the root directory of the project:
   ```powershell
   New-Item .env -ItemType File
   ```
   Open the `.env` file and add your Grok API key:
   ```env
   GROK_API_KEY="your_api_key_here"
   ```

4. **Run the Application**
   Start the backend server using Uvicorn:
   ```powershell
   uv run uvicorn mrm_os.app:app --reload --port 8081
   ```

5. **Access the Application**
   Open your browser and navigate to:
   [http://localhost:8081](http://localhost:8081)
