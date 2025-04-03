# AI Task Allocation Agent

## Overview

This project implements an AI-powered Task Allocation Agent designed to intelligently assign tasks to individuals within a team or organization. It considers factors like user skills, proficiency, availability, current workload, preferences, and historical performance to find the optimal match for each task, aiming to improve efficiency, balance workload, and increase user satisfaction. The system is built using the LangChain framework and utilizes Large Language Models (LLMs) for complex reasoning and decision-making.

## Solution Approach

The core logic of the AI Task Allocation Agent revolves around intelligently matching tasks to the most suitable users. Here's a breakdown of the approach:

1.  **Data Input & Representation:**
    *   **Tasks:** Defined with titles, descriptions, required skills (and minimum proficiency levels), priority, deadlines, and estimated duration.
    *   **Users:** Profiles include their names, unique IDs, skill sets with proficiency levels (e.g., Python: 5/5), availability schedules (working hours, busy times, OOO), preferences (e.g., preferred task types), and historical performance metrics (e.g., completion rate).

2.  **Task Analysis (Automation):** When tasks are created (or input), an automated analysis step (using an LLM) can extract required skills, estimate complexity and duration, suggest categories, and identify keywords if not fully provided.

3.  **Availability Management:** The system maintains user availability using specific time slots and recurring rules (e.g., Mon-Fri 9-5). It checks if a user has sufficient *free* time within the task's deadline window to accommodate the task's estimated duration.

4.  **Core Matching Logic (AI Agent):**
    *   When a task needs allocation, the system identifies *eligible* users based on fundamental criteria:
        *   Do they possess the *required skills* at the *minimum proficiency*?
        *   Are they *available* for the required duration before the deadline?
        *   Is their *current workload* below a reasonable threshold?
    *   The eligible users and the task details are passed to the AI Agent (built with LangChain and an LLM like GPT-4o-mini).
    *   The AI agent evaluates the eligible candidates holistically, considering:
        *   **Skill Fit:** How well do the user's skills match the task requirements, including nuanced understanding beyond simple keyword matching?
        *   **Proficiency Level:** Does the user significantly exceed or just meet the requirements?
        *   **Availability:** Prioritizes users with clearer availability windows.
        *   **Workload Balancing:** Tends to favor users with lower workloads but doesn't strictly exclude moderately busy users if they are a superior skill fit.
        *   **(Optional) Personalization:** Incorporates learned preferences and past performance data as tie-breakers or weighting factors.

5.  **Allocation Output:** The AI agent outputs its best match (user ID), a confidence score (0.0-1.0) indicating its certainty, and a detailed reasoning explaining *why* that user was chosen based on the evaluation criteria. It may also suggest alternative suitable users.

6.  **Personalization & Learning:**
    *   The system records allocation outcomes (completion status, quality, timeliness, user satisfaction/feedback).
    *   This historical data is used to learn user preferences (e.g., which types of tasks or skills they perform well on or enjoy) and update performance metrics.
    *   Learned preferences and performance can subtly influence future allocation decisions, making the system adapt over time.

7.  **User Interface & Interaction:** A Streamlit web interface allows users to:
    *   Manage users and tasks.
    *   Manage availability schedules.
    *   Trigger the AI allocation process for pending tasks.
    *   View the AI's suggestions, reasons, and confidence scores.
    *   Manually override or approve allocations.
    *   Visualize task statuses on a board (Kanban-style).
    *   Interact with a chat interface powered by the AI agent for queries and actions (e.g., "Add task...", "Show pending tasks", "Assign task X to user Y").

## Implementation Details

*   **Core AI Framework:** LangChain (`langchain-core`, `langchain-openai`)
*   **Language Model:** OpenAI GPT-4o-mini (or other compatible models)
*   **User Interface:** Streamlit
*   **Data Handling:** Pandas, NumPy
*   **Date/Time:** Python `datetime`, `pytz`, `python-dateutil`
*   **Machine Learning (Personalization):** Scikit-learn (`sklearn`) for preference modeling (e.g., RandomForestRegressor)
*   **Data Structures/Validation:** Pydantic
*   **Visualization:** Plotly, Matplotlib, Seaborn (for potential future analytics/progress tracking)
*   **Configuration:** `python-dotenv` for managing API keys.
*   **API Calls:** `requests` (used internally by LangChain and potentially for future integrations)

## Execution Steps

1.  **Prerequisites:**
    *   Python 3.10 or higher installed.
    *   `pip` (Python package installer).
    *   Git (for cloning the repository).

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
    *(Replace `<repository_url>` and `<repository_directory>` with the actual URL and folder name)*

3.  **Set up Virtual Environment (Recommended):**
    *   **Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment Variables:**
    *   Create a file named `.env` in the root project directory.
    *   Add your OpenAI API key to the `.env` file:
        ```
        OPENAI_API_KEY='your_openai_api_key_here'
        ```
    *   *(Optional: Add other keys if using specific integrations like Slack/Discord webhooks, calendar APIs, etc.)*

6.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
    *(Ensure `app.py` is the correct entry point file for the Streamlit application)*

7.  **Access the UI:** Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Dependencies

*   **Software:** Python (3.10+)
*   **Python Libraries:** See `requirements.txt`. Key libraries include:
    *   `streamlit`
    *   `langchain-core`, `langchain-openai`
    *   `openai`
    *   `pandas`, `numpy`
    *   `pytz`, `python-dateutil`
    *   `scikit-learn`
    *   `plotly`
    *   `pydantic`
    *   `python-dotenv`
    *   `requests`
    *   `matplotlib`, `seaborn`
*   **APIs:**
    *   OpenAI API Key (Required for the core AI functionality).

## Expected Output

*   **Web Application:** A running Streamlit application accessible via a web browser.
*   **User Interface:**
    *   Sections for managing Users, Tasks, and Availability.
    *   An AI Allocation section to trigger task matching.
    *   A manual allocation section.
    *   An interactive Allocation Board (Kanban-style view) to track task statuses.
    *   A chat interface to interact with the AI agent using natural language commands.
    *   (Potentially) Sections for Smart Availability, Communication, Learning insights, and Progress Tracking.
*   **Functionality:**
    *   Ability to add, view, and manage user profiles and task details.
    *   Ability to define and visualize user availability.
    *   When triggering AI allocation: The system suggests the best user for a task, providing a confidence score and reasoning. Alternative suggestions might be provided.
    *   Users can manually assign tasks.
    *   The allocation board visually updates as tasks move through statuses (Pending, Assigned, In Progress, Completed, Cancelled).
    *   The chat agent responds to queries and executes commands related to tasks and users based on its defined tools.
*   **Outcome:** Tasks are matched to users based on the described logic, aiming for efficient and appropriate assignments. The system should provide transparency into *why* a specific match was made. Data related to allocations might be saved (e.g., `allocation_history.json`, `availability_data.json` as seen in the code).
