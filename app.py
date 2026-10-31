from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import chromadb
import requests

# ---------------------------------------------------------
# Application Configuration & Metadata
# ---------------------------------------------------------
app = FastAPI(
    title="CampusBuddy - Offline Institutional FAQ Assistant",
    description="100% Offline RAG System using FastAPI, ChromaDB, and Ollama (gemma3:1b)",
    version="1.0.0"
)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"
CHROMA_COLLECTION_NAME = "campus_policies_demo"

# ---------------------------------------------------------
# 1. Initialize Local ChromaDB with Comprehensive Seed Data
# ---------------------------------------------------------
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"description": "Verified Official Campus Policies (Read-Only Demo)"}
)

# Comprehensive set of official institutional policies for public demo
CAMPUS_POLICIES = [
    {
        "id": "policy_attendance",
        "text": "Attendance Policy: Students must maintain a minimum of 75% attendance in every registered course to be eligible to sit for the end-semester examinations. Medical leave may grant a relaxation of up to 5% upon submission of a valid medical practitioner certificate within 3 days of resuming classes."
    },
    {
        "id": "policy_laptop_it",
        "text": "Laptop & IT Policy: Students must bring a fully charged laptop (with at least 4 hours of battery backup) to practical lab sessions. Campus Wi-Fi MAC address registration is compulsory and managed by the IT Helpdesk located in Academic Block B, Room 204."
    },
    {
        "id": "policy_library",
        "text": "Library Rules: Undergraduate students can borrow up to 4 books simultaneously for a duration of 14 calendar days. Overdue returns are charged a fine of Rs. 10 per day per book. Reference books, journals, and thesis copies cannot be removed from the library reading hall."
    },
    {
        "id": "policy_hostel",
        "text": "Hostel Curfew & Entry: Campus hostel gates close strictly at 9:30 PM on weekdays (Monday to Friday) and 10:30 PM on weekends (Saturday and Sunday). Late entry or overnight leave requires prior digital approval from the hostel warden submitted at least 6 hours in advance."
    },
    {
        "id": "policy_grading",
        "text": "Examination & Grading: The grading follows a 10-point CGPA system. Continuous Internal Assessment (CIA) accounts for 40% of the total course marks, and the End-Semester Examination accounts for 60%. A minimum overall aggregate of 40% is required to pass each subject."
    },
    {
        "id": "policy_medical",
        "text": "Campus Medical Centre: The campus dispensary is situated on the Ground Floor of Administrative Block A and operates 24/7 for emergencies. Routine medical consultations are available from 8:00 AM to 8:00 PM. Emergency ambulance dispatch can be requested via internal extension 108."
    },
    {
        "id": "policy_cafeteria",
        "text": "Cafeteria & Dining: The central food court and campus dining halls operate from 7:30 AM to 9:30 PM daily. Nutritious vegetarian and non-vegetarian meal plans are provided. External delivery agents are only permitted up to the main security checkpoint."
    },
    {
        "id": "policy_discipline",
        "text": "Anti-Ragging & Code of Conduct: The institution maintains a strict zero-tolerance policy towards ragging, harassment, or bullying of any form. Violations result in immediate suspension, institutional inquiry, and reporting to police authorities. The 24/7 Anti-Ragging helpline is toll-free 1800-180-5522."
    },
    {
        "id": "policy_placement",
        "text": "Career & Placement Policy: Students with a cumulative CGPA of 7.0 or higher and zero standing backlogs are eligible for Tier-1 campus placement drives. Placement registrations are managed exclusively through the Career Development Cell portal in the 7th semester."
    }
]

# Securely upsert policies into ChromaDB on startup
collection.upsert(
    documents=[p["text"] for p in CAMPUS_POLICIES],
    ids=[p["id"] for p in CAMPUS_POLICIES]
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def query_ollama(prompt: str) -> dict:
    """Sends prompt to local Ollama instance with comprehensive error handling."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return {"success": True, "answer": result.get("response", "").strip()}
        elif response.status_code == 404:
            return {
                "success": False,
                "answer": f"Ollama model '{OLLAMA_MODEL}' was not found. Please run 'ollama pull {OLLAMA_MODEL}' in your terminal."
            }
        else:
            return {
                "success": False,
                "answer": f"Ollama returned HTTP error code {response.status_code}: {response.text}"
            }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "answer": "Cannot connect to local Ollama. Please make sure Ollama is running (`ollama serve`)."
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "answer": "The local Ollama request timed out (waited 60 seconds). Your system might be under heavy load."
        }
    except Exception as exc:
        return {
            "success": False,
            "answer": f"Unexpected error while querying local Ollama: {str(exc)}"
        }

# ---------------------------------------------------------
# 2. REST API Endpoints (Read-Only for Public Demo)
# ---------------------------------------------------------
@app.get("/ask")
def ask_assistant(question: str = Query(..., description="Student or visitor query")):
    """
    RAG Endpoint:
    1. Embeds and queries ChromaDB for the most relevant campus policy.
    2. Builds a strictly grounded prompt with retrieved context.
    3. Prompts local Ollama (gemma3:1b) and returns answer + source document.
    """
    cleaned_question = question.strip()
    if not cleaned_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Retrieve top matching policy from ChromaDB
    try:
        search_results = collection.query(query_texts=[cleaned_question], n_results=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChromaDB search failed: {str(e)}")

    docs = search_results.get("documents", [[]])
    matched_doc = docs[0][0] if docs and len(docs) > 0 and len(docs[0]) > 0 else "No relevant institutional policy document found."

    # 2. Construct grounded system prompt for local LLM
    strict_prompt = (
        "You are 'CampusBuddy', an official, reliable institutional campus assistant.\n"
        "Answer the user's question accurately, concisely, and politely using ONLY the official policy excerpt below.\n"
        "If the answer cannot be determined strictly from the official policy, respond with: "
        "'I do not have sufficient information in the official campus policies to answer that question.'\n\n"
        f"--- OFFICIAL POLICY EXCERPT ---\n{matched_doc}\n--------------------------------\n\n"
        f"User Question: {cleaned_question}\n\n"
        "Official Answer:"
    )

    # 3. Query local Ollama
    llm_result = query_ollama(strict_prompt)

    return {
        "question": cleaned_question,
        "retrieved_policy": matched_doc,
        "answer": llm_result["answer"],
        "ollama_connected": llm_result["success"]
    }

@app.get("/policies")
def list_policies():
    """Returns the list of verified, preloaded campus policies."""
    return {
        "total_count": len(CAMPUS_POLICIES),
        "status": "Read-Only Demo Mode",
        "policies": [{"id": p["id"], "text": p["text"]} for p in CAMPUS_POLICIES]
    }

# ---------------------------------------------------------
# 3. Single-Page Web UI (Root Endpoint)
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CampusBuddy - 100% Offline RAG Assistant</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --primary-light: #eff6ff;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --accent-green: #10b981;
            --accent-purple: #8b5cf6;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.5;
            padding: 36px 16px;
        }

        .container {
            max-width: 820px;
            margin: 0 auto;
        }

        header {
            margin-bottom: 24px;
            text-align: center;
        }

        .badges-wrapper {
            display: flex;
            gap: 8px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            font-size: 12px;
            font-weight: 600;
            border-radius: 9999px;
            background: #dbeafe;
            color: #1e40af;
        }

        .badge-green {
            background: #dcfce7;
            color: #166534;
        }

        .badge-purple {
            background: #f3e8ff;
            color: #6b21a8;
        }

        .badge-gray {
            background: #f1f5f9;
            color: #475569;
            border: 1px solid #cbd5e1;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
        }

        h1 {
            font-size: 30px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #0f172a;
            margin-bottom: 6px;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 15px;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 26px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
            margin-bottom: 20px;
        }

        .form-group {
            margin-bottom: 16px;
        }

        label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: #334155;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .input-row {
            display: flex;
            gap: 10px;
        }

        input[type="text"] {
            width: 100%;
            padding: 12px 16px;
            font-size: 15px;
            border: 1.5px solid var(--border);
            border-radius: 10px;
            outline: none;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
            font-family: inherit;
            background: #fff;
        }

        input[type="text"]:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
        }

        button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 12px 22px;
            font-size: 15px;
            font-weight: 600;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            transition: all 0.15s ease;
            background: var(--primary);
            color: #fff;
            white-space: nowrap;
        }

        button:hover {
            background: var(--primary-hover);
        }

        button:disabled {
            opacity: 0.65;
            cursor: not-allowed;
        }

        .quick-chips {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 14px;
            align-items: center;
        }

        .quick-chips-title {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 600;
        }

        .chip {
            background: #f1f5f9;
            color: #334155;
            padding: 5px 12px;
            border-radius: 8px;
            font-size: 12.5px;
            cursor: pointer;
            border: 1px solid #e2e8f0;
            transition: all 0.15s;
        }

        .chip:hover {
            background: #e2e8f0;
            border-color: #cbd5e1;
        }

        /* Results Display */
        .result-container {
            display: none;
            margin-top: 24px;
            border-top: 1px solid var(--border);
            padding-top: 20px;
        }

        .answer-box {
            background: #f8fafc;
            border: 1.5px solid #dbeafe;
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 16px;
        }

        .box-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .box-title {
            font-size: 13px;
            font-weight: 700;
            color: #1e3a8a;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .copy-btn {
            background: transparent;
            color: #64748b;
            border: 1px solid #cbd5e1;
            padding: 3px 8px;
            font-size: 11px;
            border-radius: 6px;
            cursor: pointer;
        }

        .copy-btn:hover {
            background: #e2e8f0;
            color: #1e293b;
        }

        .answer-text {
            font-size: 15px;
            color: #1e293b;
            white-space: pre-wrap;
            line-height: 1.6;
        }

        .source-box {
            background: #fffbeb;
            border: 1px solid #fef3c7;
            border-left: 4px solid #f59e0b;
            border-radius: 8px;
            padding: 14px 16px;
            font-size: 13.5px;
            color: #78350f;
        }

        .source-title {
            font-weight: 700;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
            color: #b45309;
        }

        footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 12px;
            margin-top: 28px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badges-wrapper">
                <span class="badge badge-green"><span class="status-dot"></span> 100% Offline Local RAG</span>
                <span class="badge"><span class="status-dot"></span> ChromaDB Vector Store</span>
                <span class="badge badge-purple"><span class="status-dot"></span> Local Ollama: gemma3:1b</span>
                <span class="badge badge-gray">Demo Mode &bull; Read-Only</span>
            </div>
            <h1>CampusBuddy FAQ Assistant</h1>
            <p class="subtitle">Official institutional policy assistant powered strictly by local embeddings & open-weight LLM.</p>
        </header>

        <div class="card">
            <div class="form-group">
                <label for="questionInput">Ask a Question</label>
                <div class="input-row">
                    <input type="text" id="questionInput" placeholder="e.g., What is the minimum attendance required for semester exams?" autofocus />
                    <button id="askBtn" onclick="handleAsk()">Ask Assistant</button>
                </div>
            </div>

            <div class="quick-chips">
                <span class="quick-chips-title">Suggested:</span>
                <div class="chip" onclick="fillQuestion('What is the minimum attendance required for semester exams?')">Attendance Rule</div>
                <div class="chip" onclick="fillQuestion('How many books can I borrow from the library and for how long?')">Library Books & Fines</div>
                <div class="chip" onclick="fillQuestion('What time do the hostel gates close on weekdays and weekends?')">Hostel Curfew</div>
                <div class="chip" onclick="fillQuestion('Where do I register my laptop MAC address for campus Wi-Fi?')">Campus Wi-Fi & IT</div>
                <div class="chip" onclick="fillQuestion('What are the campus medical centre emergency hours and ambulance contact?')">Medical Emergency</div>
                <div class="chip" onclick="fillQuestion('What is the minimum passing grade and CGPA evaluation scheme?')">Grading & Exams</div>
            </div>

            <div class="result-container" id="resultContainer">
                <div class="answer-box">
                    <div class="box-header">
                        <span class="box-title">AI Generated Response (Local gemma3:1b)</span>
                        <button class="copy-btn" onclick="copyAnswer()">Copy</button>
                    </div>
                    <div class="answer-text" id="answerText"></div>
                </div>

                <div class="source-box">
                    <div class="source-title">Verified Official Policy (Retrieved from ChromaDB)</div>
                    <div id="sourceText"></div>
                </div>
            </div>
        </div>

        <footer>
            CampusBuddy Institutional AI &bull; FastAPI &bull; ChromaDB &bull; Ollama gemma3:1b &bull; 100% Private & Offline
        </footer>
    </div>

    <script>
        const questionInput = document.getElementById('questionInput');
        const askBtn = document.getElementById('askBtn');
        const resultContainer = document.getElementById('resultContainer');
        const answerText = document.getElementById('answerText');
        const sourceText = document.getElementById('sourceText');

        questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                handleAsk();
            }
        });

        function fillQuestion(text) {
            questionInput.value = text;
            handleAsk();
        }

        async function handleAsk() {
            const question = questionInput.value.trim();
            if (!question) return;

            askBtn.disabled = true;
            askBtn.innerText = 'Searching & Thinking...';
            resultContainer.style.display = 'none';

            try {
                const response = await fetch(`/ask?question=${encodeURIComponent(question)}`);
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || 'Failed to process request');
                }

                answerText.innerText = data.answer;
                sourceText.innerText = data.retrieved_policy;
                resultContainer.style.display = 'block';
            } catch (err) {
                answerText.innerText = 'Error: ' + err.message;
                sourceText.innerText = 'Could not retrieve source policy due to an error.';
                resultContainer.style.display = 'block';
            } finally {
                askBtn.disabled = false;
                askBtn.innerText = 'Ask Assistant';
            }
        }

        function copyAnswer() {
            const text = answerText.innerText;
            if (navigator.clipboard) {
                navigator.clipboard.writeText(text);
                alert('Answer copied to clipboard!');
            }
        }
    </script>
</body>
</html>
"""

# ---------------------------------------------------------
# Entrypoint for direct Python execution
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print(">> Starting CampusBuddy Demo (100% Offline RAG)")
    print(">> Open Web Interface: http://127.0.0.1:8000")
    print(">> API Documentation: http://127.0.0.1:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
