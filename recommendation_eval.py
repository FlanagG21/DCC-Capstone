import os
import time
import pickle
import random
import pandas as pd
from google import genai
from tqdm import tqdm
import re

# ── Configuration ────────────────────────────────────────────────────────────────
API_KEY        = "API_KEY"
MODEL_NAME     = "gemini-2.5-flash-preview-04-17"
BATCH_SIZE     = 50
MAX_REQUESTS   = 20
SLEEP_SECONDS  = 3
OUTPUT_INDICES = "chosen_indices_v2.pkl"
OUTPUT_CSV     = "gemini_pairwise_results_v2.csv"

# ── Setup ───────────────────────────────────────────────────────────────────────
client = genai.Client(api_key=API_KEY)
# Load your recommendation DataFrames
normal_rec = pd.read_csv("recommendation_with_scores.csv")
encoded_rec = pd.read_csv("autoencoder_recommendation_with_scores.csv")

# Load existing results if available
existing_results = None
processed_indices = set()
if os.path.exists(OUTPUT_CSV):
    existing_results = pd.read_csv(OUTPUT_CSV)
    processed_indices = set(existing_results['index'].values)
    print(f"Found existing results with {len(processed_indices)} processed songs")

# ── Sampling Indices ─────────────────────────────────────────────────────────────
total_songs = len(normal_rec)
max_songs = min(total_songs, MAX_REQUESTS * BATCH_SIZE)
chosen_idxs = random.sample(range(total_songs), max_songs) 
# Save indices for reproducibility
with open(OUTPUT_INDICES, "wb") as f:
    pickle.dump(chosen_idxs, f)

# Chunk indices into batches of size 10
batches = [
    chosen_idxs[i : i + BATCH_SIZE]
    for i in range(0, len(chosen_idxs), BATCH_SIZE)
]

results = []

# ── Batch Evaluation Loop ───────────────────────────────────────────────────────
try:
    for req_num, batch in enumerate(tqdm(batches, desc="Gemini batches"), 1):
        if req_num > MAX_REQUESTS:
            break  # Enforce max requests
            
        # Skip batches where any index has already been processed
        if any(idx in processed_indices for idx in batch):
            print(f"Skipping batch {req_num} as it contains already processed indices")
            continue

        # Build a combined prompt for 10 songs
        prompt_parts = []
        for idx in batch:
            song = normal_rec.iloc[idx]["song_title"]
            list_a = normal_rec.iloc[idx][[f"best{i}_title" for i in range(1,6)]].tolist()
            list_b = encoded_rec.iloc[idx][[f"best{i}_title" for i in range(1,6)]].tolist()

            prompt_parts.append(
                f"Song #{idx}: \"{song}\"\n"
                f"  A: {', '.join(list_a)}\n"
                f"  B: {', '.join(list_b)}"
            )
        full_prompt = "\n\n".join(prompt_parts)

        # Call Gemini in a single batch for 10 songs 
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=(
        "You are a strict judge evaluating two movie recommendation lists for each song. "
        "Rate List A and List B from 1 to 10 (1 = poor, 10 = excellent). "
        "Skip unknown songs or movies. Be critical. "
        "Format for each Song #:\n"
        "Song id: #<idx>\n"
        "Winner: <A|B|Tie>\n"
        "List_A_Score: <1-10>\n"
        "List_B_Score: <1-10>\n\n"
        "DATA:\n" + full_prompt
            )
        )
        
        # print("Response:", resp)

        # Replace the parsing section with this more robust version:
        text  = resp.candidates[0].content.parts[0].text
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        current = {}
        for line in lines:
            # 1) Start of a new song block
            if line.startswith("Song id:"):
                # save the previous song (if any)
                if current:
                    results.append(current)
                    current = {}
                # extract the index (with or without the ‘#’)
                m = re.match(r"Song id:\s*#?(\d+)", line)
                if m:
                    current["index"] = int(m.group(1))

            # 2) Winner line
            elif line.startswith("Winner:"):
                current["winner"] = line.split(":", 1)[1].strip()

            # 3) List A score
            elif line.startswith("List_A_Score"):
                try:
                    current["list_a_score"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    current["list_a_score"] = None

            # 4) List B score
            elif line.startswith("List_B_Score"):
                try:
                    current["list_b_score"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    current["list_b_score"] = None

        # don’t forget the last song block
        if current:
            results.append(current)
       
        # Pause between requests to respect pacing
        # time.sleep(SLEEP_SECONDS)
except Exception as e:
    print(f"An error occurred: {e}")

# ── Save Results ────────────────────────────────────────────────────────────────
df_results = pd.DataFrame(results) 
if os.path.exists(OUTPUT_CSV):
    # append without writing the header row again
    df_results.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
else:
    # first time: write header
    df_results.to_csv(OUTPUT_CSV, mode="w", header=True, index=False)
print("Done! Results and indices saved.")
