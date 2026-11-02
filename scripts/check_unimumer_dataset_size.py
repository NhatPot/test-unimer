import os
import requests
import pandas as pd

DATASET = "phxember/Uni-MuMER-Data"
URL = "https://datasets-server.huggingface.co/size"

def main():
    print(f"Fetching dataset size information for: {DATASET}...")
    try:
        res = requests.get(URL, params={"dataset": DATASET}, timeout=60)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: Request failed. Details: {e}")
        if 'res' in locals() and res is not None:
            print("Response text content:")
            print(res.text)
        return

    try:
        response_json = res.json()
    except ValueError as e:
        print(f"Error: Failed to parse JSON response. Details: {e}")
        print("Response text content:")
        print(res.text)
        return

    if "size" not in response_json or "splits" not in response_json["size"]:
        print("Error: Key 'size' or 'splits' not found in response JSON.")
        print("Response JSON content for debugging:")
        import json
        print(json.dumps(response_json, indent=2))
        return

    data = response_json["size"]["splits"]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Keep only target columns and sort by config
    required_cols = ["config", "split", "num_rows"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' is missing from the API data. Filling with NaN.")
            df[col] = None
            
    df = df[required_cols].sort_values("config").reset_index(drop=True)

    print("\nDataset split information:")
    print(df.to_string(index=False))
    
    total_samples = df["num_rows"].sum()
    print(f"\nTOTAL: {total_samples}")

    # Output CSV and Markdown
    csv_file = "unimumer_sample_counts.csv"
    md_file = "unimumer_sample_counts.md"
    
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    df.to_markdown(md_file, index=False)
    
    print(f"\nSuccessfully saved CSV to: {csv_file}")
    print(f"Successfully saved Markdown to: {md_file}")

if __name__ == "__main__":
    main()
