import requests
import time
import os
import sys

BASE_URL = "http://localhost:8000/api/v1"
TEST_FILE = "tests/dummy_data.csv"

def create_dummy_csv():
    """Create a simple dummy CSV for testing."""
    with open(TEST_FILE, "w") as f:
        f.write("age,salary,city,purchased\n")
        f.write("25,50000,New York,Yes\n")
        f.write("30,60000,London,No\n")
        f.write(",70000,New York,Yes\n")  # Missing age
        f.write("35,,Paris,No\n")        # Missing salary
        f.write("40,80000,,Yes\n")       # Missing city

def wait_for_server():
    """Wait for server to be ready."""
    print("⏳ Waiting for server...", end="", flush=True)
    for _ in range(10):
        try:
            requests.get(f"{BASE_URL}/health")
            print(" ✅ Ready!")
            return True
        except:
            time.sleep(1)
            print(".", end="", flush=True)
    print(" ❌ Server not reachable.")
    return False

def test_workflow():
    if not wait_for_server():
        return

    # 1. Upload
    print("\n[TEST 1] Uploading Dataset...")
    if not os.path.exists(TEST_FILE):
        create_dummy_csv()
    
    with open(TEST_FILE, "rb") as f:
        response = requests.post(f"{BASE_URL}/upload", files={"file": f})
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return
        
    dataset_id = response.json()["dataset_id"]
    print(f"✅ Upload Success. ID: {dataset_id}")

    # 2. Run Agent
    print(f"\n[TEST 2] Running Agent (dataset_id={dataset_id})...")
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/agent/run", json={"dataset_id": dataset_id}, timeout=60)
    
    if response.status_code != 200:
        print(f"❌ Agent Run failed: {response.text}")
        return

    result = response.json()
    elapsed = time.time() - start_time
    
    print(f"✅ Agent Completed in {elapsed:.2f}s")
    print(f"   Status: {result.get('status')}")
    print(f"   Steps: {result.get('step_count')}")
    print(f"   Last Error: {result.get('last_error')}")
    
    # 3. Analyze History
    print("\n[TEST 3] Validating Agent Logic...")
    messages = result.get("recent_history", [])
    if not messages:
        print("⚠️ No messages returned (Agent might have failed silently?)")
    else:
        has_thought = any("thought" in str(m) for m in messages)
        has_action = any("action" in str(m) for m in messages)
        if has_thought and has_action:
            print("✅ Agent observed, reasoned, and acted.")
        else:
            print("⚠️ Agent history looks incomplete.")
            
    # 4. Negative Test
    print("\n[TEST 4] Negative Test (Invalid ID)...")
    res_bad = requests.post(f"{BASE_URL}/agent/run", json={"dataset_id": "bad-id"})
    if res_bad.status_code == 404:
        print("✅ Correctly rejected invalid ID.")
    else:
        print(f"❌ Expected 404, got {res_bad.status_code}")

if __name__ == "__main__":
    test_workflow()
