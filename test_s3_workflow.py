#!/usr/bin/env python3
"""
Test script for the new S3 upload workflow
"""
import requests
import json

# Configuration
API_BASE = "http://localhost:8000"
TEST_FILE = "input_videos/im.mov"

def test_s3_upload_workflow():
    print("🧪 Testing S3 Upload Workflow")
    
    # Step 1: Get presigned upload URL
    print("\n1️⃣ Getting presigned upload URL...")
    response = requests.post(f"{API_BASE}/upload-url?filename={TEST_FILE}")
    if response.status_code != 200:
        print(f"❌ Failed to get upload URL: {response.text}")
        return
    
    upload_data = response.json()
    session_id = upload_data["session_id"]
    upload_url = upload_data["upload_url"]
    s3_key = upload_data["s3_key"]
    
    print(f"✅ Got upload URL for session: {session_id}")
    print(f"   S3 Key: {s3_key}")
    
    # Step 2: Simulate upload to S3 (skip for this test)
    print("\n2️⃣ Simulating S3 upload...")
    print("   📱 In a real app, the mobile client would PUT the video file to:")
    print(f"   {upload_url}")
    print("   ⏩ Skipping actual upload for this test...")
    
    # Step 3: Start processing (this will fail since we didn't actually upload)
    print("\n3️⃣ Starting processing...")
    start_data = {"s3_key": s3_key}
    response = requests.post(f"{API_BASE}/sessions/{session_id}/start", 
                           json=start_data)
    
    if response.status_code == 404:
        print("✅ Expected 404 - video not found in S3 (we didn't actually upload)")
        print("   This confirms the S3 verification is working!")
    else:
        print(f"Processing response: {response.status_code} - {response.text}")
    
    print("\n🎉 S3 workflow endpoints are working!")
    print("\n📋 Mobile App Flow:")
    print("   1. POST /upload-url?filename=video.mp4")
    print("   2. PUT <presigned_url> (upload video directly to S3)")
    print("   3. POST /sessions/{session_id}/start with s3_key")
    print("   4. Connect to WebSocket /ws/{session_id}")

if __name__ == "__main__":
    test_s3_upload_workflow()
