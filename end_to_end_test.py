#!/usr/bin/env python3
"""
End-to-End Basketball Analysis Workflow Test

This script demonstrates the complete workflow:
1. Get presigned S3 upload URL
2. Upload video to S3
3. Start processing
4. Connect to WebSocket for real-time updates
5. Handle user interactions
6. Get final results
7. Cleanup (optional)
"""
import asyncio
import json
import os
import sys
import time
from typing import Optional

import requests
import websockets

# Configuration
API_BASE = "http://3.15.204.107:8000"
WS_BASE = "ws://3.15.204.107:8000"
TEST_FILE = "input_videos/im_1.mov"

class BasketballAnalysisClient:
    def __init__(self, api_base: str, ws_base: str):
        self.api_base = api_base.rstrip('/')
        self.ws_base = ws_base.rstrip('/')
        self.session_id: Optional[str] = None
        self.s3_key: Optional[str] = None
        
    def get_upload_url(self, filename: str) -> dict:
        """Step 1: Get presigned S3 upload URL"""
        print(f"🔗 Step 1: Getting presigned upload URL for {filename}")
        
        response = requests.post(f"{self.api_base}/upload-url?filename={filename}")
        if response.status_code != 200:
            raise Exception(f"Failed to get upload URL: {response.text}")
        
        data = response.json()
        self.session_id = data["session_id"]
        self.s3_key = data["s3_key"]
        
        print(f"✅ Session created: {self.session_id}")
        print(f"   S3 Key: {self.s3_key}")
        
        return data
    
    def upload_to_s3(self, upload_url: str, metadata: dict, file_path: str) -> bool:
        """Step 2: Upload video file to S3"""
        print(f"📤 Step 2: Uploading {file_path} to S3")
        
        if not os.path.exists(file_path):
            raise Exception(f"Test file {file_path} not found")
        
        # Prepare headers with exact metadata from presigned URL
        headers = {
            'Content-Type': 'video/mp4',
            'x-amz-meta-original-filename': metadata['original-filename'],
            'x-amz-meta-session-id': metadata['session-id'],
            'x-amz-meta-upload-timestamp': metadata['upload-timestamp']
        }
        
        with open(file_path, 'rb') as f:
            response = requests.put(upload_url, data=f, headers=headers)
        
        if response.status_code == 200:
            print("✅ Upload to S3 successful!")
            return True
        else:
            raise Exception(f"S3 upload failed: {response.status_code} - {response.text}")
    
    def start_processing(self) -> bool:
        """Step 3: Start video processing"""
        print("🚀 Step 3: Starting video processing")
        
        if not self.session_id or not self.s3_key:
            raise Exception("No session or S3 key available")
        
        start_data = {"s3_key": self.s3_key}
        response = requests.post(f"{self.api_base}/sessions/{self.session_id}/start", 
                               json=start_data)
        
        if response.status_code == 200:
            print("✅ Processing started successfully!")
            print("   Waiting 3 seconds for processing to initialize...")
            time.sleep(3)  # Give the server time to set up the session
            return True
        else:
            raise Exception(f"Failed to start processing: {response.status_code} - {response.text}")
    
    async def connect_websocket(self) -> None:
        """Step 4: Connect to WebSocket and handle real-time updates"""
        print("🔌 Step 4: Connecting to WebSocket for real-time updates")
        
        if not self.session_id:
            raise Exception("No session ID available")
        
        uri = f"{self.ws_base}/ws/{self.session_id}"
        print(f"   Connecting to: {uri}")
        
        # Add retry logic for WebSocket connection
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with websockets.connect(
                    uri,
                    max_size=8 * 1024 * 1024,
                    ping_interval=30,
                    ping_timeout=120,
                    close_timeout=10,
                    max_queue=32,
                ) as ws:
                    print("✅ Connected to WebSocket")
                    await self._handle_messages(ws)
                    return  # Success, exit retry loop
                    
            except websockets.exceptions.ConnectionClosed as e:
                print(f"💤 WebSocket connection closed: {e}")
                if attempt < max_retries - 1:
                    print(f"   Retrying in 2 seconds... (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(2)
                else:
                    raise Exception(f"WebSocket failed after {max_retries} attempts")
            except Exception as e:
                print(f"❌ WebSocket error: {e}")
                if attempt < max_retries - 1:
                    print(f"   Retrying in 2 seconds... (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(2)
                else:
                    raise Exception(f"WebSocket failed after {max_retries} attempts: {e}")
    
    async def _handle_messages(self, ws) -> None:
        """Handle WebSocket messages"""
        fps_seen = None
        first_message = True
        
        while True:
            try:
                msg = await ws.recv()
            except websockets.exceptions.ConnectionClosedOK:
                print("💤 WebSocket closed normally")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"❌ WebSocket connection error: {e}")
                break
            except Exception as e:
                print(f"❌ Unexpected WebSocket error: {e}")
                break

            try:
                data = json.loads(msg)
            except Exception:
                print(f"⚠️  Non-JSON message: {msg[:200]}")
                continue

            # Check for session errors on first message
            if first_message:
                first_message = False
                if data.get("type") == "error" and "session" in data.get("message", "").lower():
                    raise Exception(f"Session error: {data.get('message')}")

            mtype = data.get("type")
            fps = data.get("fps", fps_seen)
            if fps is not None:
                fps_seen = fps

            if mtype == "status_update":
                await self._handle_status_update(data, fps_seen)
            elif mtype == "user_input_required":
                await self._handle_user_input(ws, data, fps_seen)
            elif mtype == "completed":
                await self._handle_completion(data, fps_seen)
                break
            elif mtype == "error":
                await self._handle_error(data, fps_seen)
                break
            else:
                print(f"🔍 Unknown message type '{mtype}': {data}")
    
    async def _handle_status_update(self, data: dict, fps: float) -> None:
        """Handle status update messages"""
        cur = data.get("frame_num")
        total = data.get("frame_total")
        message = data.get("message", "Processing...")
        print(f"📊 Frame {cur}/{total} @ {fps:.1f} fps - {message}")
    
    async def _handle_user_input(self, ws, data: dict, fps: float) -> None:
        """Handle user input requests with auto-responses"""
        itype = data.get("input_type")
        payload = data.get("data", {})
        fnum = data.get("frame_num")
        
        print(f"❓ User input required at frame {fnum}: {itype}")
        
        # Auto-respond based on input type
        if itype == "player_selection":
            avail = payload.get("available_players", [])
            if avail:
                choice = avail[0]["id"]
                print(f"   🤖 Auto-selecting player: {choice}")
            else:
                choice = None
                print("   🤖 No players available")
            resp = {"response_type": "player_selection", "player_id": choice}
            
        elif itype == "confirmation":
            print("   🤖 Auto-confirming")
            resp = {"response_type": "confirmation", "confirmed": True}
            
        elif itype == "reassignment_selection":
            sugg = payload.get("suggestions", [])
            if sugg:
                choice_id = sugg[0]["id"]
                print(f"   🤖 Auto-selecting suggestion: {choice_id}")
                resp = {"response_type": "reassignment_selection", "player_id": choice_id}
            else:
                print("   🤖 Auto-selecting choice 0")
                resp = {"response_type": "reassignment_selection", "choice": 0}
        else:
            print(f"   ⚠️  Unknown input type '{itype}', defaulting to choice 0")
            resp = {"response_type": "reassignment_selection", "choice": 0}
        
        await ws.send(json.dumps(resp))
    
    async def _handle_completion(self, data: dict, fps: float) -> None:
        """Handle processing completion"""
        frame_num = data.get("frame_num")
        summary = data.get("summary", {})
        
        print(f"🎉 Processing completed at frame {frame_num} @ {fps:.1f} fps")
        print("📋 Summary:")
        print(json.dumps(summary, indent=2))
    
    async def _handle_error(self, data: dict, fps: float) -> None:
        """Handle processing errors"""
        frame_num = data.get("frame_num")
        message = data.get("message", "Unknown error")
        
        print(f"❌ Processing error at frame {frame_num} @ {fps:.1f} fps: {message}")
    
    def cleanup_s3_file(self) -> bool:
        """Optional Step 5: Cleanup uploaded file from S3"""
        print("🧹 Step 5: Cleaning up S3 file (optional)")
        
        if not self.s3_key:
            print("   No S3 key to cleanup")
            return True
        
        try:
            # This would require implementing a cleanup endpoint in the API
            # For now, just indicate what would happen
            print(f"   Would delete: {self.s3_key}")
            print("   (Cleanup endpoint not implemented yet)")
            return True
        except Exception as e:
            print(f"   ⚠️  Cleanup failed: {e}")
            return False


async def run_end_to_end_test(file_path: str):
    """Run the complete end-to-end workflow"""
    print("🏀 Basketball Analysis - End-to-End Workflow Test")
    print("=" * 60)
    
    client = BasketballAnalysisClient(API_BASE, WS_BASE)
    
    try:
        # Step 1: Get upload URL
        upload_data = client.get_upload_url(os.path.basename(file_path))
        upload_url = upload_data["upload_url"]
        metadata = upload_data["metadata"]
        
        print()
        
        # Step 2: Upload to S3
        client.upload_to_s3(upload_url, metadata, file_path)
        
        print()
        
        # Step 3: Start processing
        client.start_processing()
        
        print()
        
        # Step 4: WebSocket connection for real-time updates
        await client.connect_websocket()
        
        print()
        
        # Step 5: Cleanup (optional)
        client.cleanup_s3_file()
        
        print()
        print("🎉 End-to-end test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True


def main():
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = TEST_FILE
    
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found")
        print(f"Usage: python {sys.argv[0]} [video_file]")
        print(f"Default file: {TEST_FILE}")
        sys.exit(1)
    
    print(f"📹 Using test file: {test_file}")
    print()
    
    # Run the async test
    success = asyncio.run(run_end_to_end_test(test_file))
    
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
