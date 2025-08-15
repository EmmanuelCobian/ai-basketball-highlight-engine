# Basketball Analysis API - Frontend Integration Guide

## Overview

The Basketball Analysis API provides endpoints for uploading basketball videos, processing them with AI-powered analysis, and receiving real-time updates during processing.

**Base URL:** `http://your-server:8000`

## API Workflow

```
1. POST /upload-url          → Get S3 upload URL
2. PUT <presigned_url>       → Upload video to S3
3. POST /sessions/{id}/start → Start processing  
4. WS /ws/{id}              → Receive real-time updates
```

---

## Endpoints

### 1. Get Upload URL

**`POST /upload-url`**

Get a pre-signed S3 URL for direct video upload.

#### Request Parameters

| Parameter | Type | Location | Required | Description |
|-----------|------|----------|----------|-------------|
| `filename` | string | query | Yes | Name of the video file (e.g., "basketball_game.mp4") |

#### Request Example

```http
POST /upload-url?filename=basketball_game.mp4
Content-Type: application/json
```

#### Response

**Status: 200 OK**

```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "upload_url": "https://bucket.s3.amazonaws.com/temp-uploads/session-id/video.mp4?...",
  "s3_key": "temp-uploads/123e4567-e89b-12d3-a456-426614174000/123e4567-e89b-12d3-a456-426614174000.mp4",
  "metadata": {
    "original-filename": "basketball_game.mp4",
    "upload-timestamp": "1640995200",
    "session-id": "123e4567-e89b-12d3-a456-426614174000"
  }
}
```

#### Response Schema

```json
{
  "type": "object",
  "properties": {
    "session_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique session identifier"
    },
    "upload_url": {
      "type": "string",
      "format": "uri",
      "description": "Pre-signed S3 URL for uploading video"
    },
    "s3_key": {
      "type": "string",
      "description": "S3 object key where video will be stored"
    },
    "metadata": {
      "type": "object",
      "description": "Required metadata for S3 upload"
    }
  }
}
```

#### Error Responses

| Status | Description | Response Body |
|--------|-------------|---------------|
| 500 | Failed to generate URL | `{"detail": "Failed to generate upload URL: {error}"}` |

---

### 2. Upload Video to S3

**`PUT <presigned_url>`**

Upload video file directly to S3 using the URL from step 1.

#### Request Headers

**Required headers from the metadata:**

```http
Content-Type: video/mp4
x-amz-meta-original-filename: {metadata.original-filename}
x-amz-meta-session-id: {metadata.session-id}
x-amz-meta-upload-timestamp: {metadata.upload-timestamp}
```

#### Request Body

Raw video file bytes

#### Response

**Status: 200 OK** (Empty body)

#### Error Responses

| Status | Description |
|--------|-------------|
| 403 | Signature mismatch (incorrect headers) |
| 404 | Bucket or key not found |

---

### 3. Start Processing

**`POST /sessions/{session_id}/start`**

Start video analysis after successful S3 upload.

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes | Session ID from upload URL response |

#### Request Body

```json
{
  "s3_key": "temp-uploads/123e4567-e89b-12d3-a456-426614174000/video.mp4"
}
```

#### Request Schema

```json
{
  "type": "object",
  "properties": {
    "s3_key": {
      "type": "string",
      "description": "S3 object key of the uploaded video"
    }
  },
  "required": ["s3_key"]
}
```

#### Response

**Status: 200 OK**

```json
{
  "message": "Processing started",
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### Error Responses

| Status | Description | Response Body |
|--------|-------------|---------------|
| 404 | Session not found | `{"detail": "Session not found"}` |
| 400 | S3 key mismatch | `{"detail": "S3 key mismatch"}` |
| 500 | Video not found in S3 | `{"detail": "Failed to start processing: Not Found"}` |

---

### 4. WebSocket Connection

**`WS /ws/{session_id}`**

Real-time communication during video processing.

#### Connection

```
ws://your-server:8000/ws/{session_id}
```

#### Connection Requirements

- Session must exist and be in "processing" state
- Must call `/sessions/{id}/start` before connecting

#### Incoming Message Types

##### Status Update

```json
{
  "type": "status_update",
  "frame_num": 150,
  "frame_total": 3000,
  "fps": 12.5,
  "message": "Processing frame 150/3000"
}
```

##### User Input Request

```json
{
  "type": "user_input_required",
  "input_type": "player_selection",
  "frame_num": 45,
  "data": {
    "available_players": [
      {
        "id": 1,
        "bbox": [100, 200, 150, 300],
        "center": [125, 250],
        "confidence": 0.95
      }
    ]
  }
}
```

**Input Types:**
- `player_selection`: Select which player to track
- `confirmation`: Confirm tracking decision  
- `reassignment_selection`: Choose new player when tracking fails

##### Error Message

```json
{
  "type": "error",
  "message": "Processing failed: Out of memory",
  "frame_num": 120,
  "fps": 10.2
}
```

##### Completion Message

```json
{
  "type": "completed",
  "frame_num": 3000,
  "fps": 15.8,
  "summary": {
    "total_frames": 3000,
    "acquisitions_count": 12,
    "highlights_count": 8,
    "processing_fps": 15.8,
    "highlights_file": "highlights.txt"
  }
}
```

##### Heartbeat

```json
{
  "type": "heartbeat",
  "timestamp": 1640995200.123
}
```

#### Outgoing Message Format

Send user responses as JSON:

```json
{
  "response_type": "player_selection",
  "player_id": 1
}
```

```json
{
  "response_type": "confirmation", 
  "confirmed": true
}
```

```json
{
  "response_type": "reassignment_selection",
  "player_id": 2
}
```

#### WebSocket Error Codes

| Code | Reason |
|------|--------|
| 1008 | Invalid session_id |
| 1008 | Session not in processing state |

---

### 5. Health Check

**`GET /health`**

Check API service status.

#### Response

**Status: 200 OK**

```json
{
  "status": "ok",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

---

### 6. Session Status

**`GET /sessions/{session_id}/status`**

Get current session status.

#### Response

**Status: 200 OK**

```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "processing",
  "created_at": "2024-01-01T12:00:00.000Z",
  "original_filename": "basketball_game.mp4",
  "s3_key": "temp-uploads/session-id/video.mp4"
}
```

**Status Values:**
- `created`: Session created, awaiting upload
- `processing`: Video is being analyzed
- `completed`: Analysis finished successfully  
- `error`: Processing failed

#### Error Responses

| Status | Description |
|--------|-------------|
| 404 | Session not found |

---

## Data Models

### Player Object

```json
{
  "id": 1,
  "bbox": [x1, y1, x2, y2],
  "center": [x, y], 
  "confidence": 0.95
}
```

### Processing Summary

```json
{
  "total_frames": 3000,
  "acquisitions_count": 12,
  "highlights_count": 8, 
  "processing_fps": 15.8,
  "highlights_file": "highlights.txt"
}
```

---

## Error Handling

### Standard Error Response

```json
{
  "detail": "Error description"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Session/resource doesn't exist |
| 500 | Internal Server Error - Server-side failure |

### WebSocket Errors

WebSocket errors are sent as messages with `"type": "error"` rather than connection close codes.

---

## Complete Integration Example

### 1. Frontend Flow

```javascript
// 1. Get upload URL
const uploadResponse = await fetch('/upload-url?filename=video.mp4', {
  method: 'POST'
});
const uploadData = await uploadResponse.json();

// 2. Upload to S3
const uploadResult = await fetch(uploadData.upload_url, {
  method: 'PUT',
  body: videoFile,
  headers: {
    'Content-Type': 'video/mp4',
    'x-amz-meta-original-filename': uploadData.metadata['original-filename'],
    'x-amz-meta-session-id': uploadData.metadata['session-id'],
    'x-amz-meta-upload-timestamp': uploadData.metadata['upload-timestamp']
  }
});

// 3. Start processing
await fetch(`/sessions/${uploadData.session_id}/start`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ s3_key: uploadData.s3_key })
});

// 4. Connect WebSocket
const ws = new WebSocket(`ws://server:8000/ws/${uploadData.session_id}`);
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle different message types
};
```

### 2. Key Implementation Notes

- **Always include exact metadata headers** when uploading to S3
- **Handle WebSocket reconnection** for network interruptions
- **Implement user input UI** for player selection prompts
- **Show progress indicators** using frame_num/frame_total
- **Handle timeouts** - user has 5 minutes to respond to input requests

---

## Testing

Use the provided test endpoints:

- **Health check:** `GET /health`
- **API info:** `GET /`
- **Session status:** `GET /sessions/{id}/status`

For testing without video upload, use small test files (< 10MB) to avoid timeout issues on smaller servers.
