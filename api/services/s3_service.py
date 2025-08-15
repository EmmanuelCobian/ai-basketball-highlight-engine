"""
S3 service for handling video uploads and downloads.

This module provides all S3-related functionality including:
- Pre-signed URL generation
- File upload/download operations
- S3 client management
"""
import os
import tempfile
import time
import uuid
from typing import Dict, Tuple
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException

from api.config import S3_BUCKET, AWS_REGION


class S3Service:
    """Service for handling S3 operations."""
    
    def __init__(self):
        """Initialize S3 service with boto3 client."""
        self._s3_client = None
    
    @property
    def s3_client(self):
        """Lazy-loaded S3 client."""
        if self._s3_client is None:
            self._s3_client = boto3.client('s3', region_name=AWS_REGION)
        return self._s3_client
    
    def generate_upload_url(self, filename: str) -> Tuple[str, str, str, Dict[str, str]]:
        """
        Generate a pre-signed URL for video upload.
        
        Args:
            filename: Original filename of the video
            
        Returns:
            Tuple containing (session_id, upload_url, s3_key, metadata)
            
        Raises:
            HTTPException: If URL generation fails
        """
        try:
            session_id = str(uuid.uuid4())
            file_extension = os.path.splitext(filename)[1].lower()
            s3_key = f"temp-uploads/{session_id}/{session_id}{file_extension}"
            metadata = {
                'original-filename': filename,
                'upload-timestamp': str(int(time.time())),
                'session-id': session_id
            }
            
            presigned_url = self.s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': S3_BUCKET,
                    'Key': s3_key,
                    'ContentType': 'video/mp4',
                    'Metadata': metadata
                },
                ExpiresIn=3600
            )
            
            return session_id, presigned_url, s3_key, metadata
            
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to generate upload URL: {str(e)}"
            )
    
    def download_video_to_temp(self, s3_key: str) -> str:
        """
        Download video from S3 to temporary local file.
        
        Args:
            s3_key: S3 object key of the video
            
        Returns:
            Path to temporary local file
            
        Raises:
            HTTPException: If download fails or file doesn't exist
        """
        try:
            self.s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_path = temp_file.name
            temp_file.close()
            
            self.s3_client.download_file(S3_BUCKET, s3_key, temp_path)
            return temp_path
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise HTTPException(
                    status_code=404, 
                    detail=f"Video not found in S3: {s3_key}"
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to download video: {str(e)}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to download video: {str(e)}"
            )
    
    def cleanup_temp_file(self, file_path: str) -> None:
        """
        Remove temporary file safely.
        
        Args:
            file_path: Path to temporary file to remove
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            # Log error but don't raise - cleanup failures shouldn't break the flow
            print(f"Warning: Failed to cleanup temp file {file_path}: {e}")
    
    def delete_s3_object(self, s3_key: str) -> None:
        """
        Delete object from S3.
        
        Args:
            s3_key: S3 object key to delete
            
        Raises:
            HTTPException: If deletion fails
        """
        try:
            self.s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete S3 object: {str(e)}"
            )

    def cleanup_s3_object(self, s3_key: str) -> bool:
        """
        Clean up S3 object during processing cleanup.
        Safe version that doesn't raise HTTPException.
        
        Args:
            s3_key: S3 object key to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            print(f"S3 cleanup: Attempting to delete {s3_key} from bucket {S3_BUCKET}")
            response = self.s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)
            print(f"S3 cleanup: Delete response: {response}")
            return True
        except Exception as e:
            print(f"S3 cleanup failed for {s3_key}: {e}")
            print(f"Exception type: {type(e).__name__}")
            return False


s3_service = S3Service()
