#!/usr/bin/env python3
"""
Simple test script for Basia VLM server
Tests basic text and vision capabilities
"""

import ollama
import time
import sys
import base64
import os

def test_connection():
    """Test basic connection to Ollama server"""
    print("Testing connection to Ollama server...")
    try:
        client = ollama.Client(host='http://localhost:11434')
        
        # Simple text test
        print("Running text test...")
        start_time = time.time()
        
        response = client.chat(
            model='llama3.2-vision:11b',
            messages=[{
                'role': 'user',
                'content': 'Hello! Can you help with fluorescence microscopy?'
            }]
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"Connection successful!")
        print(f"Response time: {response_time:.2f} seconds")
        print(f"Response: {response['message']['content'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def test_vision():
    """Test vision capabilities with actual image"""
    print("\nTesting vision capabilities...")
    
    # Check if test.jpg exists
    if not os.path.exists('test.jpg'):
        print("Warning: test.jpg not found, running text-only vision test...")
        return test_vision_text_only()
    
    try:
        client = ollama.Client(host='http://localhost:11434')
        
        # Load and encode image
        print("Loading test.jpg...")
        with open('test.jpg', 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        start_time = time.time()
        
        response = client.chat(
            model='llama3.2-vision:11b',
            messages=[{
                'role': 'user',
                'content': 'Analyze this image. Describe what you see and suggest if this could be useful for fluorescence microscopy analysis.',
                'images': [image_data]
            }]
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"Vision analysis complete!")
        print(f"Response time: {response_time:.2f} seconds")
        print(f"Response: {response['message']['content'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"Vision test failed: {e}")
        return False

def test_vision_text_only():
    """Fallback vision test without image"""
    try:
        client = ollama.Client(host='http://localhost:11434')
        
        start_time = time.time()
        
        response = client.chat(
            model='llama3.2-vision:11b',
            messages=[{
                'role': 'user',
                'content': 'Describe what you would expect to see in a fluorescence microscopy image of cells with GFP-labeled mitochondria.'
            }]
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"Vision model responding (text-only)!")
        print(f"Response time: {response_time:.2f} seconds")
        print(f"Response: {response['message']['content'][:150]}...")
        
        return True
        
    except Exception as e:
        print(f"Vision test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Basia VLM Test Script")
    print("=" * 50)
    
    # Test 1: Basic connection
    connection_ok = test_connection()
    
    if not connection_ok:
        print("\nBasic connection failed. Check if Ollama is running:")
        print("  1. Run 'ollama serve' in another terminal")
        print("  2. Ensure llama3.2-vision:11b is installed")
        sys.exit(1)
    
    # Test 2: Vision capabilities
    vision_ok = test_vision()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Connection: {'PASS' if connection_ok else 'FAIL'}")
    print(f"Vision:     {'PASS' if vision_ok else 'FAIL'}")
    
    if connection_ok and vision_ok:
        print("\nAll tests passed! VLM is ready for microscopy applications.")
    else:
        print("\nSome tests failed. Check Ollama setup.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()